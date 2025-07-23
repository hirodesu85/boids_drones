# -*- coding: utf-8 -*-
"""
Boid Flocking Controller for Crazyflie Drones

This controller implements boid algorithm for autonomous flocking behavior
with communication between drones using emitter/receiver.
"""

import sys
import json
import time
import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from boid_algorithm import BoidAlgorithm

# Simulation configuration
TIME_STEP = 16
HOVER_ALTITUDE = 1.0
TAKEOFF_SPEED = 0.3

class BoidFlockingController:
    def __init__(self):
        self.robot = Robot()
        
        # Get drone ID from command line arguments
        if len(sys.argv) > 1:
            self.drone_id = int(sys.argv[1])
        else:
            self.drone_id = 1
            
        print(f"=== Drone {self.drone_id} Starting Boid Flocking ===")
        
        # Initialize motors for quadcopter
        self.m1_motor = self.robot.getDevice("m1_motor")
        self.m1_motor.setPosition(float("inf"))
        self.m1_motor.setVelocity(-1)
        
        self.m2_motor = self.robot.getDevice("m2_motor")
        self.m2_motor.setPosition(float("inf"))
        self.m2_motor.setVelocity(1)
        
        self.m3_motor = self.robot.getDevice("m3_motor")
        self.m3_motor.setPosition(float("inf"))
        self.m3_motor.setVelocity(-1)
        
        self.m4_motor = self.robot.getDevice("m4_motor")
        self.m4_motor.setPosition(float("inf"))
        self.m4_motor.setVelocity(1)
        
        # Initialize sensors
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(TIME_STEP)
        
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        
        # Initialize communication devices
        try:
            self.emitter = self.robot.getDevice("emitter")
            if self.emitter is None:
                print(f"Drone {self.drone_id}: ERROR - Emitter device not found!")
            else:
                pass  # Emitter initialized successfully
        except Exception as e:
            print(f"Drone {self.drone_id}: ERROR initializing emitter: {e}")
            self.emitter = None
            
        try:
            self.receiver = self.robot.getDevice("receiver")
            if self.receiver is None:
                print(f"Drone {self.drone_id}: ERROR - Receiver device not found!")
            else:
                self.receiver.enable(TIME_STEP)
        except Exception as e:
            print(f"Drone {self.drone_id}: ERROR initializing receiver: {e}")
            self.receiver = None
        
        # Initialize control systems
        self.pid_controller = pid_velocity_fixed_height_controller()
        self.boid_algorithm = BoidAlgorithm()
        
        # State variables
        self.past_time = self.robot.getTime()
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        self.current_velocity = np.array([0.0, 0.0])
        
        # Low-pass filter for velocity (smoothing)
        self.velocity_filter_alpha = 0.7  # 0.7 = 70% current, 30% past
        self.filtered_velocity = np.array([0.0, 0.0])
        
        # Acceleration limiting
        self.max_acceleration = 0.5  # m/s^2
        self.last_desired_velocity = np.array([0.0, 0.0])
        
        # Communication variables
        self.last_send_time = 0.0
        self.send_interval = 0.1  # Send data every 100ms for better responsiveness
        self.neighbors = {}  # Dictionary to store neighbor information
        self.neighbor_timeout = 1.0  # Remove neighbors not heard from in 1 second
        
        # Flight state
        self.takeoff_complete = False
        self.flight_mode = "takeoff"  # takeoff, hovering, flocking
        
        # Leader/Follower configuration
        self.is_leader = (self.drone_id == 1)  # Drone 1 is the leader
        self.wander_angle = 0.0  # Leader's exploration direction
        self.wander_change_rate = 0.3  # How fast the direction changes

    def send_drone_state(self):
        """Send current drone state to other drones"""
        if self.emitter is None:
            return
            
        current_time = self.robot.getTime()
        
        if current_time - self.last_send_time < self.send_interval:
            return
            
        # Get current position and velocity
        position = self.gps.getValues()
        
        # Create message with position, velocity, and flight state
        message = {
            "drone_id": self.drone_id,
            "time": current_time,
            "x": position[0],
            "y": position[1], 
            "z": position[2],
            "vx": self.current_velocity[0],
            "vy": self.current_velocity[1],
            "flight_mode": self.flight_mode,
            "status": "active"
        }
        
        # Send as JSON string
        try:
            message_str = json.dumps(message)
            self.emitter.send(message_str)
            self.last_send_time = current_time
        except Exception as e:
            print(f"Drone {self.drone_id}: Send error: {e}")

    def receive_neighbor_data(self):
        """Receive and process data from other drones"""
        if self.receiver is None:
            return
            
        current_time = self.robot.getTime()
        
        while self.receiver.getQueueLength() > 0:
            try:
                message_str = self.receiver.getString()
                message = json.loads(message_str)
                sender_id = message["drone_id"]
                
                # Ignore messages from self
                if sender_id != self.drone_id:
                    # Add timestamp for timeout checking
                    message["received_time"] = current_time
                    self.neighbors[sender_id] = message
                    
            except json.JSONDecodeError as e:
                print(f"Drone {self.drone_id}: JSON decode error: {e}")
            except Exception as e:
                print(f"Drone {self.drone_id}: Receive error: {e}")
                
            self.receiver.nextPacket()
        
        # Remove outdated neighbor data
        expired_neighbors = []
        for neighbor_id, neighbor_data in self.neighbors.items():
            if current_time - neighbor_data.get("received_time", 0) > self.neighbor_timeout:
                expired_neighbors.append(neighbor_id)
        
        for neighbor_id in expired_neighbors:
            del self.neighbors[neighbor_id]

    def calculate_velocity_from_gps(self, dt):
        """Calculate current velocity from GPS readings"""
        if dt <= 0:
            return
            
        current_x_global = self.gps.getValues()[0]
        current_y_global = self.gps.getValues()[1]
        
        # Skip velocity calculation on first frame
        if self.past_x_global == 0.0 and self.past_y_global == 0.0:
            self.past_x_global = current_x_global
            self.past_y_global = current_y_global
            return
            
        # Calculate global velocity
        vx_global = (current_x_global - self.past_x_global) / dt
        vy_global = (current_y_global - self.past_y_global) / dt
        
        # Convert to body frame
        current_yaw = self.imu.getRollPitchYaw()[2]
        cos_yaw = np.cos(current_yaw)
        sin_yaw = np.sin(current_yaw)
        
        v_x = vx_global * cos_yaw + vy_global * sin_yaw
        v_y = -vx_global * sin_yaw + vy_global * cos_yaw
        
        # Apply low-pass filter to smooth velocity
        raw_velocity = np.array([v_x, v_y])
        self.filtered_velocity = (self.velocity_filter_alpha * raw_velocity + 
                                 (1 - self.velocity_filter_alpha) * self.filtered_velocity)
        
        # Limit velocity to prevent extreme values
        velocity_magnitude = np.linalg.norm(self.filtered_velocity)
        if velocity_magnitude > 2.0:  # Max 2 m/s
            self.filtered_velocity = (self.filtered_velocity / velocity_magnitude) * 2.0
        
        self.current_velocity = self.filtered_velocity
        
        # Update past values
        self.past_x_global = current_x_global
        self.past_y_global = current_y_global

    def get_boid_velocity(self):
        """Calculate desired velocity using boid algorithm"""
        if not self.neighbors:
            # No neighbors, maintain gentle hover
            return np.array([0.0, 0.0])
        
        # Prepare current state
        position = self.gps.getValues()
        neighbors_list = list(self.neighbors.values())
        
        # Filter neighbors by distance (only consider nearby drones)
        nearby_neighbors = self.boid_algorithm.filter_neighbors_by_distance(
            position, neighbors_list, self.boid_algorithm.cohesion_distance
        )
        
        if not nearby_neighbors:
            return np.array([0.0, 0.0])
        
        # Calculate boid velocity
        desired_velocity = self.boid_algorithm.calculate_boid_velocity(
            position, self.current_velocity, nearby_neighbors
        )
        
        return desired_velocity

    def update_flight_mode(self):
        """Update flight mode based on current state"""
        altitude = self.gps.getValues()[2]
        
        if self.flight_mode == "takeoff":
            if altitude >= HOVER_ALTITUDE - 0.1:
                self.flight_mode = "hovering"
                print(f"Drone {self.drone_id}: Takeoff complete, switching to hovering")
                
        elif self.flight_mode == "hovering":
            # Wait a bit before starting flocking to let all drones reach altitude
            if self.robot.getTime() > 5.0 and len(self.neighbors) > 0:
                self.flight_mode = "flocking"
                print(f"Drone {self.drone_id}: Starting flocking behavior")

    def get_leader_velocity(self):
        """Generate exploration behavior for the leader drone"""
        # Randomly change direction gradually
        self.wander_angle += np.random.uniform(-self.wander_change_rate, self.wander_change_rate)
        
        # Base speed for exploration
        base_speed = 0.3
        
        # Calculate velocity vector
        velocity = np.array([
            np.cos(self.wander_angle) * base_speed,
            np.sin(self.wander_angle) * base_speed
        ])
        
        # Boundary avoidance - keep within field limits
        position = self.gps.getValues()
        boundary_limit = 15.0
        
        # Soft boundary repulsion
        if abs(position[0]) > boundary_limit:
            velocity[0] -= np.sign(position[0]) * 0.2
        if abs(position[1]) > boundary_limit:
            velocity[1] -= np.sign(position[1]) * 0.2
            
        return velocity

    def get_desired_velocity(self):
        """Get desired velocity based on current flight mode"""
        if self.flight_mode == "takeoff" or self.flight_mode == "hovering":
            # Simple hover in place
            return np.array([0.0, 0.0])
            
        elif self.flight_mode == "flocking":
            if self.is_leader:
                # Leader explores randomly
                return self.get_leader_velocity()
            else:
                # Followers use boid algorithm
                return self.get_boid_velocity()
            
        return np.array([0.0, 0.0])

    def run(self):
        """Main control loop"""
        # Wait for sensors to initialize
        init_steps = 0
        while init_steps < 10 and self.robot.step(TIME_STEP) != -1:
            init_steps += 1
        
        # Initialize GPS readings after sensors are ready
        self.past_x_global = self.gps.getValues()[0]
        self.past_y_global = self.gps.getValues()[1]
        self.past_time = self.robot.getTime()
        
        while self.robot.step(TIME_STEP) != -1:
            dt = self.robot.getTime() - self.past_time
            
            if dt <= 0:
                self.past_time = self.robot.getTime()
                continue  # Skip if no time has passed
            
            # Update velocity calculation
            self.calculate_velocity_from_gps(dt)
            
            # Communication
            self.send_drone_state()
            self.receive_neighbor_data()
            
            # Update flight mode
            self.update_flight_mode()
            
            # Get sensor readings with validation
            imu_values = self.imu.getRollPitchYaw()
            gyro_values = self.gyro.getValues()
            gps_values = self.gps.getValues()
            
            # Check for valid sensor readings
            if any(np.isnan(imu_values)) or any(np.isnan(gyro_values)) or any(np.isnan(gps_values)):
                self.past_time = self.robot.getTime()
                continue
                
            roll = imu_values[0]
            pitch = imu_values[1]
            yaw_rate = gyro_values[2]
            altitude = gps_values[2]
            
            # Get desired velocity from current flight mode
            desired_velocity = self.get_desired_velocity()
            
            # Safety check: limit extreme attitudes
            max_attitude = 0.5  # radians (~28 degrees)
            if abs(roll) > max_attitude or abs(pitch) > max_attitude:
                print(f"Drone {self.drone_id}: WARNING - Extreme attitude detected! Roll={roll:.2f}, Pitch={pitch:.2f}")
                # Emergency: reduce desired velocity
                desired_velocity = desired_velocity * 0.3
            
            # Apply acceleration limiting for smoother changes
            velocity_change = desired_velocity - self.last_desired_velocity
            max_change = self.max_acceleration * dt
            
            if np.linalg.norm(velocity_change) > max_change:
                velocity_change = (velocity_change / np.linalg.norm(velocity_change)) * max_change
                desired_velocity = self.last_desired_velocity + velocity_change
            
            self.last_desired_velocity = desired_velocity
            
            # Use PID controller to convert desired velocity to motor commands
            motor_power = self.pid_controller.pid(
                dt,
                desired_velocity[0],  # desired_vx
                desired_velocity[1],  # desired_vy
                0,                   # desired_yaw_rate
                HOVER_ALTITUDE,      # desired_altitude
                roll,
                pitch,
                yaw_rate,
                altitude,
                self.current_velocity[0],  # actual_vx
                self.current_velocity[1]   # actual_vy
            )
            
            # Check for NaN before applying motor commands
            if any(np.isnan(motor_power)):
                # Use safe default values
                motor_power = [0, 0, 0, 0]
            
            # Apply motor commands
            self.m1_motor.setVelocity(-motor_power[0])
            self.m2_motor.setVelocity(motor_power[1])
            self.m3_motor.setVelocity(-motor_power[2])
            self.m4_motor.setVelocity(motor_power[3])
            
            # Debug output every 2 seconds
            if int(self.robot.getTime()) % 2 == 0 and int(self.robot.getTime() * 10) % 10 == 0:
                position = self.gps.getValues()
                print(f"Drone {self.drone_id}: Mode={self.flight_mode}, Pos=({position[0]:.2f},{position[1]:.2f},{position[2]:.2f}), "
                      f"Vel=({self.current_velocity[0]:.2f},{self.current_velocity[1]:.2f}), Neighbors={len(self.neighbors)}")
            
            self.past_time = self.robot.getTime()


if __name__ == "__main__":
    controller = BoidFlockingController()
    controller.run()