# -*- coding: utf-8 -*-
"""
Boid Flocking Controller for Crazyflie Drones (Fully 3D Boids Control)
"""

import sys
import json
import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from boid_algorithm import BoidAlgorithm

# --- 設定 ---
TIME_STEP = 16
INITIAL_HOVER_ALTITUDE = 1.0
MINIMUM_ALTITUDE = 0.5  # 安全のための最低高度


class BoidFlockingController:
    def __init__(self):
        self.robot = Robot()
        self.drone_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        print(f"=== Drone {self.drone_id} Starting Fully 3D Boids Control ===")

        # 初期化処理
        self.motors = self._init_motors()
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(TIME_STEP)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        if self.receiver:
            self.receiver.enable(TIME_STEP)

        self.pid_controller = pid_velocity_fixed_height_controller()
        self.boid_algorithm = BoidAlgorithm()

        # 状態変数
        self.past_time = self.robot.getTime()
        self.past_position = np.zeros(3)
        self.current_velocity = np.zeros(3)

        # ★★★ 3次元ベクトルとして平滑化変数を初期化 ★★★
        self.smoothed_desired_velocity = np.zeros(3)

        self.neighbors = {}
        self.last_send_time = 0.0
        self.flight_mode = "takeoff"
        self.desired_altitude = INITIAL_HOVER_ALTITUDE
        self.is_leader = self.drone_id == 1

    def _init_motors(self):
        motors = []
        for name in ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]:
            motor = self.robot.getDevice(name)
            motor.setPosition(float("inf"))
            velocity_sign = 1 if "m2" in name or "m4" in name else -1
            motor.setVelocity(velocity_sign)
            motors.append(motor)
        return motors

    def run(self):
        """メイン制御ループ"""
        while self.robot.step(TIME_STEP) != -1 and self.robot.getTime() < 0.5:
            pass
        self.past_position = self.gps.getValues()
        self.past_time = self.robot.getTime()

        while self.robot.step(TIME_STEP) != -1:
            dt = self.robot.getTime() - self.past_time
            if dt <= 0:
                continue

            current_pos = np.array(self.gps.getValues())
            self._update_velocity(current_pos, dt)
            self._update_flight_mode(current_pos[2])

            self._send_state(current_pos)
            self._receive_state()

            desired_vx, desired_vy = 0, 0

            if self.flight_mode == "flocking" and not self.is_leader:
                all_neighbors = list(self.neighbors.values())
                if all_neighbors:
                    # Boidsアルゴリズムで3次元の目標速度ベクトルを計算
                    boid_vel = self.boid_algorithm.calculate_boid_velocity(
                        current_pos, self.current_velocity, all_neighbors, self.is_leader
                    )

                    # ★★★ 3次元ベクトル全体を平滑化 ★★★
                    alpha = 0.2
                    self.smoothed_desired_velocity = alpha * boid_vel + (1 - alpha) * self.smoothed_desired_velocity

                    # 平滑化された速度のXY成分を目標値とする
                    desired_vx, desired_vy = self.smoothed_desired_velocity[0], self.smoothed_desired_velocity[1]

                    # ★★★ 平滑化された速度のZ成分で目標高度を更新 ★★★
                    # これにより、Boidsの力が高度にも反映される
                    self.desired_altitude += self.smoothed_desired_velocity[2] * dt

            # 安全のため、目標高度は最低高度を下回らないようにする
            self.desired_altitude = max(MINIMUM_ALTITUDE, self.desired_altitude)

            # PID制御
            roll, pitch, _ = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]
            motor_power = self.pid_controller.pid(
                dt,
                desired_vx,
                desired_vy,
                0,
                self.desired_altitude,
                roll,
                pitch,
                yaw_rate,
                current_pos[2],
                self.current_velocity[0],
                self.current_velocity[1],
            )

            # モーター出力
            self.motors[0].setVelocity(-motor_power[0])
            self.motors[1].setVelocity(motor_power[1])
            self.motors[2].setVelocity(-motor_power[2])
            self.motors[3].setVelocity(motor_power[3])

            self.past_time = self.robot.getTime()
            self.past_position = current_pos

    # (以降の補助関数は変更なし)
    def _update_velocity(self, current_pos, dt):
        self.current_velocity = (current_pos - self.past_position) / dt

    def _update_flight_mode(self, altitude):
        if self.flight_mode == "takeoff" and altitude >= INITIAL_HOVER_ALTITUDE - 0.1:
            self.flight_mode = "flocking"
            print(f"Drone {self.drone_id}: Takeoff complete. Switching to 3D Flocking.")

    def _send_state(self, pos):
        if self.robot.getTime() - self.last_send_time < 0.1:
            return
        message = {
            "drone_id": self.drone_id,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "vx": self.current_velocity[0],
            "vy": self.current_velocity[1],
            "vz": self.current_velocity[2],
        }
        if self.emitter:
            try:
                self.emitter.send(json.dumps(message))
                self.last_send_time = self.robot.getTime()
            except Exception:
                pass

    def _receive_state(self):
        if not self.receiver:
            return
        while self.receiver.getQueueLength() > 0:
            try:
                data = json.loads(self.receiver.getString())
                if data["drone_id"] != self.drone_id:
                    data["received_time"] = self.robot.getTime()
                    self.neighbors[data["drone_id"]] = data
            except (json.JSONDecodeError, KeyError):
                pass
            self.receiver.nextPacket()
        now = self.robot.getTime()
        self.neighbors = {k: v for k, v in self.neighbors.items() if now - v.get("received_time", 0) < 1.0}


if __name__ == "__main__":
    controller = BoidFlockingController()
    controller.run()
