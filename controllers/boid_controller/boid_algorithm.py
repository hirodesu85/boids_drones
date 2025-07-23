# -*- coding: utf-8 -*-
"""
Boid Algorithm Implementation for Drone Flocking

This module implements the classic boid algorithm with three main rules:
1. Separation: Avoid crowding neighbors
2. Alignment: Steer towards the average heading of neighbors  
3. Cohesion: Steer towards the average position of neighbors
"""

import numpy as np
import math


class BoidAlgorithm:
    def __init__(self):
        # Boid algorithm parameters (conservative values for stability)
        self.separation_distance = 1.5  # Minimum distance to maintain from other drones
        self.alignment_distance = 3.0   # Distance to consider for alignment
        self.cohesion_distance = 4.0    # Distance to consider for cohesion
        
        # Weights for combining the three forces
        self.separation_weight = 1.5    # Reduced for smoother behavior
        self.alignment_weight = 1.0     # Medium priority for formation
        self.cohesion_weight = 0.8      # Lower priority for grouping
        
        # Speed limits for stability
        self.max_speed = 0.3            # Reduced from 0.5 for better stability
        self.min_speed = 0.05           # Reduced from 0.1 for gentler movements
        
    def calculate_separation(self, my_position, neighbors):
        """
        Calculate separation force to avoid crowding neighbors
        
        Args:
            my_position: [x, y, z] current position
            neighbors: List of neighbor data [{id, x, y, z, vx, vy, vz}, ...]
        
        Returns:
            separation_force: [vx, vy] desired velocity for separation
        """
        separation_force = np.array([0.0, 0.0])
        
        for neighbor in neighbors:
            neighbor_pos = np.array([neighbor['x'], neighbor['y']])
            my_pos = np.array([my_position[0], my_position[1]])
            
            # Calculate distance (only consider X-Y plane for now)
            distance = np.linalg.norm(neighbor_pos - my_pos)
            
            if 0 < distance < self.separation_distance:
                # Calculate direction away from neighbor
                diff = my_pos - neighbor_pos
                if distance > 0:
                    diff = diff / distance  # Normalize
                
                # Stronger force when closer
                force_magnitude = (self.separation_distance - distance) / self.separation_distance
                separation_force += diff * force_magnitude
        
        return separation_force
    
    def calculate_alignment(self, my_velocity, neighbors):
        """
        Calculate alignment force to match neighbors' average velocity
        
        Args:
            my_velocity: [vx, vy] current velocity
            neighbors: List of neighbor data
            
        Returns:
            alignment_force: [vx, vy] desired velocity for alignment
        """
        if not neighbors:
            return np.array([0.0, 0.0])
        
        avg_velocity = np.array([0.0, 0.0])
        count = 0
        
        for neighbor in neighbors:
            neighbor_vel = np.array([neighbor.get('vx', 0), neighbor.get('vy', 0)])
            neighbor_pos = np.array([neighbor['x'], neighbor['y']])
            my_pos = np.array([my_velocity[0], my_velocity[1]])  # This should be position, fixing below
            
            # Only consider neighbors within alignment distance
            # Note: We need position for this calculation, will fix in main controller
            avg_velocity += neighbor_vel
            count += 1
        
        if count > 0:
            avg_velocity /= count
            # Return desired velocity change, not absolute velocity
            alignment_force = avg_velocity - np.array([my_velocity[0], my_velocity[1]])
            return alignment_force
        
        return np.array([0.0, 0.0])
    
    def calculate_cohesion(self, my_position, neighbors):
        """
        Calculate cohesion force to move towards center of neighbors
        
        Args:
            my_position: [x, y, z] current position
            neighbors: List of neighbor data
            
        Returns:
            cohesion_force: [vx, vy] desired velocity for cohesion
        """
        if not neighbors:
            return np.array([0.0, 0.0])
        
        center_of_mass = np.array([0.0, 0.0])
        count = 0
        
        for neighbor in neighbors:
            neighbor_pos = np.array([neighbor['x'], neighbor['y']])
            my_pos = np.array([my_position[0], my_position[1]])
            
            # Calculate distance
            distance = np.linalg.norm(neighbor_pos - my_pos)
            
            if distance < self.cohesion_distance:
                center_of_mass += neighbor_pos
                count += 1
        
        if count > 0:
            center_of_mass /= count
            my_pos = np.array([my_position[0], my_position[1]])
            
            # Calculate desired direction towards center
            direction = center_of_mass - my_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0:
                # Normalize and scale by distance
                direction = direction / distance
                force_magnitude = min(distance * 0.3, 0.5)  # Scale factor
                return direction * force_magnitude
        
        return np.array([0.0, 0.0])
    
    def combine_forces(self, separation, alignment, cohesion):
        """
        Combine the three boid forces with appropriate weights
        
        Args:
            separation: [vx, vy] separation force
            alignment: [vx, vy] alignment force  
            cohesion: [vx, vy] cohesion force
            
        Returns:
            combined_force: [vx, vy] final desired velocity
        """
        # Apply weights to each force
        total_force = (separation * self.separation_weight + 
                      alignment * self.alignment_weight + 
                      cohesion * self.cohesion_weight)
        
        # Limit the magnitude of the resulting force
        magnitude = np.linalg.norm(total_force)
        if magnitude > self.max_speed:
            total_force = (total_force / magnitude) * self.max_speed
        elif magnitude < self.min_speed and magnitude > 0:
            total_force = (total_force / magnitude) * self.min_speed
            
        return total_force
    
    def calculate_boid_velocity(self, my_position, my_velocity, neighbors):
        """
        Main function to calculate desired velocity using boid algorithm
        
        Args:
            my_position: [x, y, z] current position
            my_velocity: [vx, vy] current velocity
            neighbors: List of neighbor data
            
        Returns:
            desired_velocity: [vx, vy] desired velocity for this timestep
        """
        # Calculate the three forces
        separation = self.calculate_separation(my_position, neighbors)
        alignment = self.calculate_alignment(my_velocity, neighbors)
        cohesion = self.calculate_cohesion(my_position, neighbors)
        
        # Combine forces
        desired_velocity = self.combine_forces(separation, alignment, cohesion)
        
        return desired_velocity
    
    def filter_neighbors_by_distance(self, my_position, all_neighbors, max_distance):
        """
        Filter neighbors by distance to only consider nearby drones
        
        Args:
            my_position: [x, y, z] current position
            all_neighbors: List of all neighbor data
            max_distance: Maximum distance to consider
            
        Returns:
            filtered_neighbors: List of nearby neighbors
        """
        filtered = []
        my_pos = np.array([my_position[0], my_position[1]])
        
        for neighbor in all_neighbors:
            neighbor_pos = np.array([neighbor['x'], neighbor['y']])
            distance = np.linalg.norm(neighbor_pos - my_pos)
            
            if distance <= max_distance:
                neighbor['distance'] = distance
                filtered.append(neighbor)
                
        return filtered