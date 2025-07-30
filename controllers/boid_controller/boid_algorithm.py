# -*- coding: utf-8 -*-
"""
Boid Algorithm Implementation for Drone Flocking (3D Version)

This module implements the classic boid algorithm in 3D space.
"""

import numpy as np
import math


class BoidAlgorithm:
    def __init__(self):
        # 3D空間用に調整したBoidパラメータ
        self.separation_distance = 1.8
        self.alignment_distance = 3.5
        self.cohesion_distance = 4.5
        self.separation_weight = 1.6
        self.alignment_weight = 1.0
        self.cohesion_weight = 0.8
        self.leader_weight = 0.5

        # 速度制限
        self.max_speed = 0.5
        self.min_speed = 0.1

    def calculate_separation(self, my_position, neighbors):
        """3D空間での分離ルールの計算"""
        force = np.zeros(3)
        for neighbor in neighbors:
            neighbor_pos = np.array([neighbor["x"], neighbor["y"], neighbor["z"]])
            my_pos = np.array(my_position)
            dist = np.linalg.norm(neighbor_pos - my_pos)
            if 0 < dist < self.separation_distance:
                diff = my_pos - neighbor_pos
                force += diff / (dist + 1e-6)  # 距離が近いほど強く反発
        return force

    def calculate_alignment(self, my_velocity, neighbors):
        """3D空間での整列ルールの計算"""
        if not neighbors:
            return np.zeros(3)
        avg_velocity = np.mean([np.array([n.get("vx", 0), n.get("vy", 0), n.get("vz", 0)]) for n in neighbors], axis=0)
        return avg_velocity - np.array(my_velocity)

    def calculate_cohesion(self, my_position, neighbors):
        """3D空間での結合ルールの計算"""
        if not neighbors:
            return np.zeros(3)
        center_of_mass = np.mean([np.array([n["x"], n["y"], n["z"]]) for n in neighbors], axis=0)
        my_pos = np.array(my_position)
        return center_of_mass - my_pos

    def calculate_leader_attraction(self, my_position, neighbors):
        """リーダー機への追従力の計算"""
        leader = next((n for n in neighbors if n.get("drone_id") == 1), None)
        if leader:
            leader_pos = np.array([leader["x"], leader["y"], leader["z"]])
            my_pos = np.array(my_position)
            return leader_pos - my_pos
        return np.zeros(3)

    def apply_speed_limit(self, velocity):
        """速度制限を適用"""
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return (velocity / speed) * self.max_speed
        if speed < self.min_speed:
            return (velocity / speed) * self.min_speed
        return velocity

    def calculate_boid_velocity(self, my_position, my_velocity, neighbors, is_leader):
        """Boidsアルゴリズムの3つのルールを統合して目標速度を計算"""
        separation = self.calculate_separation(my_position, neighbors)
        alignment = self.calculate_alignment(my_velocity, neighbors)
        cohesion = self.calculate_cohesion(my_position, neighbors)

        total_force = (
            separation * self.separation_weight + alignment * self.alignment_weight + cohesion * self.cohesion_weight
        )

        if not is_leader:
            leader_attraction = self.calculate_leader_attraction(my_position, neighbors)
            total_force += leader_attraction * self.leader_weight

        return self.apply_speed_limit(total_force)
