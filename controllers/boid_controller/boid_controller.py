# -*- coding: utf-8 -*-
"""
Boid Flocking Controller for Crazyflie Drones (3D Version - Stabilized)
"""

import sys
import json
import numpy as np
from controller import Robot
from pid_controller import pid_velocity_fixed_height_controller
from boid_algorithm import BoidAlgorithm

# --- 設定 ---
TIME_STEP = 16
INITIAL_HOVER_ALTITUDE = 1.0  # 初期離陸高度
MINIMUM_ALTITUDE = 0.8  # 安全のための最低高度
MAXIMUM_ALTITUDE = 2.0  # 探索範囲の最高高度


class BoidFlockingController:
    def __init__(self):
        self.robot = Robot()
        self.drone_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
        print(f"=== Drone {self.drone_id} Starting 3D Boid Flocking (Stabilized) ===")

        # モーターとセンサーの初期化
        self.motors = self._init_motors()
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(TIME_STEP)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)

        # 通信の初期化
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        if self.receiver:
            self.receiver.enable(TIME_STEP)

        # 制御システムの初期化
        self.pid_controller = pid_velocity_fixed_height_controller()
        self.boid_algorithm = BoidAlgorithm()

        # 状態変数
        self.past_time = self.robot.getTime()
        self.past_position = np.zeros(3)
        self.current_velocity = np.zeros(3)

        # 通信変数
        self.neighbors = {}
        self.last_send_time = 0.0

        # 飛行状態
        self.flight_mode = "takeoff"
        self.desired_altitude = INITIAL_HOVER_ALTITUDE

        # リーダー機用の設定
        self.is_leader = self.drone_id == 1
        self.wander_angle = 0.0
        self.leader_altitude_time = 0.0

    def _init_motors(self):
        motors = []
        for name in ["m1_motor", "m2_motor", "m3_motor", "m4_motor"]:
            motor = self.robot.getDevice(name)
            motor.setPosition(float("inf"))
            # m1, m3は逆回転、m2, m4は正回転
            velocity_sign = 1 if "m2" in name or "m4" in name else -1
            motor.setVelocity(velocity_sign)
            motors.append(motor)
        return motors

    def run(self):
        """メイン制御ループ"""
        # センサーが安定するまで待機
        while self.robot.step(TIME_STEP) != -1 and self.robot.getTime() < 0.5:
            pass
        self.past_position = self.gps.getValues()
        self.past_time = self.robot.getTime()

        while self.robot.step(TIME_STEP) != -1:
            dt = self.robot.getTime() - self.past_time
            if dt <= 0:
                continue

            # --- 状態更新フェーズ ---
            current_pos = np.array(self.gps.getValues())
            self._update_velocity(current_pos, dt)
            self._update_flight_mode(current_pos[2])

            # --- 通信フェーズ ---
            self._send_state(current_pos)
            self._receive_state()

            # --- 目標値計算フェーズ ---
            desired_vx, desired_vy = 0, 0

            if self.flight_mode == "flocking":
                if self.is_leader:
                    # リーダーはXY方向の探索と、Z方向の滑らかな高度目標を生成
                    desired_vx, desired_vy = self._get_leader_xy_velocity()
                    self.desired_altitude = self._get_leader_target_altitude()
                else:
                    # フォロワーはBoidsに従う
                    boid_vel = self.boid_algorithm.calculate_boid_velocity(
                        current_pos, self.current_velocity, list(self.neighbors.values()), self.is_leader
                    )
                    desired_vx, desired_vy = boid_vel[0], boid_vel[1]

                    # ★重要: 垂直速度(vz)を目標高度への穏やかな補正として使用する
                    altitude_correction = boid_vel[2] * dt * 0.5  # 補正の影響を緩やかにする
                    self.desired_altitude += np.clip(altitude_correction, -0.05, 0.05)

            # 安全のため、目標高度を範囲内に収める
            self.desired_altitude = np.clip(self.desired_altitude, MINIMUM_ALTITUDE, MAXIMUM_ALTITUDE)

            # --- 制御フェーズ ---
            roll, pitch, _ = self.imu.getRollPitchYaw()
            yaw_rate = self.gyro.getValues()[2]

            # PIDコントローラーでモーター出力を計算
            motor_power = self.pid_controller.pid(
                dt,
                np.clip(desired_vx, -1.0, 1.0),  # 安全対策
                np.clip(desired_vy, -1.0, 1.0),  # 安全対策
                0,  # Yaw rate
                self.desired_altitude,
                roll,
                pitch,
                yaw_rate,
                current_pos[2],  # actual altitude
                self.current_velocity[0],  # actual_vx
                self.current_velocity[1],  # actual_vy
            )

            # モーターへ指令
            # m1,m3は-、m2,m4は+の出力
            self.motors[0].setVelocity(-motor_power[0])
            self.motors[1].setVelocity(motor_power[1])
            self.motors[2].setVelocity(-motor_power[2])
            self.motors[3].setVelocity(motor_power[3])

            # 状態更新
            self.past_time = self.robot.getTime()
            self.past_position = current_pos

    def _update_velocity(self, current_pos, dt):
        # 3D速度を計算
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
            self.emitter.send(json.dumps(message))
            self.last_send_time = self.robot.getTime()

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
        # タイムアウトした隣人を削除
        now = self.robot.getTime()
        self.neighbors = {k: v for k, v in self.neighbors.items() if now - v.get("received_time", 0) < 1.0}

    def _get_leader_xy_velocity(self):
        """リーダーのXY平面上の探索速度を計算"""
        self.wander_angle += np.random.uniform(-0.4, 0.4)  # 探索方向をランダムに変更
        speed = 0.3
        vx = np.cos(self.wander_angle) * speed
        vy = np.sin(self.wander_angle) * speed
        return vx, vy

    def _get_leader_target_altitude(self):
        """リーダーの目標高度をサイン波で滑らかに計算"""
        self.leader_altitude_time += TIME_STEP / 1000.0
        amplitude = (MAXIMUM_ALTITUDE - MINIMUM_ALTITUDE) / 2
        mid_point = (MAXIMUM_ALTITUDE + MINIMUM_ALTITUDE) / 2
        # 周期の異なるサイン波を合成して、より複雑な動きにする
        alt1 = amplitude * np.sin(self.leader_altitude_time * 0.2)
        alt2 = amplitude * np.cos(self.leader_altitude_time * 0.35)
        return mid_point + (alt1 + alt2) / 2


if __name__ == "__main__":
    controller = BoidFlockingController()
    controller.run()
