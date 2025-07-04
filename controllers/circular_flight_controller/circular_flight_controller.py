# -*- coding: utf-8 -*-
from controller import Robot
import numpy as np
from pid_controller import pid_velocity_fixed_height_controller

# --- シミュレーション設定 ---
TIME_STEP = 16
TAKEOFF_ALTITUDE = 1.0
RADIUS = 0.7
ANGULAR_SPEED = 1.2

# --- 円の中心座標 ---
XY_CIRCLE_CENTER = [0, -0.5, TAKEOFF_ALTITUDE]
YZ_CIRCLE_CENTER = [0, 0.5, TAKEOFF_ALTITUDE]


class CircularFlightController:
    def __init__(self):
        self.robot = Robot()
        self.robot_name = self.robot.getName()
        self.drone_id = int(self.robot_name.replace("cf", ""))

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
        self.imu = self.robot.getDevice("inertial_unit")
        self.imu.enable(TIME_STEP)
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        self.gyro = self.robot.getDevice("gyro")
        self.gyro.enable(TIME_STEP)

        self.pid_controller = pid_velocity_fixed_height_controller()

        # ★★★ ここからが追加/修正点 (1) ★★★
        self.past_time = self.robot.getTime()
        # 速度計算のために過去の位置を記録する変数を追加
        self.past_x_global = 0.0
        self.past_y_global = 0.0
        # ★★★ 追加/修正点ここまで ★★★

    def run(self):
        # 最初の1ステップで過去位置を初期化
        self.past_x_global = self.gps.getValues()[0]
        self.past_y_global = self.gps.getValues()[1]

        while self.robot.step(TIME_STEP) != -1:
            dt = self.robot.getTime() - self.past_time

            if dt == 0:
                continue  # dtが0になるのを防ぐ

            # センサー値の取得
            roll = self.imu.getRollPitchYaw()[0]
            pitch = self.imu.getRollPitchYaw()[1]
            yaw_rate = self.gyro.getValues()[2]
            current_x_global = self.gps.getValues()[0]
            current_y_global = self.gps.getValues()[1]
            altitude = self.gps.getValues()[2]

            # ★★★ ここからが追加/修正点 (2) ★★★
            # GPSからグローバル速度を計算
            vx_global = (current_x_global - self.past_x_global) / dt
            vy_global = (current_y_global - self.past_y_global) / dt

            # グローバル速度を機体速度に変換
            current_yaw = self.imu.getRollPitchYaw()[2]
            cos_yaw = np.cos(current_yaw)
            sin_yaw = np.sin(current_yaw)
            v_x = vx_global * cos_yaw + vy_global * sin_yaw
            v_y = -vx_global * sin_yaw + vy_global * cos_yaw
            # ★★★ 追加/修正点ここまで ★★★

            if altitude < TAKEOFF_ALTITUDE - 0.05:
                desired_vx = 0
                desired_vy = 0
                target_z = TAKEOFF_ALTITUDE
            else:
                time_now = self.robot.getTime()
                num_drones_per_circle = 5

                if 1 <= self.drone_id <= 5:
                    drone_index = self.drone_id - 1
                    phase_offset = (2 * np.pi / num_drones_per_circle) * drone_index
                    angle = ANGULAR_SPEED * time_now + phase_offset

                    target_x = XY_CIRCLE_CENTER[0] + RADIUS * np.cos(angle)
                    target_y = XY_CIRCLE_CENTER[1] + RADIUS * np.sin(angle)
                    target_z = XY_CIRCLE_CENTER[2]
                else:
                    drone_index = self.drone_id - 6
                    phase_offset = (2 * np.pi / num_drones_per_circle) * drone_index
                    angle = ANGULAR_SPEED * time_now + phase_offset

                    target_x = YZ_CIRCLE_CENTER[0]
                    target_y = YZ_CIRCLE_CENTER[1] + RADIUS * np.cos(angle)
                    target_z = YZ_CIRCLE_CENTER[2] + RADIUS * np.sin(angle)

                # ゲインを少し下げる
                Kp_pos = 1.5

                desired_vx_global = Kp_pos * (target_x - current_x_global)
                desired_vy_global = Kp_pos * (target_y - current_y_global)

                desired_vx = desired_vx_global * cos_yaw + desired_vy_global * sin_yaw
                desired_vy = -desired_vx_global * sin_yaw + desired_vy_global * cos_yaw

            # ★★★ ここからが追加/修正点 (3) ★★★
            # PIDコントローラーに実際の機体速度 v_x, v_y を渡す
            motor_power = self.pid_controller.pid(
                dt, desired_vx, desired_vy, 0, target_z, roll, pitch, yaw_rate, altitude, v_x, v_y
            )
            # ★★★ 追加/修正点ここまで ★★★

            self.m1_motor.setVelocity(-motor_power[0])
            self.m2_motor.setVelocity(motor_power[1])
            self.m3_motor.setVelocity(-motor_power[2])
            self.m4_motor.setVelocity(motor_power[3])

            self.past_time = self.robot.getTime()
            # ★★★ ここからが追加/修正点 (4) ★★★
            self.past_x_global = current_x_global
            self.past_y_global = current_y_global
            # ★★★ 追加/修正点ここまで ★★★


if __name__ == "__main__":
    controller = CircularFlightController()
    controller.run()
