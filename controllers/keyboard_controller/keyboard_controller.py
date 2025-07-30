# -*- coding: utf-8 -*-
# インポート文
import sys
import json
from controller import Robot, Keyboard
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller

# 定数の定義
FLYING_ATTITUDE = 1.0


# --- ★追加: 状態送信用の関数 ---
def send_state(robot, emitter, gps, velocity_global, drone_id):
    """自身の状態をJSON形式でブロードキャストする"""
    if emitter is None:
        return

    pos = gps.getValues()
    message = {
        "drone_id": drone_id,
        "x": pos[0],
        "y": pos[1],
        "z": pos[2],
        "vx": velocity_global[0],
        "vy": velocity_global[1],
        "vz": velocity_global[2],
    }
    try:
        emitter.send(json.dumps(message))
    except Exception as e:
        print(f"Drone {drone_id} send error: {e}")


if __name__ == "__main__":
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # --- ★追加: コマンドライン引数からドローンIDを取得 ---
    drone_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 引数がなければ0
    print(f"=== Keyboard Controller Started for Drone ID: {drone_id} ===")

    # モーターの取得と初期化
    m1_motor = robot.getDevice("m1_motor")
    m1_motor.setPosition(float("inf"))
    m1_motor.setVelocity(-1)
    m2_motor = robot.getDevice("m2_motor")
    m2_motor.setPosition(float("inf"))
    m2_motor.setVelocity(1)
    m3_motor = robot.getDevice("m3_motor")
    m3_motor.setPosition(float("inf"))
    m3_motor.setVelocity(-1)
    m4_motor = robot.getDevice("m4_motor")
    m4_motor.setPosition(float("inf"))
    m4_motor.setVelocity(1)

    # センサー類の初期化
    imu = robot.getDevice("inertial_unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- ★追加: Emitter（送信機）の初期化 ---
    emitter = robot.getDevice("emitter")

    # 変数の初期化
    past_position = gps.getValues()
    past_time = robot.getTime()
    last_send_time = 0.0

    # ドローンの制御部分の初期化
    PID_crazyflie = pid_velocity_fixed_height_controller()
    height_desired = FLYING_ATTITUDE

    # 説明文の出力
    print("\n====== Keyboard Controls =====\n")
    print("- Arrow Keys: Move in horizontal plane")
    print("- Q/E: Rotate (Yaw)")
    print("- W/S: Move up/down")

    # メインループ
    while robot.step(timestep) != -1:
        dt = robot.getTime() - past_time
        if dt <= 0:
            past_time = robot.getTime()
            continue

        # センサーデータの取得
        roll, pitch, yaw = imu.getRollPitchYaw()
        yaw_rate = gyro.getValues()[2]
        current_pos = gps.getValues()
        altitude = current_pos[2]

        # グローバル座標系での速度を計算
        vx_global = (current_pos[0] - past_position[0]) / dt
        vy_global = (current_pos[1] - past_position[1]) / dt
        vz_global = (current_pos[2] - past_position[2]) / dt
        velocity_global = [vx_global, vy_global, vz_global]

        # グローバル速度を機体座標系速度に変換 (PID制御用)
        cos_yaw, sin_yaw = cos(yaw), sin(yaw)
        v_x_body = vx_global * cos_yaw + vy_global * sin_yaw
        v_y_body = -vx_global * sin_yaw + vy_global * cos_yaw

        # --- ★追加: 定期的に状態を送信 ---
        if robot.getTime() - last_send_time > 0.1:  # 100msごとに送信
            send_state(robot, emitter, gps, velocity_global, drone_id)
            last_send_time = robot.getTime()

        # キーボード入力を処理
        forward_desired, sideways_desired, yaw_desired, height_diff_desired = 0, 0, 0, 0
        key = keyboard.getKey()
        while key > 0:
            if key == Keyboard.UP:
                forward_desired += 0.5
            elif key == Keyboard.DOWN:
                forward_desired -= 0.5
            elif key == Keyboard.RIGHT:
                sideways_desired -= 0.5
            elif key == Keyboard.LEFT:
                sideways_desired += 0.5
            elif key == ord("Q"):
                yaw_desired = +1
            elif key == ord("E"):
                yaw_desired = -1
            elif key == ord("W"):
                height_diff_desired = 0.1
            elif key == ord("S"):
                height_diff_desired = -0.1
            key = keyboard.getKey()

        height_desired += height_diff_desired * dt
        height_desired = max(0.2, height_desired)  # 最低高度を維持

        # PID制御
        motor_power = PID_crazyflie.pid(
            dt,
            forward_desired,
            sideways_desired,
            yaw_desired,
            height_desired,
            roll,
            pitch,
            yaw_rate,
            altitude,
            v_x_body,
            v_y_body,
        )

        # モーターへ指令
        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        # 状態の更新
        past_time = robot.getTime()
        past_position = current_pos
