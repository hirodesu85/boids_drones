# インポート文
from controller import Robot
from controller import Keyboard
from math import cos, sin
from pid_controller import pid_velocity_fixed_height_controller

# 定数の定義
FLYING_ATTITUDE = 1

if __name__ == "__main__":

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

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

    # キーボード入力の初期化
    keyboard = Keyboard()
    keyboard.enable(timestep)

    # 変数の初期化
    past_x_global = 0
    past_y_global = 0
    past_time = 0
    first_time = True

    # ドローンの制御部分の初期化
    PID_crazyflie = pid_velocity_fixed_height_controller()
    PID_update_last_time = robot.getTime()
    sensor_read_last_time = robot.getTime()

    height_desired = FLYING_ATTITUDE

    # 説明文の出力
    print("\n")
    print("====== Controls =====\n\n")
    print(" The Crazyflie can be controlled from your keyboard!\n")
    print(" All controllable movement is in body coordinates\n")
    print("- Use the up, back, right and left button to move in the horizontal plane\n")
    print("- Use Q and E to rotate around yaw\n ")
    print("- Use W and S to go up and down\n ")

    # メインループ
    while robot.step(timestep) != -1:

        dt = robot.getTime() - past_time

        # 初回のループだけ特別に処理
        if first_time:
            past_x_global = gps.getValues()[0]
            past_y_global = gps.getValues()[1]
            past_time = robot.getTime()
            first_time = False

        # センサーデータの取得
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        yaw = imu.getRollPitchYaw()[2]
        yaw_rate = gyro.getValues()[2]
        x_global = gps.getValues()[0]
        v_x_global = (x_global - past_x_global) / dt
        y_global = gps.getValues()[1]
        v_y_global = (y_global - past_y_global) / dt
        altitude = gps.getValues()[2]

        # グローバル座標系から機体座標系への変換
        cos_yaw = cos(yaw)
        sin_yaw = sin(yaw)
        v_x = v_x_global * cos_yaw + v_y_global * sin_yaw
        v_y = -v_x_global * sin_yaw + v_y_global * cos_yaw

        # 値の初期化
        desired_state = [0, 0, 0, 0]
        forward_desired = 0
        sideways_desired = 0
        yaw_desired = 0
        height_diff_desired = 0

        # キーボード入力を処理
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

        camera_data = camera.getImage()

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
            v_x,
            v_y,
        )

        m1_motor.setVelocity(-motor_power[0])
        m2_motor.setVelocity(motor_power[1])
        m3_motor.setVelocity(-motor_power[2])
        m4_motor.setVelocity(motor_power[3])

        past_time = robot.getTime()
        past_x_global = x_global
        past_y_global = y_global
