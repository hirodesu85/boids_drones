# -*- coding: utf-8 -*-
import sys
import json
import time
from controller import Robot

# シミュレーション設定
TIME_STEP = 16
HOVER_ALTITUDE = 1.0
HOVER_MOTOR_SPEED = 60.0  # 固定モーター速度でホバリング

class SimpleCommunicationTest:
    def __init__(self):
        self.robot = Robot()
        
        # ドローンIDをコマンドライン引数から取得
        if len(sys.argv) > 1:
            self.drone_id = int(sys.argv[1])
        else:
            self.drone_id = 1
            
        print(f"=== Drone {self.drone_id} Starting Communication Test ===")
        
        # モーター初期化（シンプルなホバリング用）
        self.m1_motor = self.robot.getDevice("m1_motor")
        self.m1_motor.setPosition(float("inf"))
        self.m1_motor.setVelocity(-HOVER_MOTOR_SPEED)
        
        self.m2_motor = self.robot.getDevice("m2_motor")
        self.m2_motor.setPosition(float("inf"))
        self.m2_motor.setVelocity(HOVER_MOTOR_SPEED)
        
        self.m3_motor = self.robot.getDevice("m3_motor")
        self.m3_motor.setPosition(float("inf"))
        self.m3_motor.setVelocity(-HOVER_MOTOR_SPEED)
        
        self.m4_motor = self.robot.getDevice("m4_motor")
        self.m4_motor.setPosition(float("inf"))
        self.m4_motor.setVelocity(HOVER_MOTOR_SPEED)
        
        # GPS（位置取得用）
        self.gps = self.robot.getDevice("gps")
        self.gps.enable(TIME_STEP)
        
        # 通信デバイス初期化
        try:
            self.emitter = self.robot.getDevice("emitter")
            if self.emitter is None:
                print(f"Drone {self.drone_id}: ERROR - Emitter device not found!")
            else:
                print(f"Drone {self.drone_id}: Emitter initialized successfully")
        except Exception as e:
            print(f"Drone {self.drone_id}: ERROR initializing emitter: {e}")
            self.emitter = None
            
        try:
            self.receiver = self.robot.getDevice("receiver")
            if self.receiver is None:
                print(f"Drone {self.drone_id}: ERROR - Receiver device not found!")
            else:
                self.receiver.enable(TIME_STEP)
                print(f"Drone {self.drone_id}: Receiver initialized successfully")
        except Exception as e:
            print(f"Drone {self.drone_id}: ERROR initializing receiver: {e}")
            self.receiver = None
        
        # 通信関連変数
        self.last_send_time = 0.0
        self.send_interval = 1.0  # 1秒間隔で送信
        self.received_messages = {}
        
        print(f"Drone {self.drone_id}: Initialization complete")

    def send_message(self):
        """メッセージ送信（1秒間隔）"""
        if self.emitter is None:
            return
            
        current_time = self.robot.getTime()
        
        if current_time - self.last_send_time < self.send_interval:
            return
            
        # GPS位置取得
        position = self.gps.getValues()
        
        # シンプルなメッセージ作成
        message = {
            "drone_id": self.drone_id,
            "time": current_time,
            "x": position[0],
            "y": position[1],
            "z": position[2],
            "status": "active"
        }
        
        # JSON文字列として送信
        message_str = json.dumps(message)
        self.emitter.send(message_str)
        
        print(f"Drone {self.drone_id} SENT: {message}")
        self.last_send_time = current_time

    def receive_messages(self):
        """メッセージ受信"""
        if self.receiver is None:
            return
            
        while self.receiver.getQueueLength() > 0:
            try:
                # 新しいAPIを使用（getString）
                message_str = self.receiver.getString()
                
                # JSONパース
                message = json.loads(message_str)
                sender_id = message["drone_id"]
                
                # 自分からのメッセージは無視
                if sender_id != self.drone_id:
                    self.received_messages[sender_id] = message
                    print(f"Drone {self.drone_id} RECEIVED from Drone {sender_id}: {message}")
                    
            except json.JSONDecodeError as e:
                print(f"Drone {self.drone_id}: JSON decode error: {e}")
            except Exception as e:
                print(f"Drone {self.drone_id}: Receive error: {e}")
                
            self.receiver.nextPacket()

    def print_status(self):
        """定期的にステータス表示"""
        current_time = self.robot.getTime()
        
        # 5秒間隔でステータス表示
        if int(current_time) % 5 == 0 and int(current_time * 10) % 10 == 0:
            position = self.gps.getValues()
            print(f"\n--- Drone {self.drone_id} Status at {current_time:.1f}s ---")
            print(f"Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
            print(f"Received messages from: {list(self.received_messages.keys())}")
            if self.received_messages:
                for drone_id, msg in self.received_messages.items():
                    print(f"  Drone {drone_id}: ({msg['x']:.2f}, {msg['y']:.2f}, {msg['z']:.2f})")
            print("----------------------------------------\n")

    def run(self):
        """メインループ"""
        print(f"Drone {self.drone_id}: Starting main loop...")
        
        step_count = 0
        
        while self.robot.step(TIME_STEP) != -1:
            step_count += 1
            
            # 通信処理
            self.send_message()
            self.receive_messages()
            
            # ステータス表示
            self.print_status()
            
            # 初期の数ステップでデバッグ情報表示
            if step_count <= 5:
                print(f"Drone {self.drone_id}: Step {step_count}, Time: {self.robot.getTime():.2f}s")


if __name__ == "__main__":
    controller = SimpleCommunicationTest()
    controller.run()