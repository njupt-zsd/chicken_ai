import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import csv
import base64
import os
from datetime import datetime

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# 数据存储路径
CSV_PATH = "hand_data.csv"

# 手部关键点中文映射（共 21 个）
landmark_names = {
    0: "腕部",
    1: "拇指腕掌关节",
    2: "拇指掌指关节",
    3: "拇指近节指间关节",
    4: "拇指尖",
    5: "食指掌指关节",
    6: "食指近节指骨关节",
    7: "食指远节指骨关节",
    8: "食指尖",
    9: "中指掌指关节",
    10: "中指近节指骨关节",
    11: "中指远节指骨关节",
    12: "中指尖",
    13: "无名指掌指关节",
    14: "无名指近节指骨关节",
    15: "无名指远节指骨关节",
    16: "无名指尖",
    17: "小指掌指关节",
    18: "小指近节指骨关节",
    19: "小指远节指骨关节",
    20: "小指尖"
}

# 创建 CSV 文件并写入表头
with open(CSV_PATH, mode='w', newline='', encoding='utf-8') as f:
    # 构建表头
    header = ["label", "timestamp", "image_base64"]
    for idx in range(21):
        name = landmark_names.get(idx, f"未知_{idx}")
        header += [f"{name}_x", f"{name}_y", f"{name}_z"]

    writer = csv.DictWriter(f, fieldnames=header)
    writer.writeheader()

# 数据计数器（仅用于界面显示，不写入文件）
count_grasp = 0
count_release = 0

# 去重参数
history_buffer = deque(maxlen=5)
MIN_DISTANCE_TO_SAVE = 0.05

print("按 'g' 表示抓握（label=0），按 'r' 表示松手（label=1）")
print("按 'q' 退出并保存数据")

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_landmarks = None
    drawn_frame = frame.copy()

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
        for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks,
                                                        results.multi_hand_world_landmarks):

            # 绘制 2D 手部骨骼图
            mp_drawing.draw_landmarks(
                drawn_frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 提取 3D 关键点（世界坐标系）
            landmarks_3d = []
            for lm in hand_world_landmarks.landmark:
                landmarks_3d.extend([lm.x, lm.y, lm.z])

            # 合并成最终特征向量（共 21×3 = 63维）
            combined_landmarks = np.array(landmarks_3d)  # shape: (63,)
            current_landmarks = combined_landmarks.copy()

    else:
        cv2.putText(drawn_frame, "No Hand Detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 实时显示采集进度
    cv2.rectangle(drawn_frame, (10, 50), (300, 100), (200, 200, 200), -1)
    cv2.putText(drawn_frame, f"Grasp: {count_grasp}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(drawn_frame, f"Release: {count_release}", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # 显示提示文字
    cv2.putText(drawn_frame, "Press 'g' to save Grasp", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    key = cv2.waitKey(1)

    if results.multi_hand_landmarks and current_landmarks is not None:

        # 去重判断
        should_save = True
        if len(history_buffer) > 0:
            distances = [np.linalg.norm(current_landmarks - old) for old in history_buffer]
            if any(d < MIN_DISTANCE_TO_SAVE for d in distances):
                should_save = False
                print("Skipped duplicate frame.")
        history_buffer.append(current_landmarks)

        # 图像转 Base64
        _, buffer = cv2.imencode('.jpg', drawn_frame)
        jpg_as_text = base64.b64encode(buffer).decode('utf-8')

        # 构建写入字典
        if should_save and key in [ord('g'), ord('r')]:
            label = 0 if key == ord('g') else 1
            timestamp = datetime.now().isoformat(timespec='milliseconds')

            row = {
                "label": label,
                "timestamp": timestamp,
                "image_base64": jpg_as_text
            }

            # 写入所有关键点（中文字段）
            for i in range(21):
                name = landmark_names.get(i, f"未知_{i}")

                # 3D 坐标
                x_3d = round(float(current_landmarks[i * 3]), 6)
                y_3d = round(float(current_landmarks[i * 3 + 1]), 6)
                z_3d = round(float(current_landmarks[i * 3 + 2]), 6)
                row[f"{name}_x"] = x_3d
                row[f"{name}_y"] = y_3d
                row[f"{name}_z"] = z_3d

            # 写入 CSV 文件
            with open(CSV_PATH, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writerow(row)

            # 更新界面显示的样本数量
            if label == 0:
                count_grasp += 1
                print(f"Saved: Grasp (Total: {count_grasp})")
            else:
                count_release += 1
                print(f"Saved: Release (Total: {count_release})")

    elif key == ord('q'):
        print("Exiting and saving data...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    cv2.imshow('Hand Data Collector', drawn_frame)
