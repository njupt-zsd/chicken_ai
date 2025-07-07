import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1  # 使用高精度模型获取 world_landmarks
)
mp_drawing = mp.solutions.drawing_utils

def calculate_3d_angle(a, b, c):
    """
    计算向量 ab 和 cb 在 3D 空间中的夹角（以度为单位）
    a, b, c: 三个点的 3D 坐标 (x, y, z)，其中 b 是顶点
    """
    ab = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    cb = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    # 归一化向量
    ab_norm = ab / np.linalg.norm(ab)
    cb_norm = cb / np.linalg.norm(cb)

    dot_product = np.dot(ab_norm, cb_norm)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    return angle_deg

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
        for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):
            try:
                # 绘制手部关键点和连线（使用 2D 图像坐标 hand_landmarks）
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                # 获取关键点（thumb tip, thumb MCP, index finger tip）的 3D 坐标
                thumb_tip = hand_world_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                thumb_mcp = hand_world_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]
                index_finger_tip = hand_world_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

                # 计算拇指和食指之间的 3D 夹角
                angle = calculate_3d_angle(thumb_tip, thumb_mcp, index_finger_tip)

                # 判断是否松手（例如角度 > 150 度）
                if angle > 150:
                    print("Detected 'Release' Action", angle)
                    cv2.putText(frame, "Release Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 可视化角度
                cv2.putText(frame, f"Angle: {int(angle)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            except Exception as e:
                print("Error calculating 3D angle:", e)

    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()