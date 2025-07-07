import cv2
import mediapipe as mp
import numpy as np

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

def vector_angle(v1, v2):
    """计算两个向量之间的夹角（度）"""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

def get_vector(a, b):
    """从 a 到 b 的向量"""
    return np.array([b.x - a.x, b.y - b.z])  # z 坐标也参与向量方向计算


# 绘制偏移+缩放后的 hand_world_landmarks（右上角）
def visualize_shifted_hand_landmarks(frame, hand_world_landmarks, offset=(400, 100), scale=1.5):
    image_height, image_width, _ = frame.shape
    landmarks = hand_world_landmarks.landmark

    overlay = frame.copy()

    for connection in mp_hands.HAND_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]

        landmark_start = landmarks[start_idx]
        landmark_end = landmarks[end_idx]

        x_start = int(landmark_start.x * image_width * scale) + offset[0]
        y_start = int(landmark_start.y * image_height * scale) + offset[1]
        x_end = int(landmark_end.x * image_width * scale) + offset[0]
        y_end = int(landmark_end.y * image_height * scale) + offset[1]

        cv2.line(overlay, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

    for landmark in landmarks:
        x = int(landmark.x * image_width * scale) + offset[0]
        y = int(landmark.y * image_height * scale) + offset[1]
        cv2.circle(overlay, (x, y), 4, (0, 0, 255), -1)

    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


# 计算各手指第一节与拇指方向夹角
def calculate_finger_angles(hand_world_landmarks):
    landmarks = hand_world_landmarks.landmark

    thumb_mcp = landmarks[mp_hands.HandLandmark.THUMB_MCP]
    thumb_ip = landmarks[mp_hands.HandLandmark.THUMB_IP]
    thumb_vec = get_vector(thumb_ip, thumb_mcp)

    angles = {}

    def add_angle(name, pip_name, mcp_name):
        pip = landmarks[pip_name]
        mcp = landmarks[mcp_name]
        vec = get_vector(pip, mcp)
        angles[name] = vector_angle(vec, thumb_vec)

    add_angle("Index", mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_MCP)
    add_angle("Middle", mp_hands.HandLandmark.MIDDLE_FINGER_PIP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP)
    add_angle("Ring", mp_hands.HandLandmark.RING_FINGER_PIP, mp_hands.HandLandmark.RING_FINGER_MCP)
    add_angle("Pinky", mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP)

    return angles


# 显示角度信息
def draw_angles_on_frame(frame, angles, y_start=50):
    for i, (finger, angle) in enumerate(angles.items()):
        text = f"{finger}: {int(angle)}"
        cv2.putText(frame, text, (10, y_start + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


# 判断是否松手
def is_hand_open(angles):
    return all(angle > 90 for angle in angles.values())


# 主循环
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks and results.multi_hand_world_landmarks:
        for hand_landmarks, hand_world_landmarks in zip(results.multi_hand_landmarks, results.multi_hand_world_landmarks):

            # 绘制原始 2D 手部骨骼图（左上角）
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 绘制偏移+缩放的 3D 手部骨骼图（右上角）
            visualize_shifted_hand_landmarks(frame, hand_world_landmarks, offset=(400, 100), scale=1.5)

            # 计算并绘制角度
            angles = calculate_finger_angles(hand_world_landmarks)
            draw_angles_on_frame(frame, angles)

            # 判断是否松手
            if is_hand_open(angles):
                cv2.putText(frame, "Release Detected", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Hand Closed", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Tracking with 2D & 3D Landmarks', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()