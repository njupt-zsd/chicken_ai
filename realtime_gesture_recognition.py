import cv2
import mediapipe as mp
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# 初始化 MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    model_complexity=1
)
mp_drawing = mp.solutions.drawing_utils

# 加载训练好的模型和 scaler
model_path = "saved_model/model.pkl"
scaler_path = "saved_model/scaler.pkl"

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

def extract_3d_features(hand_world_landmarks):
    """
    提取手部关键点的 3D 坐标作为特征
    """
    features = []
    for landmark in hand_world_landmarks.landmark:
        features.extend([landmark.x, landmark.y, landmark.z])
    return np.array(features).reshape(1, -1)

def predict_gesture(features):
    """
    使用模型预测手势
    """
    # 检查特征是否包含 NaN
    if np.isnan(features).any():
        print("Warning: Features contain NaN values. Skipping prediction.")
        return None

    # 检查特征维度
    if features.shape[1] != 63:
        print(f"Warning: Incorrect feature dimension ({features.shape[1]} instead of 63)")
        return None

    # 标准化特征
    features_scaled = scaler.transform(features)
    # 预测
    label = model.predict(features_scaled)[0]
    return "Gesture" if label == 0 else "Release"

# 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_world_landmarks:
        for hand_world_landmarks in results.multi_hand_world_landmarks:
            # 绘制手部骨骼图
            mp_drawing.draw_landmarks(
                frame,
                hand_world_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            # 提取 3D 特征
            features = extract_3d_features(hand_world_landmarks)

            # 验证特征维度
            if features.shape[1] != 63:
                print(f"Warning: Incorrect feature dimension ({features.shape[1]} instead of 63)")
                continue

            # 预测手势
            gesture = predict_gesture(features)
            print(f"Predicted Gesture: {gesture}")

            # 显示预测结果
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('Real-time Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()