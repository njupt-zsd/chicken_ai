import cv2
import mediapipe as mp
import numpy as np

# 初始化MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# 打开摄像头
cap = cv2.VideoCapture(0)
previous_hand_position = None
release_threshold = 0.1  # 松手释放距离阈值

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 转换为RGB格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # 绘制手部关键点
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # 获取关键点坐标
            landmarks = hand_landmarks.landmark
            wrist = landmarks[0]  # 手腕关键点
            index_finger_tip = landmarks[8]  # 食指指尖关键点

            # 计算手腕与食指指尖的相对距离
            distance = np.sqrt((wrist.x - index_finger_tip.x)**2 + (wrist.y - index_finger_tip.y)**2)

            # 判断是否发生"松手释放"动作
            if previous_hand_position is not None:
                movement = np.sqrt((wrist.x - previous_hand_position[0])**2 + (wrist.y - previous_hand_position[1])**2)
                if movement > release_threshold and distance > 0.1:  # 增加距离条件以排除误判
                    print("Detected 'Release' Action")

            # 更新手部位置
            previous_hand_position = (wrist.x, wrist.y)

    # 显示结果
    cv2.imshow('Hand Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()