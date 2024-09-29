import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# Initialize Mediapipe Hand
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Define the keyboard layout
keys = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M']
]

key_width = 60
key_height = 60

def draw_keyboard(img):
    for i, row in enumerate(keys):
        for j, key in enumerate(row):
            x = j * key_width + 20
            y = i * key_height + 100
            cv2.rectangle(img, (x, y), (x + key_width, y + key_height), (255, 255, 255), -1)
            cv2.putText(img, key, (x + 20, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

def is_hand_open(hand_landmarks):
    # Check if hand is open based on landmarks
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    return index_tip.y < index_pip.y

def is_click_gesture(hand_landmarks):
    # Check if index finger tip is close to thumb tip
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    distance = np.sqrt((index_tip.x - thumb_tip.x) ** 2 + (index_tip.y - thumb_tip.y) ** 2)
    return distance < 0.05  # Adjust this threshold based on your preference

def is_swipe_right_to_left(positions):
    if len(positions) < 2:
        return False
    return positions[-1][0] < positions[0][0] - 0.2  # Adjust this threshold based on your preference

# Initialize video capture
cap = cv2.VideoCapture(0)
typed_text = ""
key_pressed = False
hand_positions = deque(maxlen=10)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    draw_keyboard(frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_hand_open(hand_landmarks):
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)
                cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

                for i, row in enumerate(keys):
                    for j, key in enumerate(row):
                        key_x = j * key_width + 20
                        key_y = i * key_height + 100
                        if key_x < x < key_x + key_width and key_y < y < key_y + key_height:
                            cv2.rectangle(frame, (key_x, key_y), (key_x + key_width, key_y + key_height), (0, 255, 0), -1)
                            cv2.putText(frame, key, (key_x + 20, key_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                            if is_click_gesture(hand_landmarks):
                                if not key_pressed:
                                    typed_text += key
                                    key_pressed = True
                # Reset key_pressed when hand is not making the click gesture
                if not is_click_gesture(hand_landmarks):
                    key_pressed = False

                # Track hand position
                hand_positions.append((hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y))
            else:
                key_pressed = False

            # Check for swipe right to left gesture
            if is_swipe_right_to_left(hand_positions):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    cv2.imshow('Virtual Keyboard', frame)

    # Create a blank image for displaying typed text
    text_frame = np.zeros((200, 600, 3), np.uint8)
    cv2.putText(text_frame, typed_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('Typed Text', text_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
