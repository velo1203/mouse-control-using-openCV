import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

win_size = pyautogui.size()
win_width  = win_size.width
win_height = win_size.height

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_height, img_width, _ = image.shape


    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        width = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * img_width)
        height = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * img_height)
        print(width, height)
        width_fractions = width / img_width * 100
        height_fractions = height / img_height * 100
        
        middle_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * img_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * img_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * img_height:
                    middle_finger_state = 1

        ring_finger_state = 0
        if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * img_height:
            if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * img_height:
                if hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * img_height > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * img_height:
                    ring_finger_state = 1

        if middle_finger_state == 1 and ring_finger_state == 0:
            pyautogui.click()
        elif middle_finger_state == 1 and ring_finger_state == 1:
            pyautogui.doubleClick()

        pyautogui.moveTo(int(win_width - win_width / 100 * width_fractions), int( win_height / 100 * height_fractions))

        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())


    cv2.imshow('MediaPipe Hands',cv2.flip(image,1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()