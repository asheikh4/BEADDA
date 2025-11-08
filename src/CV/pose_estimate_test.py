import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # Raspberry Pi Camera should appear as camera 0

while True:
    success, frame = cap.read()
    if not success:
        print("Ignoring empty frame.")
        continue

    # Convert the BGR image to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('Pose Estimation', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
