import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe pose class
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize the drawing class
mp_drawing = mp.solutions.drawing_utils

st.title("Pose Detection App")
st.write("Raise your arm to see a 'Hi' message")
# Initialization
if 'message' not in st.session_state:
    st.session_state['message'] = ''

# Start the camera feed
cap = cv2.VideoCapture(0)

# Create a placeholders for the camera feed and text
frame_placeholder = st.empty()
message_placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.write("Failed to capture video")
        break

    # Convert the image color from BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB frame to get the result
    result = pose.process(frame_rgb)

    # Draw the pose annotation on the frame
    if result.pose_landmarks:
        mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract landmarks for detection
        landmarks = result.pose_landmarks.landmark

        # Get the coordinates of the left and right wrists
        left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        # Check if either wrist is above the corresponding shoulder
        if left_wrist.y < left_shoulder.y or right_wrist.y < right_shoulder.y:
            st.session_state['message'] = '👋 HI'
        else:
            st.session_state['message'] = 'BYE 🚪'

        message_placeholder.title(st.session_state['message'])

    # Convert the frame color from RGB to BGR
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    # Display the frame with annotations
    frame_placeholder.image(frame_bgr, channels="BGR")

cap.release()
cv2.destroyAllWindows()
