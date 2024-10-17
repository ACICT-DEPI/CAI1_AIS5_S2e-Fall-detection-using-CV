import cv2
import torch
import numpy as np
import mediapipe as mp
import winsound
from datetime import datetime
import cvzone
import threading

# Initialize Mediapipe Pose Estimation
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Load pre-trained YOLOv5 model (ensure only "person" class is detected)w
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' for speed

# Sound file path for the alert
alert_sound_path = r"path_to_your_sound_file.wav"  # Ensure it's a valid .wav file path

# Global variables to track fall state and sound state
fall_in_progress = False
sound_thread_active = False
stop_sound_flag = False

def play_alert_sound_continuously():
    """Continuously plays the alert sound while fall is detected."""
    global sound_thread_active, stop_sound_flag
    while fall_in_progress:
        if stop_sound_flag:  # Check if stop is flagged
            break
        winsound.PlaySound(alert_sound_path, winsound.SND_FILENAME)
    sound_thread_active = False  # Reset sound thread state when the loop ends

def stop_alert_sound():
    global stop_sound_flag
    stop_sound_flag = True  # Set flag to stop the sound
    winsound.PlaySound(None, winsound.SND_FILENAME)  # Stops the sound immediately
    stop_sound_flag = False  # Reset the flag for future use

def detect_fall(frame, results):
    fall_detected = False
    person_coords = None  # To store coordinates of the person who falls
    
    # Extract YOLO predictions
    labels, coords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
    
    # Check for detected people (class_id = 0 for 'person')
    for label, coord in zip(labels, coords):
        x1, y1, x2, y2, conf = coord
        class_id = int(label)

        if class_id == 0 and conf > 0.5:  # If class is 'person'
            h, w, _ = frame.shape
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            person = frame[y1:y2, x1:x2]

            # Use Mediapipe to estimate the pose of the detected person
            person_rgb = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            pose_results = pose.process(person_rgb)

            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 0.8:  # Adjust confidence threshold as needed
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'Person', [x1 + 8, y1 - 12], thickness=2, scale=2)
            
            # Detect fall condition based on threshold
            if threshold < 0:
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)
                fall_detected = True
                person_coords = (x1, y1, x2, y2)

    return fall_detected, person_coords

def monitor_falls(video_source=0, skip_frames=5):
    global fall_in_progress, sound_thread_active, stop_sound_flag

    cap = cv2.VideoCapture(video_source)
    frame_count = 0  # Initialize frame count

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Increment the frame count
        frame_count += 1

        # Skip frames
        if frame_count % skip_frames != 0:
            continue  # Skip processing for this frame

        # Resize and process frame with YOLOv5 model
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(frame_rgb)

        # Check if a fall is detected using YOLO and pose estimation
        fall_detected, person_coords = detect_fall(frame, results)

        if fall_detected and not fall_in_progress:
            # Fall detected for the first time
            print("Fall detected! Surrounding person and playing sound...")
            fall_in_progress = True

            # Start a new thread for playing the alert sound continuously
            if not sound_thread_active:
                sound_thread_active = True
                sound_thread = threading.Thread(target=play_alert_sound_continuously)
                sound_thread.start()

        elif not fall_detected and fall_in_progress:
            # Person has recovered from fall
            print("Person has recovered from fall.")
            fall_in_progress = False  # Reset fall state

            # Stop the sound immediately
            stop_alert_sound()

        # Display frame with bounding boxes and predictions
        cv2.imshow('Patient Monitoring - Fall Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When video ends, stop the sound
    stop_alert_sound()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Use webcam (0) or replace with video file path
    monitor_falls("Fall Detection Test.mp4")
