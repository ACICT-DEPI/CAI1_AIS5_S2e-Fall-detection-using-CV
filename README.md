# Fall-Detection-Using-CV
This project uses a combination of computer vision techniques, including the YOLOv5 object detection model and Mediapipe pose estimation, to detect falls in real-time. The system analyzes video feeds from a webcam or video file to identify people, estimate their posture, and detect potential falls. Upon detecting a fall, it triggers an alert sound to notify caregivers or monitoring personnel.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Future Improvements](#future-improvements)
- [License](#license)

## Features

- Real-time fall detection using a webcam or video file.
- Detects falls by monitoring a person's posture and dimensions.
- Triggers an alert sound when a fall is detected.
- Customizable frame skipping to optimize performance.
- Visual output with bounding boxes around detected individuals.

## Installation

### Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch
- Mediapipe
- Cvzone (for bounding box and text display)
- Winsound (for Windows-based sound alerts)

### Steps

1. **Clone the repository**:

    ```bash
    git clone https://github.com/Johnlorance/Fall-Detection-Using-CV.git
    ```

2. **Install dependencies**:

    ```bash
    pip install opencv-python torch numpy mediapipe cvzone
    ```

3. **Configure YOLOv5**:  
   The pre-trained YOLOv5 model will be downloaded automatically when you first run the script.

4. **Add alert sound file** (optional for sound alerts):  
   Ensure you have a valid `.wav` file for alert sounds.  
   Update the `alert_sound_path` in the script with the path to your `.wav` file.

## Usage

1. **Run the script**:
    - To start fall detection using your webcam:

        ```bash
        python fall_detection.py
        ```

    - To use a video file as input, modify the code in `monitor_falls()`:

        ```python
        monitor_falls("path_to_video.mp4")
        ```

    - Replace `"path_to_video.mp4"` with the path to your video file.

2. **Quit the program**:  
   Press `q` while the video window is active to exit the program.

## How It Works

1. **Person Detection**:  
   YOLOv5 is used to detect people in the frame. Only the "person" class (class_id = 0) is processed.

2. **Pose Estimation**:  
   Once a person is detected, Mediapipe's pose estimation is used to estimate the posture and identify if the person has fallen.

3. **Fall Detection Logic**:  
   A fall is determined based on the ratio of height to width of the detected person's bounding box.  
   If the person's height becomes smaller than their width (indicating a horizontal position), a fall is flagged.

4. **Alert System**:  
   If a fall is detected, a separate thread is used to play an alert sound until the person recovers or the program is stopped.

## Configuration

- **Video Source**:  
   The `monitor_falls()` function can take a webcam (default `0`) or a video file as input:

   ```python
   monitor_falls(video_source=0, skip_frames=5)
