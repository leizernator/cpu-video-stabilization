# Real-Time Video Stabilizer

This is a Python library for real-time video stabilization using OpenCV. It uses feature matching and optical flow to detect movement and applies an Exponential Moving Average (EMA) to smooth the trajectory. It is designed to remove camera jitter while smoothly following large, intentional movements.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install opencv-python numpy
```

## Features

- **Real-Time Smoothing:** Process frames as they arrive (e.g. from a webcam).
- **Video Processing:** Process entirely recorded videos.
- **Configurable Complexity:** Adjust algorithm parameters (`low`, `medium`, `high`) based on the performance constraints of your application.
- **ROI Tracking:** Choose a Region of Interest (ROI) to restrict feature detection to a specific, static part of the scene.
- **Coordinate Mapping:** Click or select a point on the stabilized frame and get its true coordinates in the raw original frame.

## Usage

### Example 1: Real-Time Frame Stabilization

You can use the `RealTimeVideoStabilizer` class to stabilize frames as they arrive (e.g., from a webcam).

```python
import cv2
from video_stabilizer import RealTimeVideoStabilizer

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize the stabilizer
# smoothing_factor controls the smoothing (0 to 1). Lower = smoother (slower to adapt to large movements)
stabilizer = RealTimeVideoStabilizer(smoothing_factor=0.1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame
    stabilized_frame = stabilizer.process_frame(frame)

    # Display the results
    cv2.imshow('Original', frame)
    cv2.imshow('Stabilized', stabilized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### Example 2: Stabilize a Recorded Video

You can use the `stabilize_video` utility function to process a complete video file and save the stabilized output.

```python
from video_stabilizer import stabilize_video

input_video = "input.mp4"
output_video = "output_stabilized.mp4"

print("Stabilizing video...")
stabilize_video(input_video, output_video)
print("Done!")
```

### Example 3: Advanced Configuration (Complexity, ROI, and Coordinate Mapping)

You can pass `complexity` and `roi` when initializing the stabilizer. Using the `get_original_coordinates` method, you can also map any click or point on the stabilized video back to the original source video.

```python
import cv2
from video_stabilizer import RealTimeVideoStabilizer

# Initialize the stabilizer with advanced parameters
# complexity can be 'low', 'medium', or 'high'.
# roi is defined as (x, y, w, h)
stabilizer = RealTimeVideoStabilizer(
    smoothing_factor=0.1,
    complexity='high',
    roi=(100, 100, 400, 300)
)

# You can also change the ROI at runtime
# stabilizer.set_roi((50, 50, 200, 200))

cap = cv2.VideoCapture("input.mp4")
ret, frame = cap.read()

# Process frame to stabilize
stabilized_frame = stabilizer.process_frame(frame)

# Assume we click on the stabilized frame at coordinate (250, 250)
stab_x, stab_y = 250, 250

# Find where this point originated from in the original, jittery frame
orig_x, orig_y = stabilizer.get_original_coordinates(stab_x, stab_y)

print(f"Point ({stab_x}, {stab_y}) on the stabilized frame maps to ({orig_x:.2f}, {orig_y:.2f}) on the original frame.")

cap.release()
```
