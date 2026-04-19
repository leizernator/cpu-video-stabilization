# Real-Time Video Stabilizer

This is a Python library for real-time video stabilization using OpenCV. It uses feature matching and optical flow to detect movement and applies an Exponential Moving Average (EMA) to smooth the trajectory. It is designed to remove camera jitter while smoothly following large, intentional movements.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install opencv-python numpy
```

## Usage

You can use the library for processing individual frames in real time, or for stabilizing a recorded video file.

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
