# Real-Time Video Stabilizer

This is a Python library for real-time video stabilization using OpenCV. It uses feature matching and optical flow to detect movement and applies Newtonian physics (a critically damped polynomial-order-two kinematics model) to smooth the trajectory. It is designed to remove camera jitter while smoothly following large, intentional movements.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install opencv-python numpy
```

## Features and Parameters

- **Real-Time Smoothing:** Process frames as they arrive (e.g. from a webcam).
- **Video Processing:** Process entirely recorded videos.
- **Newtonian Kinematic Smoothing:** Tracks camera movements utilizing simulated mass, velocity, and acceleration to guarantee completely continuous movement free of jitter. The very first frame functions as the initial reference.
- **Smoothing Factor (0.0 to 1.0):** Controls the stiffness of the tracking camera.
    - Lower values mean more smoothing (slower adaptation to large camera movements, acting like a heavier mass).
    - Higher values mean faster adaptation to camera movements.
- **Configurable Complexity (1 to 10):** Adjust algorithm complexity on a scale from 1 to 10 based on the performance constraints of your application.
    - A higher value (e.g., 10) tracks more features and utilizes larger optical flow search windows, resulting in higher accuracy and robustness at the cost of higher CPU/GPU overhead.
    - Lower values (e.g., 1) process much faster but may lose tracking in low-texture environments.
- **ROI Tracking:** Choose a Region of Interest `(x, y, w, h)` to restrict feature detection to a specific part of the scene. The default is `None`, which automatically utilizes the full image.
- **Coordinate Mapping:** Supports mapping points bidirectionally. You can map a point from the stabilized frame back to the original raw frame, or map a point from the original raw frame forward to where it currently sits on the stabilized frame.

## Usage

### Example 1: Real-Time Frame Stabilization

You can use the `RealTimeVideoStabilizer` class to stabilize frames as they arrive (e.g., from a webcam).

```python
import cv2
from video_stabilizer import RealTimeVideoStabilizer

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize the stabilizer
# smoothing_factor: 0 to 1
# complexity: 1 to 10
stabilizer = RealTimeVideoStabilizer(smoothing_factor=0.1, complexity=5)

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
# Process the video using a high smoothing factor and moderate complexity
stabilize_video(input_video, output_video, smoothing_factor=0.15, complexity=6)
print("Done!")
```

### Example 3: Advanced Configuration (ROI and Coordinate Mapping)

You can pass an `roi` when initializing the stabilizer. Using the `get_original_coordinates` and `get_stabilized_coordinates` methods, you can map points between the original and stabilized streams in either direction.

```python
import cv2
from video_stabilizer import RealTimeVideoStabilizer

# Initialize the stabilizer with advanced parameters
# complexity is set to 10 for maximum accuracy
# roi is defined as (x, y, w, h). If None, it uses the full image.
stabilizer = RealTimeVideoStabilizer(
    smoothing_factor=0.1,
    complexity=10,
    roi=(100, 100, 400, 300)
)

cap = cv2.VideoCapture("input.mp4")
ret, frame = cap.read()

# Process frame to stabilize
stabilized_frame = stabilizer.process_frame(frame)

# --- Backward Mapping ---
# Assume we click on the stabilized frame at coordinate (250, 250)
stab_x, stab_y = 250, 250
orig_x, orig_y = stabilizer.get_original_coordinates(stab_x, stab_y)
print(f"Point ({stab_x}, {stab_y}) on the stabilized frame maps to ({orig_x:.2f}, {orig_y:.2f}) on the original frame.")

# --- Forward Mapping ---
# Assume we track an object bounding box on the original frame at coordinate (150, 150)
orig_x2, orig_y2 = 150, 150
stab_x2, stab_y2 = stabilizer.get_stabilized_coordinates(orig_x2, orig_y2)
print(f"Point ({orig_x2}, {orig_y2}) on the original frame maps to ({stab_x2:.2f}, {stab_y2:.2f}) on the stabilized frame.")

cap.release()
```
