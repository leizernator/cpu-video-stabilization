# Real-Time Video Stabilizer

This is a Python library for real-time video stabilization using OpenCV. It uses feature matching (Optical Flow or ORB) to detect movement and applies a **Kalman Filter dynamic model** to smooth the trajectory. It is designed to remove camera jitter entirely while smoothly following large, intentional movements.

## Installation

Ensure you have the required dependencies installed:

```bash
pip install opencv-python numpy
```

## Features and Parameters

- **Real-Time Smoothing:** Process frames as they arrive (e.g. from a webcam).
- **Video Processing:** Process entirely recorded videos.
- **Kalman Filter Smoothing:** Utilizes a constant velocity Kalman Filter model to track and smooth the camera trajectory. This mathematically guarantees the elimination of jitter while maintaining real-time responsiveness.
- **Extractor Type (`shi_tomasi` or `orb`):**
    - `'shi_tomasi'` (default) utilizes `goodFeaturesToTrack` and Lucas-Kanade optical flow. Highly accurate but computationally expensive.
    - `'orb'` uses Oriented FAST and Rotated BRIEF descriptors. Highly optimized, extremely fast, and highly recommended for edge devices like Raspberry Pi.
- **Dynamic Feature Re-Detection:** The algorithm actively monitors tracking health. If the camera pans rapidly or orientation changes cause the algorithm to lose 20% of its initial tracking features, it proactively re-detects features on the current frame to maintain a high-quality tracking lock.
- **Smoothing Factor (0.0 to 1.0):** Controls the measurement noise covariance of the Kalman Filter.
    - Lower values mean more smoothing (less trust in the jittery raw trajectory, acting like a heavier, smoother pan).
    - Higher values mean faster adaptation to camera movements.
- **Configurable Complexity (1 to 10):** Adjust algorithm complexity on a scale from 1 to 10.
    - Higher values track more features and utilize larger optical flow search windows, resulting in higher accuracy and robustness at the cost of higher CPU/GPU overhead.
- **ROI Tracking:** Choose a Region of Interest `(x, y, w, h)` to restrict feature detection to a specific part of the scene.
- **Mask Frame:** Load a static image (matching the video frame size) to use as a tracking mask. Pixels with a non-zero value are tracked. Pixels equal to zero are ignored.
- **Coordinate Mapping:** Supports mapping points bidirectionally. Map a point from the stabilized frame back to the original raw frame, or vice versa.
- **Debug Mode:** Return the stabilized frame with visual green dots painted directly over the currently tracked feature points.

## Usage

### Example 1: Real-Time Frame Stabilization on Edge Devices

You can use the `RealTimeVideoStabilizer` class with `extractor_type='orb'` to stabilize frames efficiently on low-power devices.

```python
import cv2
from video_stabilizer import RealTimeVideoStabilizer

# Initialize video capture (0 for default webcam)
cap = cv2.VideoCapture(0)

# Initialize the stabilizer
# smoothing_factor: 0 to 1
# complexity: 1 to 10
stabilizer = RealTimeVideoStabilizer(smoothing_factor=0.1, complexity=5, extractor_type='orb')

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

### Example 2: Stabilize a Recorded Video with a Tracking Mask and Debug Mode

You can use the `stabilize_video` utility function to process a complete video file, apply an image mask, and turn on debug dots.

```python
import cv2
from video_stabilizer import stabilize_video

input_video = "input.mp4"
output_video = "output_stabilized.mp4"
mask_image_path = "tracking_mask.png" # Create an image where white/non-zero pixels are the trackable areas

print("Stabilizing video with mask and debug mode enabled...")
# Process the video. Debug=True will paint the optical flow feature dots on the output video.
stabilize_video(
    input_video,
    output_video,
    smoothing_factor=0.15,
    complexity=6,
    mask_path=mask_image_path,
    debug=True,
    extractor_type='shi_tomasi'
)
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
    roi=(100, 100, 400, 300),
    extractor_type='shi_tomasi'
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
