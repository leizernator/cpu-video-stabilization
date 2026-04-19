import cv2
import numpy as np

class RealTimeVideoStabilizer:
    def __init__(self, smoothing_factor=0.1, complexity=5, roi=None):
        """
        Real-time video stabilizer using optical flow and Newtonian physics smoothing.

        Args:
            smoothing_factor: A float between 0 and 1. Controls the stiffness of the
                              tracking camera. Lower values mean more smoothing (slower
                              adaptation to large camera movements, acting like a heavier mass).
                              Higher values mean faster adaptation to camera movements.
            complexity: Algorithm complexity on a scale from 1 to 10. Higher values track
                        more features and use larger optical flow windows, improving
                        accuracy and robustness but increasing CPU/GPU processing overhead.
            roi: Region of Interest to track features within, specified as (x, y, w, h).
                 If None, the default ROI is the full image (entire frame).
        """
        self.smoothing_factor = max(0.0, min(1.0, float(smoothing_factor)))
        self.roi = roi

        # Configure complexity parameters based on scale 1 to 10
        complexity = max(1, min(10, int(complexity)))

        # Linear interpolation mapped across the 1-10 range:
        # max_corners: 50 -> 500
        self.max_corners = int(50 + (complexity - 1) * (450 / 9))

        # quality_level: 0.1 -> 0.01
        self.quality_level = 0.1 - (complexity - 1) * (0.09 / 9)

        # min_distance: 40 -> 10
        self.min_distance = int(40 - (complexity - 1) * (30 / 9))

        # lk_win_size: 11 -> 41 (must be odd numbers)
        win_dim = int(11 + (complexity - 1) * (30 / 9))
        if win_dim % 2 == 0:
            win_dim += 1
        self.lk_win_size = (win_dim, win_dim)

        # lk_max_level: 1 -> 5
        self.lk_max_level = int(1 + (complexity - 1) * (4 / 9))

        self.prev_gray = None
        self.prev_pts = None

        # Cumulative transformations (trajectory)
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0 # angle

        # Smoothed trajectory (kinematic physics model)
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        self.smoothed_a = 0.0

        # Velocities for Newtonian smoothing (polynomial order 2)
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.vel_a = 0.0

        self.is_first_frame = True

        # Store the current transformation matrix (stabilized -> original mapping needs its inverse)
        self.current_M = None
        self.current_frame_shape = None

    def set_roi(self, roi):
        """
        Update the Region of Interest (ROI) for feature tracking.
        roi: (x, y, w, h)
        """
        self.roi = roi
        # Force feature recalculation on next frame
        self.prev_pts = None

    def _get_good_features(self, gray):
        mask = None
        if self.roi is not None:
            # Create a mask to only detect features inside the ROI
            mask = np.zeros_like(gray)
            x, y, w, h = self.roi
            # Ensure ROI is within bounds
            h_img, w_img = gray.shape
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(0, min(w, w_img - x))
            h = max(0, min(h, h_img - y))

            mask[y:y+h, x:x+w] = 255

        # Find features to track
        pts = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=self.max_corners,
                                      qualityLevel=self.quality_level,
                                      minDistance=self.min_distance,
                                      blockSize=3,
                                      mask=mask)
        return pts

    def process_frame(self, frame):
        """
        Receives a video frame, stabilizes it in real time based on historical data,
        and returns the stabilized frame. If there is a large movement, the code
        tracks the movement to its new projection and does not lock onto the original frame.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_frame_shape = frame.shape[:2]

        # Identity matrix fallback
        self.current_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        if self.is_first_frame:
            self.prev_gray = curr_gray
            self.prev_pts = self._get_good_features(curr_gray)
            self.is_first_frame = False
            return frame.copy()

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = self._get_good_features(self.prev_gray)

        if self.prev_pts is None:
            # Cannot find features, just return original frame
            self.prev_gray = curr_gray
            return frame.copy()

        # Calculate optical flow
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        # Filter only valid points
        if curr_pts is not None and status is not None:
            idx = np.where(status == 1)[0]
            prev_pts_good = self.prev_pts[idx]
            curr_pts_good = curr_pts[idx]
        else:
            prev_pts_good = np.array([])
            curr_pts_good = np.array([])

        if len(prev_pts_good) < 10:
            # Not enough points, re-initialize
            self.prev_gray = curr_gray
            self.prev_pts = self._get_good_features(curr_gray)
            return frame.copy()

        # Estimate partial 2D affine transformation (translation + rotation)
        m, inliers = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)

        if m is None:
            self.prev_pts = self._get_good_features(curr_gray)
            self.prev_gray = curr_gray
            return frame.copy()

        # Extract transformations
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        # Accumulate trajectory
        self.x += dx
        self.y += dy
        self.a += da

        # Smooth trajectory using a critically damped Newtonian (mass-spring-damper) model
        # This simulates acceleration (polynomial order 2) for smooth, continuous movement
        # spring constant (k) defines how fast we want to pull towards the actual position.
        # damping (c) is set to 2 * sqrt(k) for critical damping (no oscillation).

        # Map smoothing factor (0 -> 1) to a spring constant
        # higher smoothing = lower spring constant (loose spring, heavy smoothing)
        k = max(0.001, (self.smoothing_factor) ** 2)
        c = 2 * np.sqrt(k)

        # Force/acceleration = spring_force - damping_force
        accel_x = k * (self.x - self.smoothed_x) - c * self.vel_x
        accel_y = k * (self.y - self.smoothed_y) - c * self.vel_y
        accel_a = k * (self.a - self.smoothed_a) - c * self.vel_a

        # Update velocities (Euler integration with dt=1)
        self.vel_x += accel_x
        self.vel_y += accel_y
        self.vel_a += accel_a

        # Update positions
        self.smoothed_x += self.vel_x
        self.smoothed_y += self.vel_y
        self.smoothed_a += self.vel_a

        # Difference between smoothed trajectory and actual trajectory
        diff_x = self.smoothed_x - self.x
        diff_y = self.smoothed_y - self.y
        diff_a = self.smoothed_a - self.a

        # Calculate transform for the current frame
        # We need to apply inverse of the difference to stabilize
        dx_corr = diff_x
        dy_corr = diff_y
        da_corr = diff_a

        # Warp frame
        h, w = frame.shape[:2]

        # Rotate around the center
        center_x = w / 2
        center_y = h / 2

        M = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(da_corr), 1.0)
        M[0, 2] += dx_corr
        M[1, 2] += dy_corr

        self.current_M = M.copy()

        stabilized_frame = cv2.warpAffine(frame, M, (w, h))

        # Update state
        self.prev_gray = curr_gray
        self.prev_pts = curr_pts_good.reshape(-1, 1, 2)

        return stabilized_frame

    def get_stabilized_coordinates(self, orig_x, orig_y):
        """
        Maps a point (orig_x, orig_y) from the original raw frame to the stabilized frame's coordinates.
        Returns (stab_x, stab_y).
        """
        if self.current_M is None:
            return float(orig_x), float(orig_y)

        # Convert to homogeneous coordinate
        pt = np.array([orig_x, orig_y, 1.0], dtype=np.float64)

        # Apply forward transform
        stab_pt = self.current_M.dot(pt)

        return stab_pt[0], stab_pt[1]

    def get_original_coordinates(self, x, y):
        """
        Maps a point (x, y) from the stabilized frame back to the original frame's coordinates.
        Returns (orig_x, orig_y).
        """
        if self.current_M is None:
            return float(x), float(y)

        # We applied M to original frame to get stabilized frame: stabilized_pt = M * original_pt
        # So we need to apply the inverse of M to the stabilized point.
        inv_M = cv2.invertAffineTransform(self.current_M)

        # Convert to homogeneous coordinate
        pt = np.array([x, y, 1.0], dtype=np.float64)

        # Apply inverse transform
        orig_pt = inv_M.dot(pt)

        return orig_pt[0], orig_pt[1]

def stabilize_video(input_path, output_path, smoothing_factor=0.1, complexity=5, roi=None):
    """
    Receives a recorded video and saves a stabilized version of it.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    stabilizer = RealTimeVideoStabilizer(smoothing_factor=smoothing_factor, complexity=complexity, roi=roi)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stabilized_frame = stabilizer.process_frame(frame)
        out.write(stabilized_frame)

    cap.release()
    out.release()
