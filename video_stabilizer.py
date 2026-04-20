import cv2
import numpy as np

class RealTimeVideoStabilizer:
    def __init__(self, smoothing_factor=0.1, complexity=5, roi=None, mask_frame=None, debug=False):
        """
        Real-time video stabilizer using optical flow and Kalman Filter dynamic modeling.

        Args:
            smoothing_factor: A float between 0 and 1. Controls the measurement noise covariance
                              of the Kalman Filter. Lower values mean more smoothing (less trust
                              in the jittery raw trajectory). Higher values mean less smoothing.
            complexity: Algorithm complexity on a scale from 1 to 10. Higher values track
                        more features and use larger optical flow windows, improving
                        accuracy and robustness but increasing CPU/GPU processing overhead.
            roi: Region of Interest to track features within, specified as (x, y, w, h).
                 If None, the default ROI is the full image (entire frame).
            mask_frame: A numpy array representing a 2D image mask (same size as the video frames).
                        Pixels with a non-zero value are tracked. If the mask size does not match
                        the frame size, a ValueError will be raised during processing.
            debug: If True, draws the tracked feature points on the stabilized output frame.
        """
        self.smoothing_factor = max(0.001, min(1.0, float(smoothing_factor)))
        self.roi = roi
        self.mask_frame = mask_frame
        self.debug = debug

        # Configure complexity parameters based on scale 1 to 10
        complexity = max(1, min(10, int(complexity)))

        # Linear interpolation mapped across the 1-10 range:
        self.max_corners = int(50 + (complexity - 1) * (450 / 9))
        self.quality_level = 0.1 - (complexity - 1) * (0.09 / 9)
        self.min_distance = int(40 - (complexity - 1) * (30 / 9))
        win_dim = int(11 + (complexity - 1) * (30 / 9))
        if win_dim % 2 == 0: win_dim += 1
        self.lk_win_size = (win_dim, win_dim)
        self.lk_max_level = int(1 + (complexity - 1) * (4 / 9))

        self.prev_gray = None
        self.prev_pts = None

        # Cumulative transformations (trajectory)
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0 # angle

        # Kalman Filter setup
        # State vector: [x, y, a, dx, dy, da]
        # Measurement vector: [x, y, a]
        self.kalman = cv2.KalmanFilter(6, 3, 0)

        # Transition matrix (constant velocity model)
        # x_k = x_{k-1} + dx_{k-1} * dt
        dt = 1.0
        self.kalman.transitionMatrix = np.array([
            [1, 0, 0, dt, 0,  0 ],
            [0, 1, 0, 0,  dt, 0 ],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0 ],
            [0, 0, 0, 0,  1,  0 ],
            [0, 0, 0, 0,  0,  1 ]
        ], np.float32)

        # Measurement matrix (we only measure [x, y, a])
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        # Process noise covariance
        # How much we expect the true state to change.
        self.kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4

        # Measurement noise covariance
        # How much we trust the optical flow measurements.
        # smoothing_factor -> 0 means high noise (trust model, heavy smoothing).
        # smoothing_factor -> 1 means low noise (trust measurement, light smoothing).
        noise_level = 1.0 / (self.smoothing_factor ** 2)
        self.kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * noise_level

        # Error covariance
        self.kalman.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

        # Initial State
        self.kalman.statePost = np.zeros((6, 1), np.float32)

        self.is_first_frame = True
        self.current_M = None
        self.current_frame_shape = None

    def set_roi(self, roi):
        """
        Update the Region of Interest (ROI) for feature tracking.
        roi: (x, y, w, h)
        """
        self.roi = roi
        self.prev_pts = None

    def set_mask(self, mask_frame):
        """
        Set a specific image mask for feature tracking.
        """
        self.mask_frame = mask_frame
        self.prev_pts = None

    def _get_good_features(self, gray):
        # Base mask handling
        mask = None

        if self.mask_frame is not None:
            # Check mask shape
            if self.mask_frame.shape[:2] != gray.shape[:2]:
                raise ValueError(f"Mask frame shape {self.mask_frame.shape[:2]} does not match video frame shape {gray.shape[:2]}")

            # Convert mask to grayscale if it's not already
            if len(self.mask_frame.shape) == 3:
                mask = cv2.cvtColor(self.mask_frame, cv2.COLOR_BGR2GRAY)
            else:
                mask = self.mask_frame.copy()

            # Ensure it's a binary mask where non-zero is 255
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        elif self.roi is not None:
            # Use ROI if no mask frame is provided
            mask = np.zeros_like(gray)
            x, y, w, h = self.roi
            h_img, w_img = gray.shape
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(0, min(w, w_img - x))
            h = max(0, min(h, h_img - y))
            mask[y:y+h, x:x+w] = 255

        pts = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=self.max_corners,
                                      qualityLevel=self.quality_level,
                                      minDistance=self.min_distance,
                                      blockSize=3,
                                      mask=mask)
        return pts

    def process_frame(self, frame):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_frame_shape = frame.shape[:2]
        self.current_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Prepare output frame
        out_frame = frame.copy()

        if self.is_first_frame:
            self.prev_gray = curr_gray
            self.prev_pts = self._get_good_features(curr_gray)
            self.is_first_frame = False
            return out_frame

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = self._get_good_features(self.prev_gray)

        if self.prev_pts is None:
            self.prev_gray = curr_gray
            return out_frame

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray, self.prev_pts, None,
            winSize=self.lk_win_size,
            maxLevel=self.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        if curr_pts is not None and status is not None:
            idx = np.where(status == 1)[0]
            prev_pts_good = self.prev_pts[idx]
            curr_pts_good = curr_pts[idx]
        else:
            prev_pts_good = np.array([])
            curr_pts_good = np.array([])

        if len(prev_pts_good) < 10:
            self.prev_gray = curr_gray
            self.prev_pts = self._get_good_features(curr_gray)
            return out_frame

        # Draw features in debug mode
        if self.debug:
            for pt in curr_pts_good:
                cv2.circle(out_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        m, inliers = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)

        if m is None:
            self.prev_pts = self._get_good_features(curr_gray)
            self.prev_gray = curr_gray
            return out_frame

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        self.x += dx
        self.y += dy
        self.a += da

        # Kalman Filter prediction
        prediction = self.kalman.predict()

        # Kalman Filter update (Correction)
        measurement = np.array([[np.float32(self.x)], [np.float32(self.y)], [np.float32(self.a)]])
        self.kalman.correct(measurement)

        # The estimated smooth state
        smoothed_x = self.kalman.statePost[0, 0]
        smoothed_y = self.kalman.statePost[1, 0]
        smoothed_a = self.kalman.statePost[2, 0]

        # Difference
        diff_x = smoothed_x - self.x
        diff_y = smoothed_y - self.y
        diff_a = smoothed_a - self.a

        h, w = frame.shape[:2]
        center_x = w / 2
        center_y = h / 2

        M = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(diff_a), 1.0)
        M[0, 2] += diff_x
        M[1, 2] += diff_y

        self.current_M = M.copy()

        stabilized_frame = cv2.warpAffine(out_frame, M, (w, h))

        self.prev_gray = curr_gray
        self.prev_pts = curr_pts_good.reshape(-1, 1, 2)

        return stabilized_frame

    def get_stabilized_coordinates(self, orig_x, orig_y):
        if self.current_M is None:
            return float(orig_x), float(orig_y)
        pt = np.array([orig_x, orig_y, 1.0], dtype=np.float64)
        stab_pt = self.current_M.dot(pt)
        return stab_pt[0], stab_pt[1]

    def get_original_coordinates(self, x, y):
        if self.current_M is None:
            return float(x), float(y)
        inv_M = cv2.invertAffineTransform(self.current_M)
        pt = np.array([x, y, 1.0], dtype=np.float64)
        orig_pt = inv_M.dot(pt)
        return orig_pt[0], orig_pt[1]

def stabilize_video(input_path, output_path, smoothing_factor=0.1, complexity=5, roi=None, mask_path=None, debug=False):
    """
    Receives a recorded video and saves a stabilized version of it.
    """
    mask_frame = None
    if mask_path:
        mask_frame = cv2.imread(mask_path)
        if mask_frame is None:
            print(f"Warning: Could not read mask file {mask_path}. Ignoring mask.")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error opening video file {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    stabilizer = RealTimeVideoStabilizer(
        smoothing_factor=smoothing_factor,
        complexity=complexity,
        roi=roi,
        mask_frame=mask_frame,
        debug=debug
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stabilized_frame = stabilizer.process_frame(frame)
        out.write(stabilized_frame)

    cap.release()
    out.release()
