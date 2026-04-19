import cv2
import numpy as np

class RealTimeVideoStabilizer:
    def __init__(self, smoothing_factor=0.1):
        """
        Real-time video stabilizer using optical flow and moving average.
        smoothing_factor: A float between 0 and 1. Lower values mean more smoothing
                          (slower adaptation to large camera movements).
        """
        self.smoothing_factor = smoothing_factor

        self.prev_gray = None
        self.prev_pts = None

        # Cumulative transformations (trajectory)
        self.x = 0.0
        self.y = 0.0
        self.a = 0.0 # angle

        # Smoothed trajectory
        self.smoothed_x = 0.0
        self.smoothed_y = 0.0
        self.smoothed_a = 0.0

        self.is_first_frame = True

    def _get_good_features(self, gray):
        # Find features to track
        pts = cv2.goodFeaturesToTrack(gray,
                                      maxCorners=200,
                                      qualityLevel=0.01,
                                      minDistance=30,
                                      blockSize=3)
        return pts

    def process_frame(self, frame):
        """
        Receives a video frame, stabilizes it in real time based on historical data,
        and returns the stabilized frame. If there is a large movement, the code
        tracks the movement to its new projection and does not lock onto the original frame.
        """
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, curr_gray, self.prev_pts, None)

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

        # Smooth trajectory using EMA (Exponential Moving Average)
        # This acts as a low-pass filter, allowing large, slow movements
        # but filtering out fast, small jitters.
        self.smoothed_x = self.smoothing_factor * self.x + (1 - self.smoothing_factor) * self.smoothed_x
        self.smoothed_y = self.smoothing_factor * self.y + (1 - self.smoothing_factor) * self.smoothed_y
        self.smoothed_a = self.smoothing_factor * self.a + (1 - self.smoothing_factor) * self.smoothed_a

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

        stabilized_frame = cv2.warpAffine(frame, M, (w, h))

        # Update state
        self.prev_gray = curr_gray
        self.prev_pts = curr_pts_good.reshape(-1, 1, 2)

        return stabilized_frame

def stabilize_video(input_path, output_path):
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
    stabilizer = RealTimeVideoStabilizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stabilized_frame = stabilizer.process_frame(frame)
        out.write(stabilized_frame)

    cap.release()
    out.release()
