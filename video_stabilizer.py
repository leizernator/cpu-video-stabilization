import cv2
import numpy as np

class RealTimeVideoStabilizer:
    def __init__(self, smoothing_factor=0.1, complexity=5, roi=None, mask_frame=None, debug=False, extractor_type='shi_tomasi', loss_threshold=0.5, grid_size=1, translation_limit=None, fov=None, focal_length=None):
        """
        Real-time video stabilizer using feature tracking and Kalman Filter dynamic modeling.

        Args:
            smoothing_factor: A float between 0 and 1. Controls the measurement noise covariance
                              of the Kalman Filter. Lower values mean more smoothing (less trust
                              in the raw trajectory). Higher values mean less smoothing.
            complexity: Algorithm complexity on a scale from 1 to 10. Higher values track
                        more features and use larger optical flow windows, improving
                        accuracy and robustness but increasing CPU/GPU processing overhead.
            roi: Region of Interest to track features within, specified as (x, y, w, h).
                 If None, the default ROI is the full image (entire frame).
            mask_frame: A numpy array representing a 2D image mask (same size as the video frames).
                        Pixels with a non-zero value are tracked. If the mask size does not match
                        the frame size, a ValueError will be raised during processing.
            debug: If True, draws the tracked feature points on the stabilized output frame.
            extractor_type: 'shi_tomasi' (default, uses goodFeaturesToTrack + Lucas-Kanade optical flow)
                            or 'orb' (uses ORB descriptors + BFMatcher). 'orb' is much faster and
                            recommended for low-power edge devices like Raspberry Pi.
            loss_threshold: A float between 0.0 and 1.0 (default 0.5) defining the percentage of
                            initial features that can be lost before the algorithm triggers
                            a full re-detection of features on the frame.
            grid_size: An integer (default 1). If > 1, splits the frame into a `grid_size x grid_size`
                       grid to search for features independently per tile. Helps spread features out.
            translation_limit: A float between 0.0 and 1.0, or None (default). If set, limits the maximum
                               correction translation to this percentage of the frame's width/height.
                               For example, 0.15 limits stabilization shifting to 15% of the frame.
            fov: Optional float representing the diagonal field of view in degrees.
                 Used to approximate focal length for 6 DOF movement estimation.
            focal_length: Optional float representing the camera's focal length in pixels.
                          If provided, takes precedence over fov. Used for 6 DOF movement estimation.
        """
        self.smoothing_factor = max(0.001, min(1.0, float(smoothing_factor)))
        self.roi = roi
        self.mask_frame = mask_frame
        self.debug = debug
        self.extractor_type = extractor_type
        self.loss_threshold = max(0.01, min(1.0, float(loss_threshold)))
        self.grid_size = max(1, int(grid_size))
        self.translation_limit = float(translation_limit) if translation_limit is not None else None
        if self.translation_limit is not None:
            self.translation_limit = max(0.0, min(1.0, self.translation_limit))

        # Configure complexity parameters based on scale 1 to 10
        complexity = max(1, min(10, int(complexity)))

        # Linear interpolation mapped across the 1-10 range:
        self.max_corners = int(50 + (complexity - 1) * (450 / 9))

        self.quality_level = 0.01
        self.min_distance = 10
        win_dim = int(11 + (complexity - 1) * (30 / 9))
        if win_dim % 2 == 0: win_dim += 1
        self.lk_win_size = (win_dim, win_dim)
        self.lk_max_level = int(1 + (complexity - 1) * (4 / 9))

        # ORB specific parameters
        self.orb = None
        self.bf_matcher = None
        if self.extractor_type == 'orb':
            self.orb = cv2.ORB_create(nfeatures=self.max_corners)
            self.bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.prev_gray = None
        self.prev_pts = None
        self.prev_des = None # For ORB
        self.initial_feature_count = 0

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

        self.fov = fov
        self.focal_length = focal_length
        self.movement_6dof = {
            'scene': {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0},
            'camera': {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
        }

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
        self.prev_des = None
        self.initial_feature_count = 0

    def _get_mask(self, gray):
        mask = None
        if self.mask_frame is not None:
            if self.mask_frame.shape[:2] != gray.shape[:2]:
                raise ValueError(f"Mask frame shape {self.mask_frame.shape[:2]} does not match video frame shape {gray.shape[:2]}")
            if len(self.mask_frame.shape) == 3:
                mask = cv2.cvtColor(self.mask_frame, cv2.COLOR_BGR2GRAY)
            else:
                mask = self.mask_frame.copy()
            _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        elif self.roi is not None:
            mask = np.zeros_like(gray)
            x, y, w, h = self.roi
            h_img, w_img = gray.shape
            x = max(0, min(x, w_img))
            y = max(0, min(y, h_img))
            w = max(0, min(w, w_img - x))
            h = max(0, min(h, h_img - y))
            mask[y:y+h, x:x+w] = 255
        return mask

    def _get_good_features(self, gray):
        mask = self._get_mask(gray)
        h, w = gray.shape[:2]

        # If grid_size is 1, search the entire image normally.
        if self.grid_size == 1:
            if self.extractor_type == 'orb':
                self.orb.setMaxFeatures(self.max_corners)
                kp, des = self.orb.detectAndCompute(gray, mask)
                if kp is None or len(kp) == 0:
                    return None, None
                pts = np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 1, 2)
                return pts, des
            else:
                pts = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance,
                    blockSize=3,
                    mask=mask
                )
                return pts, None

        # If grid_size > 1, divide into a grid and search tiles independently
        grid_cols = self.grid_size
        grid_rows = self.grid_size
        cell_w = w // grid_cols
        cell_h = h // grid_rows

        # Calculate target per tile
        target_per_cell = max(1, int(np.ceil(self.max_corners / (grid_cols * grid_rows))))

        all_pts = []
        all_des = []

        for row in range(grid_rows):
            for col in range(grid_cols):
                x1 = col * cell_w
                y1 = row * cell_h
                x2 = w if col == grid_cols - 1 else (col + 1) * cell_w
                y2 = h if row == grid_rows - 1 else (row + 1) * cell_h

                cell_gray = gray[y1:y2, x1:x2]
                cell_mask = mask[y1:y2, x1:x2] if mask is not None else None

                if cell_mask is not None and cv2.countNonZero(cell_mask) == 0:
                    continue

                if self.extractor_type == 'orb':
                    self.orb.setMaxFeatures(target_per_cell)
                    kp, des = self.orb.detectAndCompute(cell_gray, cell_mask)
                    if kp is not None and len(kp) > 0:
                        pts = np.array([p.pt for p in kp], dtype=np.float32).reshape(-1, 1, 2)
                        pts[:, 0, 0] += x1
                        pts[:, 0, 1] += y1
                        all_pts.extend(pts)
                        all_des.extend(des)
                else:
                    pts = cv2.goodFeaturesToTrack(
                        cell_gray,
                        maxCorners=target_per_cell,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance,
                        blockSize=3,
                        mask=cell_mask
                    )
                    if pts is not None and len(pts) > 0:
                        pts[:, 0, 0] += x1
                        pts[:, 0, 1] += y1
                        all_pts.extend(pts)

        if len(all_pts) == 0:
            return None, None

        final_pts = np.array(all_pts, dtype=np.float32)
        final_des = np.array(all_des, dtype=np.uint8) if self.extractor_type == 'orb' else None

        return final_pts, final_des

    def process_frame(self, frame):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.current_frame_shape = frame.shape[:2]
        self.current_M = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)

        # Prepare output frame - only copy if debug is enabled to save significant CPU time
        out_frame = frame.copy() if self.debug else frame

        if self.is_first_frame:
            self.prev_gray = curr_gray
            self.prev_pts, self.prev_des = self._get_good_features(curr_gray)
            self.initial_feature_count = len(self.prev_pts) if self.prev_pts is not None else 0
            self.is_first_frame = False

            # Draw mask overlay for first frame if debug
            if self.debug:
                mask = self._get_mask(curr_gray)
                if mask is not None:
                    red_overlay = np.zeros_like(out_frame)
                    red_overlay[:, :] = [0, 0, 255]
                    invalid_mask = cv2.bitwise_not(mask)
                    alpha = 0.3
                    out_float = out_frame.astype(np.float32)
                    red_float = red_overlay.astype(np.float32)
                    for c in range(3):
                        out_float[:, :, c] = np.where(
                            invalid_mask == 255,
                            out_float[:, :, c] * (1 - alpha) + red_float[:, :, c] * alpha,
                            out_float[:, :, c]
                        )
                    out_frame[:] = np.clip(out_float, 0, 255).astype(np.uint8)
                if self.prev_pts is not None:
                    for pt in self.prev_pts:
                        x, y = pt.ravel()
                        cv2.circle(out_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
            return out_frame

        # Determine if we need to re-initialize due to low feature count (lost 20% of original features)
        reinitialize = False
        if self.prev_pts is None:
            reinitialize = True
        else:
            current_count = len(self.prev_pts)
            if current_count < 10 or current_count < ((1.0 - self.loss_threshold) * self.initial_feature_count):
                reinitialize = True

        if reinitialize:
            self.prev_pts, self.prev_des = self._get_good_features(self.prev_gray)
            self.initial_feature_count = len(self.prev_pts) if self.prev_pts is not None else 0

        if self.prev_pts is None:
            self.prev_gray = curr_gray
            return out_frame

        if self.extractor_type == 'orb':
            mask = self._get_mask(curr_gray)
            curr_kp, curr_des = self.orb.detectAndCompute(curr_gray, mask)
            if curr_des is not None and self.prev_des is not None and len(curr_kp) > 0:
                matches = self.bf_matcher.match(self.prev_des, curr_des)
                matches = sorted(matches, key=lambda x: x.distance)

                prev_pts_good = []
                curr_pts_good = []

                for m in matches:
                    prev_pts_good.append(self.prev_pts[m.queryIdx][0])
                    curr_pts_good.append(curr_kp[m.trainIdx].pt)

                prev_pts_good = np.array(prev_pts_good, dtype=np.float32)
                curr_pts_good = np.array(curr_pts_good, dtype=np.float32)
                curr_pts = curr_pts_good.reshape(-1, 1, 2)
            else:
                prev_pts_good = np.array([])
                curr_pts_good = np.array([])
                curr_des = None
        else:
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

        # Enforce mask actively during tracking
        mask = self._get_mask(curr_gray)
        if mask is not None and len(curr_pts_good) > 0:
            valid_idx = []
            for i, pt in enumerate(curr_pts_good):
                x, y = pt.ravel()
                x_int, y_int = int(x), int(y)
                # Check if point is inside frame bounds
                if 0 <= y_int < mask.shape[0] and 0 <= x_int < mask.shape[1]:
                    # Check if point falls on a valid mask area (non-zero)
                    if mask[y_int, x_int] > 0:
                        valid_idx.append(i)

            prev_pts_good = prev_pts_good[valid_idx]
            curr_pts_good = curr_pts_good[valid_idx]

            if self.extractor_type == 'orb' and curr_des is not None:
                # We need to filter the active matched indices for ORB as well to maintain descriptor parity
                filtered_matches = []
                for idx in valid_idx:
                    if idx < len(matches):
                        filtered_matches.append(matches[idx])
                matches = filtered_matches

        if len(prev_pts_good) < 10:
            self.prev_gray = curr_gray
            self.prev_pts, self.prev_des = self._get_good_features(curr_gray)
            self.initial_feature_count = len(self.prev_pts) if self.prev_pts is not None else 0
            return out_frame

        # Draw features in debug mode
        if self.debug:
            # Overlay semi-transparent red on masked-out (invalid) areas
            mask = self._get_mask(curr_gray)
            if mask is not None:
                # Create a red overlay
                red_overlay = np.zeros_like(out_frame)
                red_overlay[:, :] = [0, 0, 255] # BGR red

                # Apply where mask is 0 (invalid)
                invalid_mask = cv2.bitwise_not(mask)

                # Blend the red overlay with the original frame only in invalid areas
                # alpha controls transparency
                alpha = 0.3

                # Create a float copy for accurate blending
                out_float = out_frame.astype(np.float32)
                red_float = red_overlay.astype(np.float32)

                for c in range(3):
                    out_float[:, :, c] = np.where(
                        invalid_mask == 255,
                        out_float[:, :, c] * (1 - alpha) + red_float[:, :, c] * alpha,
                        out_float[:, :, c]
                    )
                out_frame[:] = np.clip(out_float, 0, 255).astype(np.uint8)


            # Draw tracked points
            for pt in curr_pts_good:
                x, y = pt.ravel()
                cv2.circle(out_frame, (int(x), int(y)), 3, (0, 255, 0), -1)

        m, inliers = cv2.estimateAffinePartial2D(prev_pts_good, curr_pts_good)

        if m is None:

            self.prev_pts, self.prev_des = self._get_good_features(curr_gray)
            self.initial_feature_count = len(self.prev_pts) if self.prev_pts is not None else 0
            self.prev_gray = curr_gray
            return out_frame

        # Filter actively tracked points to only keep inliers determined by estimateAffinePartial2D
        inliers_idx = np.where(inliers == 1)[0]
        curr_pts_good = curr_pts_good[inliers_idx]
        prev_pts_inliers = prev_pts_good[inliers_idx]

        # --- 6 DOF Movement Calculation ---
        h, w = frame.shape[:2]
        cx, cy = w / 2.0, h / 2.0

        # Approximate focal length
        if self.focal_length is not None:
            f = self.focal_length
        elif self.fov is not None:
            diagonal = np.sqrt(w**2 + h**2)
            f = diagonal / (2.0 * np.tan(np.radians(self.fov) / 2.0))
        else:
            f = w  # Default assumption if none provided

        camera_matrix = np.array([[f, 0, cx],
                                  [0, f, cy],
                                  [0, 0, 1]], dtype=np.float64)

        if len(prev_pts_inliers) >= 5:
            # Find essential matrix and recover pose
            E, _ = cv2.findEssentialMat(prev_pts_inliers, curr_pts_good, camera_matrix)
            if E is not None and E.shape == (3, 3):
                _, R, t, _ = cv2.recoverPose(E, prev_pts_inliers, curr_pts_good, camera_matrix)

                # Scene movement
                angles_scene, _, _, _, _, _ = cv2.RQDecomp3x3(R)
                self.movement_6dof['scene'] = {
                    'tx': float(t[0, 0]), 'ty': float(t[1, 0]), 'tz': float(t[2, 0]),
                    'rx': angles_scene[0], 'ry': angles_scene[1], 'rz': angles_scene[2]
                }

                # Camera movement (inverse of scene movement)
                R_cam = R.T
                t_cam = -R_cam @ t
                angles_cam, _, _, _, _, _ = cv2.RQDecomp3x3(R_cam)
                self.movement_6dof['camera'] = {
                    'tx': float(t_cam[0, 0]), 'ty': float(t_cam[1, 0]), 'tz': float(t_cam[2, 0]),
                    'rx': angles_cam[0], 'ry': angles_cam[1], 'rz': angles_cam[2]
                }
        else:
            # Not enough points for 6 DOF
            self.movement_6dof = {
                'scene': {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0},
                'camera': {'tx': 0.0, 'ty': 0.0, 'tz': 0.0, 'rx': 0.0, 'ry': 0.0, 'rz': 0.0}
            }
        # ----------------------------------

        if self.extractor_type == 'orb' and curr_des is not None:
            # We extract ONLY the inliers from the current keypoints and descriptors.
            # This maintains tracking health state properly, so if points are lost,
            # len(self.prev_pts) drops and triggers the 20% re-initialization.
            matched_pts = []
            matched_des = []

            for m_idx in inliers_idx:
                if m_idx < len(matches):
                    train_idx = matches[m_idx].trainIdx
                    matched_pts.append(curr_kp[train_idx].pt)
                    matched_des.append(curr_des[train_idx])

            if len(matched_pts) > 0:
                self.prev_pts = np.array(matched_pts, dtype=np.float32).reshape(-1, 1, 2)
                self.prev_des = np.array(matched_des, dtype=curr_des.dtype)
            else:
                self.prev_pts = curr_pts_good.reshape(-1, 1, 2)
                self.prev_des = curr_des
        else:
            self.prev_pts = curr_pts_good.reshape(-1, 1, 2)

        # Extract transformations decoupled from the top-left origin.
        # We calculate how the *center* of the frame moved to decouple rotation from translation.
        h, w = frame.shape[:2]
        center_x = w / 2.0
        center_y = h / 2.0

        c = np.array([center_x, center_y, 1.0])
        c_new = m.dot(c)
        dx = c_new[0] - center_x
        dy = c_new[1] - center_y

        # estimateAffinePartial2D maps p1 -> p2, but its angle sign is inverted relative to getRotationMatrix2D
        da = -np.arctan2(m[1, 0], m[0, 0])

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

        if self.translation_limit is not None:
            max_dx = w * self.translation_limit
            max_dy = h * self.translation_limit
            diff_x = np.clip(diff_x, -max_dx, max_dx)
            diff_y = np.clip(diff_y, -max_dy, max_dy)

        M = cv2.getRotationMatrix2D((center_x, center_y), np.degrees(diff_a), 1.0)
        M[0, 2] += diff_x
        M[1, 2] += diff_y

        self.current_M = M.copy()
        #
        #print(f"diff_x: {diff_x}, diff_y: {diff_y}, diff_a: {diff_a}")

        stabilized_frame = cv2.warpAffine(out_frame, M, (w, h))

        self.prev_gray = curr_gray

        return stabilized_frame

    def get_movement(self, camera=True):
        """
        Retrieves the 6 Degrees of Freedom (DOF) movement calculated between the previous frame and the current frame.

        Args:
            camera (bool): If True (default), returns the estimated movement of the camera.
                           If False, returns the estimated movement of the scene.

        Returns:
            dict: A dictionary containing the translation direction ('tx', 'ty', 'tz') and
                  rotation angles in degrees ('rx', 'ry', 'rz'). Returns zeros if movement
                  could not be calculated.
        """
        if camera:
            return self.movement_6dof['camera'].copy()
        else:
            return self.movement_6dof['scene'].copy()

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

def stabilize_video(input_path, output_path, smoothing_factor=0.1, complexity=5, roi=None, mask_path=None, debug=False, extractor_type='shi_tomasi', loss_threshold=0.5, grid_size=1, translation_limit=None):
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
        debug=debug,
        extractor_type=extractor_type,
        loss_threshold=loss_threshold,
        grid_size=grid_size,
        translation_limit=translation_limit
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        stabilized_frame = stabilizer.process_frame(frame)
        out.write(stabilized_frame)

    cap.release()
    out.release()
