import torch
from scipy.signal import savgol_coeffs
from collections import deque

FILTER_CONFIG = {
    "window_size": 7,  # Must be an odd number (e.g., 5, 7, 11)
    "poly_order": 3,   # Typically less than window_size
    "max_jump": 0.2,     # Max allowed jump in normalized coordinates (e.g., 0.5 means 50% of the frame)
    "max_hold_frames": 60, # Max frames to hold a lost joint before resetting
    "global_confidence_threshold": 0.5,  # Threshold for overall detection confidence
    "joint_confidence_threshold": 0.3,    # Threshold for individual joint confidence
}


class RealTimeSGFilter:
    
    def __init__(self, window_size, poly_order):
        """
        Initializes the vectorized real-time causal Savitzky-Golay filter.
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be an odd number (e.g., 5, 7, 11).")
        if poly_order >= window_size:
            raise ValueError("poly_order must be smaller than window_size.")
            
        self.window_size = window_size
        self.poly_order = poly_order
        self.coeffs = savgol_coeffs(window_size, poly_order, pos=window_size - 1)
        self.buffer = deque(maxlen=window_size)
        self.input_shape = None
        self._coeffs_cache = {}


    def _as_tensor(self, values):
        tensor = values.detach().clone() if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if not tensor.is_floating_point():
            tensor = tensor.to(dtype=torch.float32)
        return tensor


    def _get_coeffs(self, reference_tensor):
        key = (reference_tensor.device.type, reference_tensor.device.index, reference_tensor.dtype)
        coeffs = self._coeffs_cache.get(key)
        if coeffs is None:
            coeffs = torch.as_tensor(self.coeffs, dtype=reference_tensor.dtype, device=reference_tensor.device)
            self._coeffs_cache[key] = coeffs
        return coeffs


    def update(self, new_values):
        """
        Feeds raw features into the filter and returns a smoothed torch tensor.
        """
        new_values = self._as_tensor(new_values)

        if self.input_shape is None:
            self.input_shape = tuple(new_values.shape)
        elif tuple(new_values.shape) != self.input_shape:
            raise ValueError(
                f"Input shape changed from {self.input_shape} to {tuple(new_values.shape)}. "
                "All updates must use the same feature shape."
            )
        
        self.buffer.append(new_values)
        
        if len(self.buffer) < self.window_size:
            return new_values.clone()
            
        buffer_tensor = torch.stack(list(self.buffer), dim=0)
        coeffs = self._get_coeffs(buffer_tensor)
        coeffs = coeffs.view(-1, *([1] * (buffer_tensor.ndim - 1)))
        smoothed_values = (buffer_tensor * coeffs).sum(dim=0)
        
        return smoothed_values


    def reset(self):
        """Clears the buffer for a new task/worker."""
        self.buffer.clear()
        self.input_shape = None
        
        


class PerJointKinematicTracker:
    def __init__(self, sg_filter, num_joints, dims_per_joint, max_jump, max_hold_frames=5,
                 global_confidence_threshold=0.5, joint_confidence_threshold=0.5):
        self.sg_filter = sg_filter
        self.num_joints = num_joints
        self.dims_per_joint = dims_per_joint # 2 for (X,Y), 3 for (X,Y,Z)
        self.max_jump = max_jump
        self.max_hold_frames = max_hold_frames
        self.global_confidence_threshold = global_confidence_threshold  # This can be adjusted based on empirical observations
        self.joint_confidence_threshold = joint_confidence_threshold  # This can be adjusted based on empirical observations
        
        
        self.last_good_joints = None
        self.hold_counters = None
        self.hold_counter = 0  # Counter for how many consecutive frames we've held the last good joints


    def _as_joint_tensor(self, values):
        joints = values.detach().clone() if isinstance(values, torch.Tensor) else torch.as_tensor(values)
        if not joints.is_floating_point():
            joints = joints.to(dtype=torch.float32)
        return joints.reshape(self.num_joints, self.dims_per_joint)


    def _ensure_hold_counters(self, reference_joints):
        if self.hold_counters is None:
            self.hold_counters = torch.zeros(
                self.num_joints,
                dtype=torch.int64,
                device=reference_joints.device,
            )
        elif self.hold_counters.device != reference_joints.device:
            self.hold_counters = self.hold_counters.to(device=reference_joints.device)

    def update(
        self,
        current_features,
        joint_confidence: torch.Tensor | None,
        detection_confidence: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if current_features is None:
            # Complete frame dropout. Forward-fill everything if we can.
            if self.last_good_joints is None:
                return None
            return self.sg_filter.update(self.last_good_joints)

        current_joints = self._as_joint_tensor(current_features)
        self._ensure_hold_counters(current_joints)
        assert self.hold_counters is not None

        if self.last_good_joints is not None and (
            current_joints.device != self.last_good_joints.device
            or current_joints.dtype != self.last_good_joints.dtype
        ):
            current_joints = current_joints.to(
                device=self.last_good_joints.device,
                dtype=self.last_good_joints.dtype,
            )
            self._ensure_hold_counters(current_joints)

        detection_conf = None
        if detection_confidence is not None:
            detection_conf = (
                detection_confidence.detach().clone()
                if isinstance(detection_confidence, torch.Tensor)
                else torch.as_tensor(detection_confidence)
            )
            if not detection_conf.is_floating_point():
                detection_conf = detection_conf.to(dtype=torch.float32)
            detection_conf = detection_conf.reshape(-1)[0].to(
                device=current_joints.device,
                dtype=current_joints.dtype,
            )

        if detection_conf is not None and (
            torch.isnan(detection_conf)
            or detection_conf < self.global_confidence_threshold
        ) and self.hold_counter < self.max_hold_frames:
            self.hold_counter += 1
            if self.last_good_joints is None:
                return None
            return self.sg_filter.update(self.last_good_joints)

        if self.last_good_joints is None:
            self.last_good_joints = current_joints.clone()
            return self.sg_filter.update(current_joints)
        assert self.last_good_joints is not None

        # Calculate per-joint movement in the current tensor space.
        distances = torch.linalg.vector_norm(current_joints - self.last_good_joints, dim=1)
        # print("Distances:", distances.max().item(), distances.mean().item(), distances.min().item())
        glitch_mask = distances > self.max_jump

        patience_expired_mask = self.hold_counters >= self.max_hold_frames
        glitch_mask = glitch_mask & ~patience_expired_mask

        self.hold_counters[glitch_mask] += 1
        self.hold_counters[~glitch_mask] = 0

        current_joints = current_joints.clone()
        ignored_mask = glitch_mask
        current_joints[ignored_mask] = self.last_good_joints[ignored_mask]

        self.last_good_joints = current_joints.clone()

        return self.sg_filter.update(current_joints)

    def reset(self):
        self.last_good_joints = None
        self.hold_counters = None
        self.sg_filter.reset()