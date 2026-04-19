import torch
from scipy.signal import savgol_coeffs
from collections import deque

FILTER_CONFIG = {
    "window_size": 7,  # Must be an odd number (e.g., 5, 7, 11)
    "poly_order": 3,   # Typically less than window_size
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