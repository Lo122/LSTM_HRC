
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class LabelConfiguration:
    buffer: float = 15.0
    smooth_type: str = "plateau"
    function_type: str = "gaussian"
    


def define_step_label_entry(label_name, frame_size, start_frame, end_frame,
                            buffer: float = 15.0, smooth_type: str = "plateau", function_type: str = "gaussian"):

    frame_size, start_frame, end_frame = _clean_data(frame_size, start_frame, end_frame)
    frame_index = pd.Series(np.arange(frame_size, dtype=np.float64), name=label_name)
    label_values = frame_index.apply(
        lambda frame: label_smooth_function(
            frame=frame,
            start_frame=start_frame,
            end_frame=end_frame,
            buffer=buffer,
            smooth_type=smooth_type,
            function_type=function_type,
        )
    )

    return label_values


def cal_status_progress(label_name, frame_size, start_frame, end_frame, start_progress=0.0, end_progress=100.0):
    
    frame_size, start_frame, end_frame = _clean_data(frame_size, start_frame, end_frame)
    
    frame_index = pd.Series(np.arange(frame_size, dtype=np.float64), name=label_name)
    label_values = frame_index.apply(
        lambda frame: _linear_function(start_frame, end_frame, frame, start_progress, end_progress))
    
    return label_values



def _clean_data(frame_size, start_frame, end_frame):
    frame_size = int(frame_size)
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size}.")

    start_frame = float(start_frame)
    end_frame = float(end_frame)
    if end_frame < start_frame:
        raise ValueError(
            "end_frame must be greater than or equal to start_frame, "
            f"got start_frame={start_frame} and end_frame={end_frame}."
        )
    
    return frame_size, start_frame, end_frame


def label_smooth_function(frame: float, start_frame: float, end_frame: float, 
                          buffer: float = 15.0, smooth_type: str = "plateau",
                          function_type: str = "gaussian") -> float:
    """
    Maps a frame to a [0.0, 1.0] value.
    
    Args:
        buffer: Acts as the 'alfa' scaling factor for the curves.
        smooth_type: "plateau" or "asymmetric_peak".
        function_type: "bezier" (bounded to 0), "exponential", or "gaussian" (asymptotic to 0).
    """
    frame = float(frame)
    start = float(start_frame)
    end   = float(end_frame)
    buffer = float(buffer)

    # 1. Determine Delta (distance from peak) and active Alfa for this specific frame
    delta = 0.0
    active_alfa = buffer
    is_peak = False

    if smooth_type == "plateau":
        if start <= frame <= end:
            is_peak = True
        elif frame < start:
            delta = start - frame
            active_alfa = buffer
        else: # frame > end
            delta = frame - end
            active_alfa = buffer

    elif smooth_type == "asymmetric_peak":
        if frame == start:
            is_peak = True
        elif frame < start:
            delta = start - frame
            active_alfa = buffer
        else: # frame > start
            delta = frame - start
            # Stretch the decay alfa to cover the distance from start to (end + buffer)
            active_alfa = (end + buffer) - start 
            
            # Failsafe to prevent division by zero
            if active_alfa <= 0:
                active_alfa = 1e-6
                
    else:
        raise ValueError(f"Unknown smooth_type: {smooth_type}")


    # 2. Return 1.0 immediately if we are in the peak/plateau zone
    if is_peak:
        return 1.0


    # 3. Apply the chosen mathematical decay curve
    if function_type == "bezier":
        # Strict Boundary: Hits exactly 0.0 when distance exceeds alfa
        if delta >= active_alfa:
            return 0.0
        t = 1.0 - (delta / active_alfa)
        return float(t * t * (3.0 - 2.0 * t))
        
    elif function_type == "exponential":
        # Asymptotic: Immediate sharp drop, trailing off infinitely without touching 0
        return float(np.exp(-delta / active_alfa))
        
    elif function_type == "gaussian":
        # Asymptotic: Smooth bell-curve drop, trailing off infinitely without touching 0
        return float(np.exp(-(delta ** 2) / (2 * (active_alfa ** 2))))
        
    else:
        raise ValueError(f"Unknown function_type: {function_type}")
    
    


def _linear_function(start_frame, end_frame, frame, start_progress=0.0, end_progress=100.0):
    if start_frame <= frame <= end_frame:
        return start_progress + (frame - start_frame) / (end_frame - start_frame) * (end_progress - start_progress)
    else:  # Outside the defined range, return 0.0 (or could be another default value)
        return 0.0