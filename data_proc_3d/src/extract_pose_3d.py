#https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/python
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_utils
from mediapipe.tasks.python.vision import drawing_styles

import cv2
import numpy as np
from tqdm import tqdm
from filterpy.kalman import KalmanFilter

model_path = r"data_proc_3d\pose_landmarker_full.task"

world_filters = {}
image_filters = {}

class PoseSmoother:

    def __init__(self, fps=30.0):
        self.dt = 1.0 / fps
        self.world_filters = {}
        self.image_filters = {}

    def _create_kf(self, initial_pos):

        kf = KalmanFilter(dim_x=6, dim_z=3)

        kf.F = np.array([[1, 0, 0, self.dt, 0, 0],
                         [0, 1, 0, 0, self.dt, 0],
                         [0, 0, 1, 0, 0, self.dt],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0]])
        
        kf.x[:3] = np.array(initial_pos).reshape(3, 1)
        kf.R *= 0.05  
        kf.Q *= 0.01  
        kf.P *= 10.0  
        return kf

    def smooth_landmarks(self, landmarks, filter_dict):
        if landmarks is None:
            return None
        
        smoothed_data = []
        for i, lm in enumerate(landmarks):
            if i not in filter_dict:
                filter_dict[i] = self._create_kf([lm.x, lm.y, lm.z])
            
            kf = filter_dict[i]
            kf.predict()
            
            # only update if the landmark is visible enough
            if lm.visibility > 0.5:
                kf.update([lm.x, lm.y, lm.z])
            
            # return [x, y, z, vx, vy, vz]
            smoothed_data.append(kf.x.copy().flatten())
        
        return np.array(smoothed_data)

    def process(self, kpts_world, kpts_image):
        """
        input: kpts_world and kpts_image are lists of landmarks for each person detected in the frame. Each landmark has x,y,z,visibility attributes.
        output: smoothed landmarks for world and image coordinates.
        """
        s_world = self.smooth_landmarks(kpts_world, self.world_filters)
        s_image = self.smooth_landmarks(kpts_image, self.image_filters)
        return s_world, s_image

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  pose_landmark_style = drawing_styles.get_default_pose_landmarks_style()
  pose_connection_style = drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)

  for pose_landmarks in pose_landmarks_list:
    drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=pose_landmarks,
        connections=vision.PoseLandmarksConnections.POSE_LANDMARKS,
        landmark_drawing_spec=pose_landmark_style,
        connection_drawing_spec=pose_connection_style)

  return annotated_image

def load_marker(model_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.3)

    landmarker = PoseLandmarker.create_from_options(options)
    return landmarker

def extract_pose_3d(frame, marker):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    pose_landmarker_result = marker.detect(mp_image)
    kpts_world = pose_landmarker_result.pose_world_landmarks
    kpts_image = pose_landmarker_result.pose_landmarks

    world_out = kpts_world[0] if kpts_world and len(kpts_world) > 0 else None
    image_out = kpts_image[0] if kpts_image and len(kpts_image) > 0 else None

    # visualize the landmarks on the image for debugging
    annotated_image = draw_landmarks_on_image(frame, pose_landmarker_result)
    cv2.imshow("Pose", annotated_image)
    cv2.waitKey(1)
    
    return world_out, image_out, pose_landmarker_result

def extract_degree(kpts_world):
    pass

def extract_speed_acc(kpts_image):
    pass

def extract_distance(kpts_world):
    pass

def orgnize_feat():
    pass    

#app test
if __name__ == "__main__":
    vid_path = r"G:\.shortcut-targets-by-id\1nZZWQUKOdxeC-oo-NKucbuUj38ir4mZC\ITECH_Thesis\Videos\cropped\cam-01\video__cam-01_uid-01_take-01.mp4"
    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    marker = load_marker(model_path)

    smoother = PoseSmoother(fps=fps)

    for _ in tqdm(range(total_frames), desc="Extracting pose"):
        ret, frame = cap.read()
        if not ret:
            break

        raw_world, raw_image, result = extract_pose_3d(frame, marker)
        smoothed_world, smoothed_image = smoother.process(raw_world, raw_image)


        if smoothed_world is not None and smoothed_image is not None:
            # extract features from smoothed landmarks
            degree_features = extract_degree(smoothed_world)
            speed_acc_features = extract_speed_acc(smoothed_image)
            distance_features = extract_distance(smoothed_world)

            # save all the features as .pt
            

    
    cap.release() 
    cv2.destroyAllWindows()

# The output contains the following normalized coordinates (Landmarks):

# x and y: Landmark coordinates normalized between 0.0 and 1.0 by the image width (x) and height (y).

# z: The landmark depth, with the depth at the midpoint of the hips as the origin. The smaller the value, the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x.

# visibility: The likelihood of the landmark being visible within the image.

# The output contains the following world coordinates (WorldLandmarks):

# x, y, and z: Real-world 3-dimensional coordinates in meters, with the midpoint of the hips as the origin.

# visibility: The likelihood of the landmark being visible within the image.