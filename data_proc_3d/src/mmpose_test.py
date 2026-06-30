from mmpose.apis import MMPoseInferencer

# 1. Initialize the Inferencer for 3D human pose estimation
# The alias 'human3d' automatically downloads and links three models:
# A bounding box detector -> a 2D pose estimator -> a 3D pose lifter
print("Initializing models... (This may take a moment to download weights on the first run)")
inferencer = MMPoseInferencer(pose3d='human3d')

# 2. Define your input video and output directories
video_path = r"G:\My Drive\University of Stuttgart\ITECH_Thesis\Videos\raw\cam-04\video__cam-04_uid-01_take-01.mp4"      # Replace with your video file name
vis_out_dir = 'output_visualizations'  # The folder where the drawn video will be saved
pred_out_dir = 'output_predictions'    # The folder where the 3D coordinates (JSON) will be saved

# 3. Setup the inference parameters
# The inferencer returns a generator, so it won't process until we loop through it.
result_generator = inferencer(
    video_path,
    show=True,                  # Set to False if you don't want a pop-up window during processing
    vis_out_dir=vis_out_dir,    # Saves the visual skeleton overlay 
    pred_out_dir=pred_out_dir,  # Saves the raw 3D joint coordinates
    thickness=2,                # Visual skeleton line thickness
    radius=3                    # Visual skeleton joint radius
)

# 4. Execute the processing loop frame by frame
print(f"Starting 3D pose estimation on {video_path}...")

for frame_idx, result in enumerate(result_generator):
    # 'result' contains the actual 3D coordinates if you need to manipulate them in real-time
    # e.g., result['predictions'][0][0]['keypoints_3d']
    if frame_idx % 30 == 0:
        print(f"Processed {frame_idx} frames...")

print("Video processing complete! Check your output folders.")