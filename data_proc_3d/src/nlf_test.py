import torch
import torchvision
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import cv2
import trimesh


# 1. Load the model
print("Loading model...")
model_path = r"C:\Users\Owner\Downloads\models\nlf_l_multi_0.3.2.torchscript"
model = torch.jit.load(model_path).cuda().eval()

image_path = r"C:\Users\Owner\Pictures\Screenshots\Screenshot 2026-06-29 175833.png"

# 2. Load the image and FORCE it to be RGB (3 channels)
print("Loading image...")
image = torchvision.io.read_image(
    image_path, 
    mode=torchvision.io.ImageReadMode.RGB  # <--- This prevents the crash!
)

# Move to GPU
image = image.cuda()
frame_batch = image.unsqueeze(0)

# 3. Run Inference
print("Running inference...")
with torch.inference_mode(), torch.device('cuda'):
    pred = model.detect_smpl_batched(frame_batch)
    
print("Success! 3D Joints Shape:", pred['joints3d'])


# ==========================================
# 1. VISUALIZE IN 3D SPACE (Matplotlib)
# ==========================================
print("Generating 3D plot...")

# Extract the 3D joints for the first person [Batch=0, Person=0]
# Move from GPU to CPU, then convert to a NumPy array
joints_3d = pred['joints3d'][0][0].cpu().numpy()

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')

# NLF uses standard camera coordinates (X right, Y down, Z forward).
# To make the person stand "upright" in Matplotlib, we plot (X, Z, -Y).
joints_3d = np.array(joints_3d)  # Ensure it's a NumPy array
x = joints_3d[:, 0]
y = joints_3d[:, 1]
z = joints_3d[:, 2]

# Scatter plot the joints
ax.scatter(x, z, -y, c='red', marker='o', s=30)

# Fix the aspect ratio so the person doesn't look stretched
max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
mid_x = (x.max()+x.min()) * 0.5
mid_y = (y.max()+y.min()) * 0.5
mid_z = (z.max()+z.min()) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_z - max_range, mid_z + max_range)
ax.set_zlim(-(mid_y + max_range), -(mid_y - max_range))

ax.set_xlabel('X (Left/Right)')
ax.set_ylabel('Z (Depth)')
ax.set_zlabel('Y (Up/Down)')
ax.set_title('Interactive 3D Coordinates')

# This will pause the script and pop up the interactive 3D window
plt.show() 


# ==========================================
# 2. VISUALIZE ON 2D IMAGE (OpenCV)
# ==========================================
# Check if the model also outputted the 2D image projections
if 'joints2d' in pred:
    print("Drawing 2D joints on image...")
    
    # Extract the 2D pixel coordinates
    joints_2d = pred['joints2d'][0][0].cpu().numpy()
    
    # Read the original image
    img = cv2.imread(image_path)
    
    # Loop through every joint and draw a green dot
    for point in joints_2d:
        # Grab the X and Y pixel coordinates
        px, py = int(point[0]), int(point[1])
        
        # Draw a solid green circle (Radius=4, Color=(BGR 0,255,0), Thickness=-1 for solid)
        cv2.circle(img, (px, py), radius=4, color=(0, 255, 0), thickness=-1)
        
    # Show the image in a pop-up window
    cv2.imshow('2D Pose Overlay', img)
    cv2.waitKey(0)       # Wait forever until you press any key
    cv2.destroyAllWindows()
else:
    print("No 2D joints found in the prediction output.")


# Check if the model outputted the dense 3D mesh vertices
if 'vertices' in pred:
    print("Extracting full 3D mesh data...")
    
    # Extract the 6,890 vertices for the first person
    # Move from GPU to CPU, and convert to NumPy
    vertices = pred['vertices'][0][0].cpu().numpy()
    
    # Extract the 3D joints for reference
    joints_3d = pred['joints3d'][0][0].cpu().numpy()
    
    # --- Trimesh Visualization ---
    # 1. Create a point cloud of the skin/surface (colored gray)
    mesh_cloud = trimesh.points.PointCloud(vertices, colors=[150, 150, 150, 255])
    
    # 2. Create a point cloud of the internal joints (colored red)
    joint_cloud = trimesh.points.PointCloud(joints_3d, colors=[255, 0, 0, 255])
    
    # 3. Create a 3D scene and add both clouds
    scene = trimesh.Scene([mesh_cloud, joint_cloud])
    
    # 4. Display the interactive 3D viewer
    print("Opening 3D Mesh Viewer. You can click and drag to rotate!")
    scene.show()
    
else:
    print("Mesh vertices not found in prediction output. Ensure you are using the SMPL version of the model.")