from pathlib import Path
from ultralytics import FastSAM
from ultralytics import YOLO
import cv2
import numpy as np
import torch


import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM

# 1. Initialize the combined model
# This automatically downloads Grounding DINO and SAM into your RAM/VRAM.
# (Note: This might take a minute or two the very first time you run it).

root_path = Path(__file__).resolve().parents[1]
model_path = root_path / "model_data" / "yolo26x-seg.pt"

# 2. Define the path to your test data
source_path = root_path / "dataset" / "images_"
source_path = source_path.glob("*.jpg")  # Adjust the pattern if your images have a different extension



def unpack_prediction_output(prediction_output):
    """Handle both tuple and list-of-dicts outputs across LangSAM versions."""
    if isinstance(prediction_output, tuple) and len(prediction_output) == 4:
        return prediction_output

    if isinstance(prediction_output, list):
        if not prediction_output:
            return [], [], [], []
        first = prediction_output[0]
        return (
            first.get("masks", []),
            first.get("boxes", []),
            first.get("phrases", []),
            first.get("logits", []),
        )

    raise TypeError(f"Unexpected LangSAM output type: {type(prediction_output)}")


def langsam_inference(root_path: Path, source_path):

    preview_dir = root_path / "results" / "langsam_preview"
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("Loading Grounding DINO + SAM...")
    model = LangSAM()

    for img_path in source_path:
        # 2. Load your image
        # LangSAM prefers standard PIL (Python Imaging Library) images in RGB format.
        image_pil = Image.open(img_path).convert("RGB")

        # 3. Define your prompt
        # You can use phrases, separated by dots if you are looking for multiple things.
        text_prompt = "timber slat."

        print(f"Searching for: '{text_prompt}'...")

        # 4. Run the Zero-Shot Inference
        # masks:   The pixel-perfect polygons (what you want!)
        # boxes:   The bounding boxes found by DINO
        # phrases: The text it matched to the box
        # logits:  The confidence scores
        # Newer LangSAM versions expect batched inputs ([image], [prompt]).
        prediction_output = model.predict([image_pil], [text_prompt])
        masks, boxes, phrases, logits = unpack_prediction_output(prediction_output)

        if len(masks) == 0:
            print("No timber slats found.")
        else:
            print(f"Success! Found {len(masks)} timber slat(s).")
            
            # 5. Extract the data for Open3D
            # Visualize every detected timber mask, one by one.
            original_cv2_image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
            green_overlay = np.zeros_like(original_cv2_image, dtype=np.uint8)
            green_overlay[:] = (0, 255, 0)

            for idx, mask in enumerate(masks, start=1):
                # Convert tensor/array -> 2D uint8 mask for OpenCV/Open3D.
                if hasattr(mask, "cpu"):
                    mask_numpy = mask.cpu().numpy()
                else:
                    mask_numpy = np.asarray(mask)

                mask_numpy = np.squeeze(mask_numpy)
                if mask_numpy.ndim != 2:
                    raise ValueError(f"Expected 2D mask, got shape {mask_numpy.shape}")

                mask_numpy = (mask_numpy > 0).astype(np.uint8) * 255
                
                if mask_numpy.shape[:2] != original_cv2_image.shape[:2]:
                    mask_numpy = cv2.resize(
                        mask_numpy,
                        (original_cv2_image.shape[1], original_cv2_image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                # Apply the mask to the overlay
                highlighted_wood = cv2.bitwise_and(green_overlay, green_overlay, mask=mask_numpy)
                
                # Blend the overlay with the original image (50% opacity)
                result_image = cv2.addWeighted(original_cv2_image, 1.0, highlighted_wood, 0.5, 0)

                # Show every detection if HighGUI is available; otherwise save each one.
                window_name = f"Zero-Shot Timber Detection ({idx}/{len(masks)})"
                try:
                    cv2.imshow(window_name, result_image)
                    cv2.waitKey(0)
                    cv2.destroyWindow(window_name)
                except cv2.error:
                    output_path = preview_dir / f"{Path(img_path).stem}_mask_{idx:02d}.png"
                    cv2.imwrite(str(output_path), result_image)
                    print(f"OpenCV GUI unavailable. Saved preview to: {output_path}")
        


def yolo_inference(root_path: Path, source_path, model_path: Path):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = YOLO(model_path)
    model.to(device)
    print(f"Running inference on: {device}")

    # 3. Run Inference
    # conf=0.5:  Only keep predictions where the AI is >50% confident
    # save=True: Automatically draws the masks and saves them to 'runs/segment/predict/'
    # show=False: The script creates its own preview window for each processed image
    for img_path in source_path:
        print(f"Processing {img_path}...")
        results = model.predict(source=str(img_path), conf=0.1, save=True, show=False, device=torch.device,
                                project=str(root_path / "results"), name="predictions")

        # 4. Extracting the data for your Robot / Open3D Pipeline
        for result in results:
            print(f"\n--- Processing: {result.path} ---")
            print(result.orig_img.shape)
            # Check if the model actually found any wood in this image
            if result.masks is not None:
                # Extract the segmentation masks as numpy arrays
                # Move tensor-backed masks to CPU when needed before converting to numpy.
                masks = masks_to_numpy(result.masks.data)
                
                # Extract the original raw image
                orig_img = result.orig_img
                
                print(f"Successfully detected {len(masks)} timber piece(s).")

                visualization = build_detection_visualization(result, masks)
                cv2.imshow("Timber Detection Results", visualization)
                
                # --- Bridge to Open3D ---
                # If you want to grab the mask for the FIRST detected piece of wood:
                first_mask = masks[0]
                
                # YOLO sometimes outputs masks at a slightly different resolution for speed.
                # Resize it back to the exact dimensions of your original image/depth map:
                first_mask_resized = cv2.resize(
                    first_mask,
                    (orig_img.shape[1], orig_img.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                
                # Now you can use 'first_mask_resized' to filter your RealSense Depth Map!
                cv2.waitKey(0)  # Wait for a key press to close the preview window
                
            else:
                print("No timber detected in this image.")

        cv2.destroyAllWindows()


def masks_to_numpy(mask_data):
    if hasattr(mask_data, "cpu"):
        return mask_data.cpu().numpy()

    return np.asarray(mask_data)



def build_detection_visualization(result, masks):
    annotated_img = result.plot()
    h, w = annotated_img.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for mask in masks:
        resized_mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        combined_mask = np.maximum(combined_mask, (resized_mask > 0.5).astype(np.uint8) * 255)

    combined_mask_bgr = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    cv2.putText(annotated_img, "Detections", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined_mask_bgr, "Combined Mask", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return np.hstack((annotated_img, combined_mask_bgr))


if __name__ == "__main__":
    # langsam_inference(root_path, source_path)
    yolo_inference(root_path, source_path, model_path)
