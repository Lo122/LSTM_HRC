"""Generate print-ready checkerboard + ArUco marker images at true physical scale.

Sized to match the defaults used in your intrinsic_calibration.py /
extrinsic_calibration.py scripts:
  - checkerboard: 9x6 internal corners, 25mm squares  (--board-cols 9 --board-rows 6 --square-size-mm 25)
  - ArUco marker: DICT_5X5_50, id 0, 100mm side        (--aruco-dict DICT_5X5_50 --marker-id 0 --marker-length-mm 100)

Images are saved at 300 DPI with DPI metadata embedded, so if you print at
"Actual Size" / 100% scale (NOT "Fit to page"), the physical dimensions will
match exactly. Always verify with a ruler after printing before calibrating.

Usage (defaults match the calibration scripts' example commands):
    python generate_calibration_targets.py --outdir ./calib_targets

Customize if you used different args in your calibration scripts:
    python generate_calibration_targets.py --outdir ./calib_targets \
        --board-cols 9 --board-rows 6 --square-size-mm 25 \
        --aruco-dict DICT_5X5_50 --marker-id 0 --marker-length-mm 100
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DPI = 300
MM_PER_INCH = 25.4


def mm_to_px(mm, dpi=DPI):
    return int(round(mm / MM_PER_INCH * dpi))


def generate_checkerboard(cols, rows, square_mm, margin_mm=15, dpi=DPI):
    """cols/rows = internal corners (cv2 convention) -> squares = cols+1 x rows+1."""
    squares_x = cols + 1
    squares_y = rows + 1
    sq_px = mm_to_px(square_mm, dpi)
    margin_px = mm_to_px(margin_mm, dpi)

    board_w = squares_x * sq_px
    board_h = squares_y * sq_px
    img = np.full((board_h + 2 * margin_px, board_w + 2 * margin_px), 255, dtype=np.uint8)

    for r in range(squares_y):
        for c in range(squares_x):
            if (r + c) % 2 == 0:
                y0 = margin_px + r * sq_px
                x0 = margin_px + c * sq_px
                img[y0:y0 + sq_px, x0:x0 + sq_px] = 0

    total_w_mm = squares_x * square_mm + 2 * margin_mm
    total_h_mm = squares_y * square_mm + 2 * margin_mm
    return img, total_w_mm, total_h_mm


def generate_aruco_marker(dict_name, marker_id, marker_mm, margin_mm=20, dpi=DPI):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    marker_px = mm_to_px(marker_mm, dpi)
    marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_px)

    margin_px = mm_to_px(margin_mm, dpi)
    canvas = np.full((marker_px + 2 * margin_px, marker_px + 2 * margin_px), 255, dtype=np.uint8)
    canvas[margin_px:margin_px + marker_px, margin_px:margin_px + marker_px] = marker_img

    total_mm = marker_mm + 2 * margin_mm
    return canvas, total_mm


def add_label(img_gray, lines, dpi=DPI):
    """Add a text footer with calibration parameters, for reference when printed."""
    from PIL import ImageDraw, ImageFont
    pil_img = Image.fromarray(img_gray).convert("RGB")
    footer_h = mm_to_px(22, dpi)
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h + footer_h), "white")
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm_to_px(3.2, dpi))
    except Exception:
        font = ImageFont.load_default()
    y = h + mm_to_px(3, dpi)
    for line in lines:
        draw.text((mm_to_px(5, dpi), y), line, fill="black", font=font)
        y += mm_to_px(4.5, dpi)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--board-cols", type=int, default=9)
    p.add_argument("--board-rows", type=int, default=6)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    p.add_argument("--marker-id", type=int, default=0)
    p.add_argument("--marker-length-mm", type=float, default=100.0)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # --- Checkerboard ---
    board_img, board_w_mm, board_h_mm = generate_checkerboard(
        args.board_cols, args.board_rows, args.square_size_mm)
    labeled = add_label(board_img, [
        f"Checkerboard: {args.board_cols}x{args.board_rows} internal corners, "
        f"{args.square_size_mm:.1f}mm squares",
        f"Printed size should measure {board_w_mm:.1f}mm x {board_h_mm:.1f}mm "
        f"(WxH) -- verify with a ruler.",
        "PRINT AT 100% / 'ACTUAL SIZE' -- do NOT use 'fit to page'. Mount on a flat, rigid surface.",
    ])
    board_path = outdir / ("checkerboard_%dx%d_%dmm.png" % (
        args.board_cols, args.board_rows, int(args.square_size_mm)))
    labeled.save(board_path, dpi=(DPI, DPI))
    print(f"Saved checkerboard -> {board_path}  ({board_w_mm:.1f}mm x {board_h_mm:.1f}mm pattern area)")

    # --- ArUco marker ---
    marker_img, marker_total_mm = generate_aruco_marker(
        args.aruco_dict, args.marker_id, args.marker_length_mm)
    labeled_marker = add_label(marker_img, [
        f"ArUco marker: {args.aruco_dict}, id={args.marker_id}, {args.marker_length_mm:.1f}mm side",
        f"Printed marker (black square, excluding white margin) should measure "
        f"{args.marker_length_mm:.1f}mm x {args.marker_length_mm:.1f}mm.",
        "PRINT AT 100% / 'ACTUAL SIZE'. Keep the white border -- it's required for detection. "
        "Mount flat.",
    ])
    marker_path = outdir / ("aruco_%s_id%d_%dmm.png" % (
        args.aruco_dict, args.marker_id, int(args.marker_length_mm)))
    labeled_marker.save(marker_path, dpi=(DPI, DPI))
    print(f"Saved ArUco marker -> {marker_path}  ({args.marker_length_mm:.1f}mm side)")


if __name__ == "__main__":
    main()
