"""Generate a print-ready ChArUco board sized to fit on A4 paper.

Board: 7x9 squares, 25mm squares, 19mm markers, DICT_5X5_50
  -> pattern area ~175mm x 225mm, fits inside A4 (210mm x 297mm) with margin.

Usage:
    python generate_charuco_board.py --outdir ./calib_targets
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 300
MM_PER_INCH = 25.4


def mm_to_px(mm, dpi=DPI):
    return int(round(mm / MM_PER_INCH * dpi))


def generate_charuco(squares_x, squares_y, square_mm, marker_mm, dict_name, margin_mm=15, dpi=DPI):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_mm / 1000.0,
        marker_mm / 1000.0,
        dictionary,
    )

    board_w_px = mm_to_px(squares_x * square_mm, dpi)
    board_h_px = mm_to_px(squares_y * square_mm, dpi)
    img = board.generateImage((board_w_px, board_h_px), marginSize=0, borderBits=1)

    margin_px = mm_to_px(margin_mm, dpi)
    canvas = np.full((board_h_px + 2 * margin_px, board_w_px + 2 * margin_px), 255, dtype=np.uint8)
    canvas[margin_px:margin_px + board_h_px, margin_px:margin_px + board_w_px] = img

    total_w_mm = squares_x * square_mm + 2 * margin_mm
    total_h_mm = squares_y * square_mm + 2 * margin_mm
    return canvas, total_w_mm, total_h_mm, board


def add_label(img_gray, lines, dpi=DPI):
    pil_img = Image.fromarray(img_gray).convert("RGB")
    footer_h = mm_to_px(24, dpi)
    w, h = pil_img.size
    canvas = Image.new("RGB", (w, h + footer_h), "white")
    canvas.paste(pil_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", mm_to_px(3.0, dpi))
    except Exception:
        font = ImageFont.load_default()
    y = h + mm_to_px(3, dpi)
    for line in lines:
        draw.text((mm_to_px(5, dpi), y), line, fill="black", font=font)
        y += mm_to_px(4.2, dpi)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument("--squares-x", type=int, default=7)
    p.add_argument("--squares-y", type=int, default=9)
    p.add_argument("--square-size-mm", type=float, default=25.0)
    p.add_argument("--marker-size-mm", type=float, default=19.0)
    p.add_argument("--aruco-dict", type=str, default="DICT_5X5_50")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img, w_mm, h_mm, board = generate_charuco(
        args.squares_x, args.squares_y, args.square_size_mm, args.marker_size_mm, args.aruco_dict)

    labeled = add_label(img, [
        f"ChArUco board: {args.squares_x}x{args.squares_y} squares, {args.square_size_mm:.1f}mm squares, "
        f"{args.marker_size_mm:.1f}mm markers, {args.aruco_dict}",
        f"Printed pattern area should measure {w_mm:.1f}mm x {h_mm:.1f}mm -- verify with a ruler.",
        "PRINT AT 100% / 'ACTUAL SIZE' on A4, portrait -- do NOT use 'fit to page'. Mount flat and rigid.",
    ])
    out_path = outdir / ("charuco_%dx%d_%dmm_%s.png" % (
        args.squares_x, args.squares_y, int(args.square_size_mm), args.aruco_dict))
    labeled.save(out_path, dpi=(DPI, DPI))
    print(f"Saved ChArUco board -> {out_path}  ({w_mm:.1f}mm x {h_mm:.1f}mm pattern area, fits A4 portrait)")


if __name__ == "__main__":
    main()
