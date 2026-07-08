from pathlib import Path

import cv2
import numpy as np


DATA_DIR = Path(__file__).resolve().parent
INPUT_DIR = DATA_DIR / "test" / "images"
OUTPUT_DIR = DATA_DIR / "test" / "re_images"


def recolor_blue_region(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([95, 30, 30], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

    kernel = np.ones((5, 5), np.uint8)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(
        blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    smooth_mask = np.zeros_like(blue_mask)

    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
        hull = cv2.convexHull(contour)
        cv2.drawContours(smooth_mask, [hull], -1, 255, -1)

    output = image.copy()
    output[smooth_mask == 255] = (255, 0, 0)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(INPUT_DIR.glob("*.png")):
        image = cv2.imread(str(image_path))
        if image is None:
            continue

        output = recolor_blue_region(image)
        cv2.imwrite(str(OUTPUT_DIR / image_path.name), output)

    print(f"Processed images saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
