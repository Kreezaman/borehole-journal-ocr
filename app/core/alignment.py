from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class AlignmentResult:
    image: np.ndarray
    quality: float
    method: str


class PageAligner:
    def __init__(self, template_rgb: np.ndarray):
        self.template_rgb = template_rgb
        self.height, self.width = template_rgb.shape[:2]
        self.template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
        self.orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
        self.template_kp, self.template_des = self.orb.detectAndCompute(self.template_gray, None)

    def align(self, page_rgb: np.ndarray) -> AlignmentResult:
        page = self._landscape(page_rgb)
        gray = cv2.cvtColor(page, cv2.COLOR_RGB2GRAY)
        kp, des = self.orb.detectAndCompute(gray, None)
        if des is not None and self.template_des is not None:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            pairs = matcher.knnMatch(des, self.template_des, k=2)
            good = [a for a, b in pairs if a.distance < 0.72 * b.distance]
            if len(good) >= 18:
                src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([self.template_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                if matrix is not None and mask is not None:
                    inliers = int(mask.sum())
                    quality = min(1.0, inliers / max(24.0, len(good) * 0.55))
                    warped = cv2.warpPerspective(
                        page,
                        matrix,
                        (self.width, self.height),
                        flags=cv2.INTER_CUBIC,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )
                    return AlignmentResult(warped, quality, "homography")
        resized = cv2.resize(page, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return AlignmentResult(resized, 0.2, "resize")

    @staticmethod
    def _landscape(image: np.ndarray) -> np.ndarray:
        if image.shape[0] > image.shape[1]:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return image


def erase_grid_lines(crop: np.ndarray) -> np.ndarray:
    """Whiten long printed grid lines while retaining handwriting."""
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)[1]
    horizontal = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, crop.shape[1] // 4), 1)),
    )
    vertical = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(8, crop.shape[0] // 2))),
    )
    cleaned = crop.copy()
    cleaned[(horizontal > 0) | (vertical > 0)] = 255
    return cleaned
