from __future__ import annotations

import cv2
import numpy as np


class EmotionCompensator:
    """情感识别与补偿特征生成。"""

    def __init__(self) -> None:
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        # 时序平滑，减少标签抖动
        self.ema_smile = 0.0
        self.ema_frown = 0.0
        self.last_label = "calm"

    @staticmethod
    def _frown_score(roi_gray: np.ndarray) -> float:
        """皱眉近似分数：上脸区域垂直纹理越明显，分数越高。"""
        h, w = roi_gray.shape[:2]
        if h < 30 or w < 30:
            return 0.0

        # 眉间附近（上半脸中间 1/3）
        y1, y2 = int(0.18 * h), int(0.48 * h)
        x1, x2 = int(0.33 * w), int(0.67 * w)
        patch = roi_gray[y1:y2, x1:x2]
        if patch.size == 0:
            return 0.0

        patch = cv2.GaussianBlur(patch, (3, 3), 0)
        grad_x = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        score = float(np.mean(np.abs(grad_x)) / 255.0)
        return float(np.clip(score * 4.5, 0.0, 1.0))

    def infer(self, raw_frames: list[np.ndarray], keypoint_features: np.ndarray) -> dict:
        # 离线模式：退化使用动作强度估计，保持兼容。
        intensity = float(np.clip(keypoint_features[:, 2].mean() * 2.2, 0.0, 1.0))
        label = "smile" if intensity >= 0.55 else "calm"

        return {
            "label": label,
            "confidence": round(0.5 + 0.45 * intensity, 3),
            "intensity": round(intensity, 3),
        }

    def infer_from_frame(self, frame: np.ndarray) -> dict:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(70, 70))

        if len(faces) == 0:
            self.ema_smile *= 0.9
            self.ema_frown *= 0.9
            self.last_label = "calm"
            return {"label": "calm", "confidence": 0.5, "intensity": 0.0, "face_bbox": None}

        x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
        roi_gray = gray[y : y + h, x : x + w]
        smiles = self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.45,
            minNeighbors=12,
            minSize=(18, 18),
        )
        eyes = self.eye_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.15,
            minNeighbors=6,
            minSize=(14, 14),
        )

        smile_ratio = 0.0
        if len(smiles) > 0:
            best_smile = max(smiles, key=lambda b: b[2] * b[3])
            smile_ratio = float(np.clip(best_smile[2] / max(w, 1), 0.0, 1.0))
        frown = self._frown_score(roi_gray)

        # EMA 平滑（当前帧占 35%）
        self.ema_smile = 0.65 * self.ema_smile + 0.35 * smile_ratio
        self.ema_frown = 0.65 * self.ema_frown + 0.35 * frown

        # 判定策略：先 smile，最后 calm
        # smile 更容易触发
        if self.ema_smile >= 0.055:
            self.last_label = "smile"
            confidence = float(np.clip(0.62 + self.ema_smile * 0.45, 0.0, 0.99))
            return {
                "label": "smile",
                "confidence": round(confidence, 3),
                "intensity": round(self.ema_smile, 3),
                "face_bbox": (int(x), int(y), int(w), int(h)),
            }

        # calm：两种分数都低时更明显
        self.last_label = "calm"
        calm_strength = float(np.clip(1.0 - max(self.ema_smile * 2.0, self.ema_frown * 1.6), 0.0, 1.0))
        confidence = float(np.clip(0.62 + 0.3 * calm_strength, 0.0, 0.95))
        return {
            "label": "calm",
            "confidence": round(confidence, 3),
            "intensity": round(calm_strength, 3),
            "face_bbox": (int(x), int(y), int(w), int(h)),
        }

    def draw_face_annotation(self, frame: np.ndarray, emotion_info: dict) -> np.ndarray:
        bbox = emotion_info.get("face_bbox")
        if bbox is None:
            cv2.putText(
                frame,
                "face: not found",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            return frame

        x, y, w, h = bbox
        label = str(emotion_info.get("label", "calm"))
        color = (0, 255, 0) if label == "smile" else (255, 255, 0)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame,
            f"{label}",
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )
        return frame
