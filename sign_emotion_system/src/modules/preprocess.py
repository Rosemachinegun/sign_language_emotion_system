from __future__ import annotations

import cv2
import numpy as np

from config import PipelineConfig


class Preprocessor:
    """视频预处理与关键点占位提取。

    说明：该版本未接入 MediaPipe，关键点特征先使用简化统计量占位。
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def extract_features_from_frame(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        frame = cv2.resize(frame, (224, 224))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 视觉特征：亮度均值、标准差、边缘密度
        edges = cv2.Canny(gray, 100, 200)
        visual_feat = np.array([
            gray.mean() / 255.0,
            gray.std() / 255.0,
            (edges > 0).mean(),
        ], dtype=np.float32)

        # 关键点占位特征：用左右区域运动强度近似手部活动
        _h, w = gray.shape
        left = gray[:, : w // 2]
        right = gray[:, w // 2 :]
        kp_feat = np.array([
            left.mean() / 255.0,
            right.mean() / 255.0,
            abs(left.mean() - right.mean()) / 255.0,
        ], dtype=np.float32)

        return frame, visual_feat, kp_feat

    def process_capture(self, cap: cv2.VideoCapture, source_name: str) -> dict:
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开输入源: {source_name}")

        raw_frames: list[np.ndarray] = []
        frame_features: list[np.ndarray] = []
        keypoint_features: list[np.ndarray] = []

        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if index % self.config.frame_sample_rate == 0:
                frame, visual_feat, kp_feat = self.extract_features_from_frame(frame)

                raw_frames.append(frame)
                frame_features.append(visual_feat)
                keypoint_features.append(kp_feat)

                if len(raw_frames) >= self.config.max_frames:
                    break

            index += 1

        if not frame_features:
            raise ValueError("输入源中未采样到有效帧，请检查设备或视频。")

        return {
            "raw_frames": raw_frames,
            "frame_features": np.vstack(frame_features),
            "keypoint_features": np.vstack(keypoint_features),
        }

    def process(self, video_path: str) -> dict:
        cap = cv2.VideoCapture(video_path)
        try:
            return self.process_capture(cap, video_path)
        finally:
            cap.release()
