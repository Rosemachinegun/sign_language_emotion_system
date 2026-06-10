from __future__ import annotations

import numpy as np


class FeatureFusion:
    """多模态特征融合（基线版：拼接 + 时序池化）。"""

    def fuse(self, frame_features: np.ndarray, keypoint_features: np.ndarray) -> np.ndarray:
        if frame_features.shape[0] != keypoint_features.shape[0]:
            raise ValueError("视觉特征与关键点特征时间步不一致")

        fused = np.concatenate([frame_features, keypoint_features], axis=1)
        # 简化时序建模：拼接均值、方差、最大值
        mean = fused.mean(axis=0)
        std = fused.std(axis=0)
        maxv = fused.max(axis=0)
        return np.concatenate([mean, std, maxv], axis=0)
