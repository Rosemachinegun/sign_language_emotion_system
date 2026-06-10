from __future__ import annotations

import numpy as np


class SemanticDecoder:
    """语义解码模块（可替换为真实 LLM 推理）。"""

    def decode(self, fused_feature: np.ndarray) -> str:
        score = float(np.clip(fused_feature.mean(), 0.0, 1.0))

        if score > 0.66:
            return "我正在积极表达一个较明确的内容。"
        if score > 0.33:
            return "我在进行一般性的交流表达。"
        return "我在进行较弱幅度的手语表达。"


class SignPhraseRecognizer:
    """简短手语语句识别（规则版）。"""

    labels = ["点赞", "再见", "我爱你"]

    def predict(self, feature_seq: np.ndarray) -> dict:
        if feature_seq.ndim != 2 or feature_seq.shape[0] < 6:
            return {"label": "未识别", "confidence": 0.0}

        mean = feature_seq.mean(axis=0)
        std = feature_seq.std(axis=0)

        num_hands = float(mean[0])
        openness = float(mean[1])
        spread = float(mean[2])
        extended = float(mean[3])
        wrist_height = float(mean[4])
        motion = float(mean[5])
        ily_shape = float(mean[6]) if mean.shape[0] > 6 else max(0.0, 1.0 - abs(extended - 0.62) / 0.28)
        open_palm = float(mean[7]) if mean.shape[0] > 7 else min(1.0, max(0.0, (extended - 0.78) / 0.22))
        motion_var = float(std[5])

        scores = {"点赞": 0.0, "再见": 0.0, "我爱你": 0.0}

        wave_strength = 0.6 * min(1.0, motion * 45) + 0.4 * min(1.0, motion_var * 140)

        # ILY 手型：食指/小指/拇指伸出，中指/无名指弯曲
        ily_ext_peak = ily_shape
        ily_motion_still = 1.0 - min(1.0, motion * 18)

        # open-palm（再见）常见满伸展并伴随挥手
        open_palm_strength = open_palm

        # 再见：典型特征是挥手，运动幅度和波动较高
        scores["再见"] += 0.55 * wave_strength
        scores["再见"] += 0.30 * open_palm_strength
        scores["再见"] += 0.15 * min(1.0, num_hands + 0.2)

        # 我爱你：更依赖手型，且通常不需要大幅挥动
        scores["我爱你"] += 0.60 * ily_ext_peak
        scores["我爱你"] += 0.30 * ily_motion_still
        scores["我爱你"] += 0.07 * min(1.0, openness * 6.5)
        scores["我爱你"] += 0.03 * min(1.0, spread * 5)

        # 反混淆：若挥手强，则压低 ILY；若 ILY 手型明显且稳定，则压低再见
        scores["我爱你"] *= (1.0 - 0.55 * wave_strength)
        scores["再见"] *= (1.0 - 0.30 * ily_ext_peak * ily_motion_still)

        # 点赞：通常单手、动作较稳，手位较高
        scores["点赞"] += 0.4 * (1.0 - min(1.0, abs(num_hands - 0.5) * 2))
        scores["点赞"] += 0.35 * (1.0 - min(1.0, motion * 18))
        scores["点赞"] += 0.25 * min(1.0, wrist_height * 1.2)

        best_label = max(scores, key=scores.get)
        best_score = float(np.clip(scores[best_label], 0.0, 1.0))
        if best_score < 0.28:
            return {"label": "未识别", "confidence": 0.0}
        confidence = round(0.45 + 0.55 * best_score, 3)
        return {"label": best_label, "confidence": confidence}
