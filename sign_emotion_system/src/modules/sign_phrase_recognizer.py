from __future__ import annotations

import numpy as np


class SignPhraseRecognizer:
    """简短手语语句识别（规则版）。

    目标短句：你好、谢谢、我爱你、对不起、再见
    说明：该版本为原型规则识别，不是监督训练模型。
    """

    labels = ["你好", "谢谢", "我爱你", "对不起", "再见"]

    def predict(self, feature_seq: np.ndarray) -> dict:
        if feature_seq.ndim != 2 or feature_seq.shape[0] < 6:
            return {"label": "未识别", "confidence": 0.0}

        mean = feature_seq.mean(axis=0)
        std = feature_seq.std(axis=0)

        # 特征定义（来自 pipeline）
        # 0: num_hands_norm, 1: openness, 2: spread,
        # 3: extended_ratio, 4: wrist_height, 5: motion
        num_hands = float(mean[0])
        openness = float(mean[1])
        spread = float(mean[2])
        extended = float(mean[3])
        wrist_height = float(mean[4])
        motion = float(mean[5])
        motion_var = float(std[5])

        # 规则打分
        scores = {
            "你好": 0.0,
            "谢谢": 0.0,
            "我爱你": 0.0,
            "对不起": 0.0,
            "再见": 0.0,
        }

        # 再见：通常双手或单手较大幅度挥动
        scores["再见"] += 0.45 * min(1.0, num_hands + 0.2)
        scores["再见"] += 0.35 * min(1.0, motion * 40)
        scores["再见"] += 0.20 * min(1.0, motion_var * 120)

        # 谢谢：单手较多，靠近口部/胸前的中小幅移动
        scores["谢谢"] += 0.45 * (1.0 - min(1.0, abs(num_hands - 0.5) * 2))
        scores["谢谢"] += 0.25 * min(1.0, max(0.0, 0.06 - abs(motion - 0.06)) / 0.06)
        scores["谢谢"] += 0.30 * min(1.0, wrist_height * 1.2)

        # 我爱你：手势较展开，手指伸展比例偏高，动作中等
        scores["我爱你"] += 0.45 * min(1.0, extended * 1.4)
        scores["我爱你"] += 0.35 * min(1.0, openness * 6)
        scores["我爱你"] += 0.20 * min(1.0, spread * 5)

        # 对不起：动作较小，单手偏多，手位中上
        scores["对不起"] += 0.45 * (1.0 - min(1.0, motion * 20))
        scores["对不起"] += 0.35 * (1.0 - min(1.0, abs(num_hands - 0.5) * 2))
        scores["对不起"] += 0.20 * min(1.0, wrist_height * 1.1)

        # 你好：中等动作，手位较高，开放度中等
        scores["你好"] += 0.40 * min(1.0, wrist_height * 1.4)
        scores["你好"] += 0.30 * min(1.0, max(0.0, 0.08 - abs(motion - 0.08)) / 0.08)
        scores["你好"] += 0.30 * min(1.0, spread * 5)

        best_label = max(scores, key=scores.get)
        best_score = float(np.clip(scores[best_label], 0.0, 1.0))

        # 置信度下限抬高一点，避免总是 0.x 很低
        confidence = round(0.45 + 0.55 * best_score, 3)
        return {"label": best_label, "confidence": confidence}
