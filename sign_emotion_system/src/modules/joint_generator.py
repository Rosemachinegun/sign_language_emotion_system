class JointGenerator:
    """语义-情感联合生成。"""

    def generate(self, semantic_text: str, emotion: dict) -> str:
        label = emotion.get("label", "平静")
        if label in {"高兴", "smile"}:
            return f"{semantic_text} 😊 语气偏积极。"
        if label in {"低落", "sad"}:
            return f"{semantic_text} 😔 语气偏克制。"
        return f"{semantic_text} 🙂 语气较为平稳。"
