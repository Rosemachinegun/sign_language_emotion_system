"""Deprecated module.

手部关键点逻辑已迁移到 `pipeline.py` 中的运行时动态加载实现。
保留该文件仅用于兼容历史引用。
"""


class HandTracker:  # pragma: no cover
    def __init__(self) -> None:
        raise RuntimeError("HandTracker 已弃用，请使用 SignEmotionPipeline.run_webcam()")
