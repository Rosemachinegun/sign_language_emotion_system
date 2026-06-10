from dataclasses import dataclass


@dataclass
class PipelineConfig:
    frame_sample_rate: int = 4
    max_frames: int = 64
    min_confidence: float = 0.5
