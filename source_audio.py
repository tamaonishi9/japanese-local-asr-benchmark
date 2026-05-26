from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np


@dataclass
class SourceAudio:
    # 録音で得た numpy 配列（録音原本）。全 adapter の比較起点
    audio_array: np.ndarray
    sample_rate: int
    channels: int
    dtype: str
    # 全 RTF の共通分母。source audio 確定時に一度だけ計算し、エンジンごとに再計算しない
    duration_seconds: float
    recorded_at: datetime
