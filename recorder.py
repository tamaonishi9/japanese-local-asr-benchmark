from __future__ import annotations

import threading
from datetime import datetime
from typing import List

import numpy as np
import sounddevice as sd

from source_audio import SourceAudio


class Recorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "float32",
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self._frames: List[np.ndarray] = []
        self._stream = None
        self._lock = threading.Lock()
        self.is_recording = False
        self._started_at: datetime | None = None

    # sounddevice コールバック: 音声フレームをスレッドセーフに蓄積
    def _callback(self, indata, frames, time_info, status) -> None:
        with self._lock:
            self._frames.append(indata.copy())

    def start(self) -> None:
        if self.is_recording:
            return
        with self._lock:
            self._frames = []
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._callback,
        )
        self._stream.start()
        # 録音開始日時を記録し、source audio の recorded_at として使用する
        self._started_at = datetime.now()
        self.is_recording = True

    def stop_as_source_audio(self) -> SourceAudio | None:
        # ストリーム停止と SourceAudio 確定を一括で行い、録音原本を確定する
        if not self.is_recording:
            return None

        recorded_at = self._started_at or datetime.now()

        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        finally:
            self._stream = None
            self.is_recording = False

        with self._lock:
            if not self._frames:
                return None
            audio_array = np.concatenate(self._frames, axis=0)

        duration_seconds = len(audio_array) / self.sample_rate
        return SourceAudio(
            audio_array=audio_array,
            sample_rate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            duration_seconds=duration_seconds,
            recorded_at=recorded_at,
        )
