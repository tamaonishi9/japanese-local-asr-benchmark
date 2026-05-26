from __future__ import annotations

import os
import sys
import time
from types import ModuleType, SimpleNamespace
from typing import Any

from adapters.base import BaseAdapter
from engine_result import DerivedEngineInput, EngineResult
from source_audio import SourceAudio

ENGINE_NAME = "faster-whisper"


def _install_av_stub() -> None:
    # faster_whisper が内部で av を import するため、av 未インストール環境向けにスタブを注入
    if "av" in sys.modules:
        return
    av_stub = ModuleType("av")

    class InvalidDataError(Exception):
        pass

    av_stub.error = SimpleNamespace(InvalidDataError=InvalidDataError)
    sys.modules["av"] = av_stub


def _configure_ctranslate2_runtime() -> None:
    # OpenMP ライブラリ重複ロードによるエラーを回避
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class FasterWhisperAdapter(BaseAdapter):
    def __init__(
        self,
        profile_id: str,
        model_id: str,
        backend: str = "faster_whisper",
        language: str = "ja",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int | None = None,
        num_workers: int = 1,
    ) -> None:
        self.profile_id = profile_id
        self.model_id = model_id
        self.backend = backend
        self.language = language
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.num_workers = num_workers
        self._model: Any = None
        self.model_load_seconds: float = 0.0

    def release(self) -> None:
        # WhisperModel 参照を破棄し gc 収集を促す。sequential 実行で 1 モデル分の VRAM/RAM のみ占有する
        self._model = None

    def prepare(self) -> None:
        # モデル読込時間を推論時間と分離して計測する
        _configure_ctranslate2_runtime()
        _install_av_stub()

        from faster_whisper import WhisperModel

        model_kwargs: dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
            "num_workers": self.num_workers,
        }
        if self.cpu_threads is not None:
            model_kwargs["cpu_threads"] = self.cpu_threads

        load_started = time.perf_counter()
        self._model = WhisperModel(self.model_id, **model_kwargs)
        if self.device == "cuda":
            # CUDA 初回カーネル読込・キャッシュ生成を model_load_seconds に吸収し
            # 後続の inference_seconds をプロファイル間で公平に比較できるようにする
            import numpy as np
            _dummy = np.zeros(16000, dtype=np.float32)
            list(self._model.transcribe(_dummy, language=self.language)[0])
        self.model_load_seconds = time.perf_counter() - load_started

    def run(self, source_audio: SourceAudio) -> EngineResult:
        settings: dict[str, Any] = {
            "language": self.language,
            "device": self.device,
            "compute_type": self.compute_type,
            "cpu_threads": self.cpu_threads,
            "num_workers": self.num_workers,
        }

        if self._model is None:
            return EngineResult(
                engine_name=ENGINE_NAME,
                model_name=self.model_id,
                profile_id=self.profile_id,
                model_id=self.model_id,
                backend=self.backend,
                settings=settings,
                status="error",
                transcript_raw="",
                model_load_seconds=0.0,
                preprocessing_seconds=0.0,
                inference_seconds=0.0,
                execution_seconds=0.0,
                inference_rtf=0.0,
                end_to_end_rtf=0.0,
                measurement_scope="inference_only",
                engine_input=None,
                error_message="モデル未準備。prepare() を先に呼ぶ。",
            )

        # faster-whisper はモノラル float32 numpy 配列を受け入れる
        prep_started = time.perf_counter()
        audio = source_audio.audio_array
        channel_converted = False
        if audio.ndim == 2:
            # sounddevice は channels=1 でも (N, 1) の 2D 配列を返す。shape[1] > 1 のみステレオ変換扱い
            channel_converted = audio.shape[1] > 1
            audio = audio[:, 0]
        preprocessing_seconds = time.perf_counter() - prep_started

        engine_input = DerivedEngineInput(
            input_kind="numpy",
            sample_rate=source_audio.sample_rate,
            channels=1,
            dtype=str(audio.dtype),
            resampled=False,
            channel_converted=channel_converted,
            serialized_to_wav=False,
            preprocessing_seconds=preprocessing_seconds,
            preprocessing_note="ステレオ → モノラル変換（ch0）" if channel_converted else "",
        )

        try:
            # segment generator の消費完了まで推論時間に含め、遅延評価されるデコード時間を落とさない
            inference_started = time.perf_counter()
            segments, _info = self._model.transcribe(audio, language=self.language)
            transcript_raw = "".join(seg.text for seg in segments)
            inference_seconds = time.perf_counter() - inference_started
        except Exception as error:
            return EngineResult(
                engine_name=ENGINE_NAME,
                model_name=self.model_id,
                profile_id=self.profile_id,
                model_id=self.model_id,
                backend=self.backend,
                settings=settings,
                status="error",
                transcript_raw="",
                model_load_seconds=self.model_load_seconds,
                preprocessing_seconds=preprocessing_seconds,
                inference_seconds=0.0,
                execution_seconds=preprocessing_seconds,
                inference_rtf=0.0,
                end_to_end_rtf=preprocessing_seconds / source_audio.duration_seconds,
                measurement_scope="inference_only",
                engine_input=engine_input,
                error_message=str(error),
            )

        execution_seconds = preprocessing_seconds + inference_seconds
        inference_rtf = inference_seconds / source_audio.duration_seconds
        end_to_end_rtf = execution_seconds / source_audio.duration_seconds

        return EngineResult(
            engine_name=ENGINE_NAME,
            model_name=self.model_id,
            profile_id=self.profile_id,
            model_id=self.model_id,
            backend=self.backend,
            settings=settings,
            status="success",
            transcript_raw=transcript_raw,
            model_load_seconds=self.model_load_seconds,
            preprocessing_seconds=preprocessing_seconds,
            inference_seconds=inference_seconds,
            execution_seconds=execution_seconds,
            inference_rtf=inference_rtf,
            end_to_end_rtf=end_to_end_rtf,
            measurement_scope="inference_only",
            engine_input=engine_input,
            error_message="",
        )
