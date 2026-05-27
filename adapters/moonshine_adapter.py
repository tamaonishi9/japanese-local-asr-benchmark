from __future__ import annotations

import time
from typing import Any

import numpy as np

from adapters.base import BaseAdapter
from engine_result import DerivedEngineInput, EngineResult
from source_audio import SourceAudio

ENGINE_NAME = "moonshine-onnx"


class MoonshineAdapter(BaseAdapter):
    def __init__(
        self,
        profile_id: str,
        model_id: str,
        backend: str = "moonshine",
        language: str = "ja",
    ) -> None:
        self.profile_id = profile_id
        self.model_id = model_id
        self.backend = backend
        self.language = language
        self._model: Any = None
        self._tokenizer: Any = None
        self.model_load_seconds: float = 0.0

    def release(self) -> None:
        # モデルと tokenizer 参照を破棄しメモリを解放する
        self._model = None
        self._tokenizer = None

    def prepare(self) -> None:
        # MoonshineOnnxModel 生成 + tokenizer 読込時間をモデル読込時間として計測する
        # tokenizer はモデルと独立して load_tokenizer() で取得する
        import moonshine_onnx
        from moonshine_onnx import MoonshineOnnxModel

        load_started = time.perf_counter()
        self._model = MoonshineOnnxModel(model_name=self.model_id)
        self._tokenizer = moonshine_onnx.load_tokenizer()
        self.model_load_seconds = time.perf_counter() - load_started

    def run(self, source_audio: SourceAudio) -> EngineResult:
        settings: dict[str, Any] = {
            "language": self.language,
        }

        if self._model is None or self._tokenizer is None:
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

        # moonshine-onnx は 1D float32 numpy 配列（16kHz モノラル）を受け入れる
        prep_started = time.perf_counter()
        audio = source_audio.audio_array
        channel_converted = False
        if audio.ndim == 2:
            # (N, 1) または (N, ch) の 2D 配列をモノラルに変換する
            channel_converted = audio.shape[1] > 1
            audio = audio[:, 0]
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
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
            # generate() は (1, N) バッチ次元付き配列を受け取りトークン列を返す
            # tokenizer.decode_batch() でトークン列を文字列リストに変換する
            inference_started = time.perf_counter()
            tokens = self._model.generate(audio[np.newaxis, :])
            text_list = self._tokenizer.decode_batch(tokens)
            transcript_raw = text_list[0] if text_list else ""
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
