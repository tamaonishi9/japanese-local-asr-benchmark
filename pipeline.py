from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from adapters.base import BaseAdapter
from engine_result import EngineResult
from source_audio import SourceAudio


@dataclass
class SessionResult:
    schema_version: int
    created_at: str
    source_audio: dict[str, Any]
    # 実行順を記録し、後続エンジンの計測に先行エンジンの影響が混入した場合の追跡を可能にする
    execution_order: list[str]
    engine_results: list[EngineResult]
    # MD/CSV/クリップボードの保存・コピー結果。JSON 書き込み直前に確定する
    output_status: dict[str, Any] = field(default_factory=dict)
    # CLI 上書き適用後の実効設定スナップショット。config.toml + CLI args の合成結果を記録する
    effective_configuration: dict[str, Any] = field(default_factory=dict)


class ComparisonPipeline:
    def __init__(self, adapters: list[BaseAdapter]) -> None:
        self._adapters = adapters

    def run(self, source_audio: SourceAudio) -> SessionResult:
        engine_results: list[EngineResult] = []
        execution_order: list[str] = []

        for adapter in self._adapters:
            # 1 エンジン失敗でも他エンジンの結果を継続して回収する
            try:
                result = adapter.run(source_audio)
            except Exception as error:
                from engine_result import EngineResult as _ER
                result = _ER(
                    engine_name=type(adapter).__name__,
                    model_name="",
                    settings={},
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
                    error_message=str(error),
                )
            engine_results.append(result)
            execution_order.append(result.engine_name)

        return SessionResult(
            schema_version=1,
            created_at=datetime.now(timezone.utc).isoformat(),
            source_audio={
                "sample_rate": source_audio.sample_rate,
                "channels": source_audio.channels,
                "dtype": source_audio.dtype,
                "duration_seconds": round(source_audio.duration_seconds, 6),
                "recorded_at": source_audio.recorded_at.isoformat(),
            },
            execution_order=execution_order,
            engine_results=engine_results,
        )
