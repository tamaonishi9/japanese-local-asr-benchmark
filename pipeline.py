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

        total = len(self._adapters)
        for i, adapter in enumerate(self._adapters, 1):
            profile_id = getattr(adapter, "profile_id", "")
            model_id = getattr(adapter, "model_id", "")
            print(f"[{i}/{total}] {profile_id} ({model_id}) モデル読み込み中 ...")
            # 1 プロファイル完了後に即時解放し、複数モデル同時保持による RAM/VRAM 合算を防ぐ
            try:
                adapter.prepare()
                load_sec = getattr(adapter, "model_load_seconds", 0.0)
                print(f"  読み込み完了 ({load_sec:.3f} s)  推論中 ...")
                result = adapter.run(source_audio)
                if result.status == "success":
                    print(
                        f"  推論完了"
                        f" ({result.inference_seconds:.3f} s"
                        f"  RTF {result.inference_rtf:.3f})"
                    )
                else:
                    print(f"  エラー: {result.error_message}")
            except Exception as error:
                print(f"  失敗: {error}")
                from engine_result import EngineResult as _ER
                result = _ER(
                    engine_name=type(adapter).__name__,
                    model_name="",
                    profile_id=profile_id,
                    model_id=model_id,
                    backend=getattr(adapter, "backend", ""),
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
            finally:
                adapter.release()
            engine_results.append(result)
            execution_order.append(result.profile_id or result.engine_name)

        return SessionResult(
            schema_version=2,
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
