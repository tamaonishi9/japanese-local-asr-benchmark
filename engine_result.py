from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DerivedEngineInput:
    # adapter が source audio から engine 仕様に合わせて作成した派生入力の条件
    input_kind: str  # "numpy" | "wav"
    sample_rate: int
    channels: int
    dtype: str
    resampled: bool = False
    channel_converted: bool = False
    serialized_to_wav: bool = False
    preprocessing_seconds: float = 0.0
    preprocessing_note: str = ""


@dataclass
class EngineResult:
    engine_name: str
    model_name: str
    settings: dict[str, Any]
    # "success" | "error" | "skipped"
    status: str
    # 生 transcript。評価前に文字列を加工しない
    transcript_raw: str
    # モデル読込時間。RTF に含めない
    model_load_seconds: float
    preprocessing_seconds: float
    # 純粋な推論時間（measurement_scope が "inference_only" の場合に有効）
    inference_seconds: float
    # adapter 全体の実行時間（前処理・subprocess 起動等を含みうる）
    execution_seconds: float
    # inference_seconds / source_audio.duration_seconds
    inference_rtf: float
    # execution_seconds / source_audio.duration_seconds
    end_to_end_rtf: float
    # "inference_only" | "engine_reported_inference" | "adapter_execution_including_process_overhead"
    measurement_scope: str
    engine_input: DerivedEngineInput | None
    error_message: str = ""
