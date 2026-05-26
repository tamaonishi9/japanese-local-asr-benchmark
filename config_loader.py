from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Python 3.11+ 標準 tomllib、それ以前は tomli（requirements.txt で条件付きインストール）
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        tomllib = None  # type: ignore[assignment]

# 初期版は numpy 入力を前提。faster-whisper は ndarray 入力時リサンプルを通らないため固定値のみ許容
_SUPPORTED_SAMPLE_RATES = frozenset({16000})
_SUPPORTED_CHANNELS = frozenset({1})
_SUPPORTED_DTYPES = frozenset({"float32"})

# スクリプト配置ディレクトリ基準。別ディレクトリからの起動でも同梱 config.toml を解決する
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"


@dataclass
class OutputConfig:
    output_dir: str = "results"
    copy_markdown_to_clipboard: bool = True
    save_markdown: bool = True
    save_csv: bool = True
    save_json: bool = True


@dataclass
class FasterWhisperConfig:
    enabled: bool = True
    model: str = "small"
    language: str = "ja"
    device: str = "cpu"
    compute_type: str = "int8"
    cpu_threads: int | None = None
    num_workers: int = 1


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    faster_whisper: FasterWhisperConfig = field(default_factory=FasterWhisperConfig)


def _validate_audio(audio: AudioConfig) -> None:
    # 未対応値は黙って実行せず停止する。リサンプリング対応は他エンジン追加時に実装する
    errors: list[str] = []
    if audio.sample_rate not in _SUPPORTED_SAMPLE_RATES:
        errors.append(
            f"sample_rate={audio.sample_rate} (サポート: {sorted(_SUPPORTED_SAMPLE_RATES)})"
        )
    if audio.channels not in _SUPPORTED_CHANNELS:
        errors.append(
            f"channels={audio.channels} (サポート: {sorted(_SUPPORTED_CHANNELS)})"
        )
    if audio.dtype not in _SUPPORTED_DTYPES:
        errors.append(
            f"dtype={audio.dtype!r} (サポート: {sorted(_SUPPORTED_DTYPES)})"
        )
    if errors:
        raise ValueError(
            "config.toml [audio] 未対応設定:\n" + "\n".join(f"  {e}" for e in errors)
        )


def _apply_section(obj: Any, raw: dict[str, Any]) -> None:
    for key, value in raw.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> AppConfig:
    cfg = AppConfig()

    if not path.exists():
        _validate_audio(cfg.audio)
        return cfg

    if tomllib is None:
        import sys
        print(
            f"警告: TOML ライブラリ未インストール ({path} を無視)。"
            "pip install tomli を実行してください。",
            file=sys.stderr,
        )
        _validate_audio(cfg.audio)
        return cfg

    with path.open("rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    if "audio" in raw:
        _apply_section(cfg.audio, raw["audio"])
    if "output" in raw:
        _apply_section(cfg.output, raw["output"])
    if "faster_whisper" in raw:
        _apply_section(cfg.faster_whisper, raw["faster_whisper"])

    _validate_audio(cfg.audio)
    return cfg


def apply_cli_overrides(cfg: AppConfig, args: Any) -> None:
    # None は「CLI で未指定」。config 値を上書きしない
    fw = cfg.faster_whisper
    if getattr(args, "model", None) is not None:
        fw.model = args.model
    if getattr(args, "language", None) is not None:
        fw.language = args.language
    if getattr(args, "device", None) is not None:
        fw.device = args.device
    if getattr(args, "compute_type", None) is not None:
        fw.compute_type = args.compute_type
    if getattr(args, "cpu_threads", None) is not None:
        fw.cpu_threads = args.cpu_threads
    if getattr(args, "num_workers", None) is not None:
        fw.num_workers = args.num_workers
    if getattr(args, "output_dir", None) is not None:
        cfg.output.output_dir = str(args.output_dir)
