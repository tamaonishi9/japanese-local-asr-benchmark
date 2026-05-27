from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from adapters.base import BaseAdapter
from adapters.faster_whisper_adapter import FasterWhisperAdapter
from adapters.moonshine_adapter import MoonshineAdapter
from config_loader import ProfileConfig, apply_cli_overrides, load_config
from output import (
    copy_to_clipboard,
    make_output_paths,
    render_comparison_markdown,
    write_csv,
    write_json,
    write_md,
)
from pipeline import ComparisonPipeline
from recorder import Recorder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="録音した音声を複数プロファイルで文字起こしし、比較レポートを出力する。"
    )
    # デフォルトはすべて None。config.toml 値を CLI 引数で上書きする場合のみ非 None になる
    parser.add_argument("--output-dir", type=Path, default=None, help="レポート出力先")
    return parser.parse_args()


def _build_adapter(profile: ProfileConfig) -> BaseAdapter:
    # backend 値でアダプタを選択する。未対応 backend は呼び出し元で事前検証済み
    if profile.backend == "moonshine":
        return MoonshineAdapter(
            profile_id=profile.profile_id,
            model_id=profile.model_id,
            backend=profile.backend,
            language=profile.language,
        )
    return FasterWhisperAdapter(
        profile_id=profile.profile_id,
        model_id=profile.model_id,
        backend=profile.backend,
        language=profile.language,
        device=profile.device,
        compute_type=profile.compute_type,
        cpu_threads=profile.cpu_threads,
        num_workers=profile.num_workers,
    )


def main() -> int:
    args = parse_args()

    try:
        cfg = load_config()
    except ValueError as error:
        print(f"error: 設定エラー: {error}", file=sys.stderr)
        return 1

    apply_cli_overrides(cfg, args)

    enabled_profiles = [p for p in cfg.profiles.values() if p.enabled]
    if not enabled_profiles:
        print(
            "error: 有効なプロファイルがありません。"
            "config.toml の [profiles.*] で enabled = true を設定してください。",
            file=sys.stderr,
        )
        return 1

    out = cfg.output
    if not (out.save_markdown or out.save_csv or out.save_json or out.copy_markdown_to_clipboard):
        print(
            "error: 全出力無効"
            " (save_markdown/save_csv/save_json/copy_markdown_to_clipboard がすべて false)。",
            file=sys.stderr,
        )
        return 1

    # 対応 backend を明示する。未知の backend は早期終了してミスリードな結果を防ぐ
    _SUPPORTED_BACKENDS = {"faster_whisper", "moonshine"}
    for profile in enabled_profiles:
        if profile.backend not in _SUPPORTED_BACKENDS:
            print(
                f"error: [{profile.profile_id}] backend={profile.backend!r} は未対応。"
                f"対応 backend: {sorted(_SUPPORTED_BACKENDS)}",
                file=sys.stderr,
            )
            return 1

    # pipeline.run() 内で順次 prepare→run→release する。同時保持による RAM/VRAM 合算を防ぐ
    adapters: list[BaseAdapter] = [_build_adapter(p) for p in enabled_profiles]

    print(f"{len(adapters)} プロファイル 順次実行: {[p.profile_id for p in enabled_profiles]}")

    recorder = Recorder(
        sample_rate=cfg.audio.sample_rate,
        channels=cfg.audio.channels,
        dtype=cfg.audio.dtype,
    )

    input("Enter キーで録音開始: ")
    recorder.start()
    print("録音中 ... Enter キーで停止。")
    input()

    source_audio = recorder.stop_as_source_audio()
    if source_audio is None:
        print("error: 音声が録音されませんでした。", file=sys.stderr)
        return 1
    print(f"録音完了: {source_audio.duration_seconds:.3f} s")

    pipeline = ComparisonPipeline(adapters=adapters)
    session = pipeline.run(source_audio)

    # CLI 上書き適用後の実効設定を記録し、JSON 出力から実行条件を再現できるようにする
    profiles_snapshot: dict[str, Any] = {}
    for profile in enabled_profiles:
        profiles_snapshot[profile.profile_id] = {
            "model_id": profile.model_id,
            "backend": profile.backend,
            "enabled": profile.enabled,
            "language": profile.language,
            "device": profile.device,
            "compute_type": profile.compute_type,
            "cpu_threads": profile.cpu_threads,
            "num_workers": profile.num_workers,
        }
    session.effective_configuration = {
        "audio": {
            "sample_rate": cfg.audio.sample_rate,
            "channels": cfg.audio.channels,
            "dtype": cfg.audio.dtype,
        },
        "output": {
            "output_dir": cfg.output.output_dir,
            "copy_markdown_to_clipboard": cfg.output.copy_markdown_to_clipboard,
            "save_markdown": cfg.output.save_markdown,
            "save_csv": cfg.output.save_csv,
            "save_json": cfg.output.save_json,
        },
        "profiles": profiles_snapshot,
    }

    markdown = render_comparison_markdown(session)

    need_file_output = (
        cfg.output.save_markdown or cfg.output.save_csv or cfg.output.save_json
    )

    # 出力先準備失敗とクリップボードコピーを独立させるため、mkdir 失敗を先に捕捉する
    output_paths: tuple[Path, Path, Path] | None = None
    if need_file_output:
        try:
            output_paths = make_output_paths(Path(cfg.output.output_dir))
        except Exception as error:
            print(f"error: 出力先準備失敗: {error}", file=sys.stderr)

    json_path = md_path = csv_path = None
    md_ok = csv_ok = False

    if output_paths is not None:
        json_path, md_path, csv_path = output_paths
        if cfg.output.save_markdown:
            md_ok = write_md(md_path, session)
        if cfg.output.save_csv:
            csv_ok = write_csv(csv_path, session)

    # 保存失敗・パス未確定でもクリップボードコピーは継続する
    clipboard_ok = False
    if cfg.output.copy_markdown_to_clipboard:
        clipboard_ok = copy_to_clipboard(markdown)

    # output_status: 試みた操作のみ記録する
    output_status: dict[str, Any] = {}
    if cfg.output.save_markdown and md_path is not None:
        output_status["markdown"] = {"path": str(md_path), "success": md_ok}
    if cfg.output.save_csv and csv_path is not None:
        output_status["csv"] = {"path": str(csv_path), "success": csv_ok}
    if cfg.output.copy_markdown_to_clipboard:
        output_status["clipboard"] = {"success": clipboard_ok}

    json_ok = False
    if cfg.output.save_json and output_paths is not None:
        session.output_status = output_status
        json_ok = write_json(json_path, session)

    if output_paths is not None:
        if cfg.output.save_json:
            print(f"JSON:     {json_path}  ({'ok' if json_ok else 'failed'})")
        if cfg.output.save_markdown:
            print(f"Markdown: {md_path}  ({'ok' if md_ok else 'failed'})")
        if cfg.output.save_csv:
            print(f"CSV:      {csv_path}  ({'ok' if csv_ok else 'failed'})")

    if cfg.output.copy_markdown_to_clipboard:
        if clipboard_ok:
            print("比較 Markdown をクリップボードにコピーしました。")
        else:
            print("クリップボードへのコピーに失敗しました。", file=sys.stderr)

    # 全プロファイルの結果サマリーを表示
    any_success = False
    for result in session.engine_results:
        if result.status == "success":
            print(
                f"[{result.profile_id}]"
                f"  load: {result.model_load_seconds:.3f} s"
                f"  推論: {result.inference_seconds:.3f} s"
                f"  RTF: {result.inference_rtf:.3f}"
            )
            any_success = True
        else:
            print(
                f"error: [{result.profile_id}] 推論失敗: {result.error_message}",
                file=sys.stderr,
            )

    return 0 if any_success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n中断しました。", file=sys.stderr)
        raise SystemExit(1)
