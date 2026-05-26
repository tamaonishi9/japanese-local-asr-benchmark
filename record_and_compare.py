from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from adapters.faster_whisper_adapter import FasterWhisperAdapter
from config_loader import apply_cli_overrides, load_config
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
        description="録音した音声を faster-whisper で文字起こしし、比較レポートを出力する。"
    )
    # デフォルトはすべて None。config.toml 値を CLI 引数で上書きする場合のみ非 None になる
    parser.add_argument("--output-dir", type=Path, default=None, help="レポート出力先")
    parser.add_argument("--model", default=None, help="Whisper モデル名")
    parser.add_argument("--language", default=None, help="文字起こし言語コード")
    parser.add_argument("--device", default=None, help="推論デバイス: cpu または cuda")
    parser.add_argument("--compute-type", default=None, help="CTranslate2 計算精度")
    parser.add_argument("--cpu-threads", type=int, default=None, help="CPU 推論スレッド数")
    parser.add_argument("--num-workers", type=int, default=None, help="WhisperModel worker 数")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        cfg = load_config()
    except ValueError as error:
        print(f"error: 設定エラー: {error}", file=sys.stderr)
        return 1

    apply_cli_overrides(cfg, args)

    if not cfg.faster_whisper.enabled:
        print("error: [faster_whisper] enabled = false。実行をスキップ。", file=sys.stderr)
        return 1

    out = cfg.output
    if not (out.save_markdown or out.save_csv or out.save_json or out.copy_markdown_to_clipboard):
        print(
            "error: 全出力無効"
            " (save_markdown/save_csv/save_json/copy_markdown_to_clipboard がすべて false)。",
            file=sys.stderr,
        )
        return 1

    fw = cfg.faster_whisper

    # 録音待ち中にユーザーを待たせないため、モデルを先に読み込む
    print(f"モデル読み込み中: {fw.model} ...")
    adapter = FasterWhisperAdapter(
        model_name=fw.model,
        language=fw.language,
        device=fw.device,
        compute_type=fw.compute_type,
        cpu_threads=fw.cpu_threads,
        num_workers=fw.num_workers,
    )
    try:
        adapter.prepare()
    except Exception as error:
        print(f"error: モデル読み込み失敗: {error}", file=sys.stderr)
        return 1
    print(f"モデル読み込み完了 ({adapter.model_load_seconds:.3f} s)")

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

    pipeline = ComparisonPipeline(adapters=[adapter])
    session = pipeline.run(source_audio)
    # CLI 上書き適用後の実効設定を記録し、JSON 出力から実行条件を再現できるようにする
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
        "engines": {
            "faster_whisper": {
                "enabled": cfg.faster_whisper.enabled,
                "model": cfg.faster_whisper.model,
                "language": cfg.faster_whisper.language,
                "device": cfg.faster_whisper.device,
                "compute_type": cfg.faster_whisper.compute_type,
                "cpu_threads": cfg.faster_whisper.cpu_threads,
                "num_workers": cfg.faster_whisper.num_workers,
            },
        },
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

    result = session.engine_results[0]
    if result.status == "success":
        print(f"推論時間: {result.inference_seconds:.3f} s  RTF: {result.inference_rtf:.3f}")
    else:
        print(f"error: 推論失敗: {result.error_message}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
