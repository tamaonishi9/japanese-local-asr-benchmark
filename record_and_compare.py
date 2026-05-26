from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adapters.faster_whisper_adapter import FasterWhisperAdapter
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
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"), help="レポート出力先"
    )
    parser.add_argument("--model", default="small", help="Whisper モデル名")
    parser.add_argument("--language", default="ja", help="文字起こし言語コード")
    parser.add_argument("--device", default="cpu", help="推論デバイス: cpu または cuda")
    parser.add_argument(
        "--compute-type", default="int8", help="CTranslate2 計算精度"
    )
    parser.add_argument(
        "--cpu-threads", type=int, default=None, help="CPU 推論スレッド数"
    )
    parser.add_argument(
        "--num-workers", type=int, default=1, help="WhisperModel worker 数"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 録音待ち中にユーザーを待たせないため、モデルを先に読み込む
    print(f"モデル読み込み中: {args.model} ...")
    adapter = FasterWhisperAdapter(
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        cpu_threads=args.cpu_threads,
        num_workers=args.num_workers,
    )
    try:
        adapter.prepare()
    except Exception as error:
        print(f"error: モデル読み込み失敗: {error}", file=sys.stderr)
        return 1
    print(f"モデル読み込み完了 ({adapter.model_load_seconds:.3f} s)")

    recorder = Recorder()

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

    markdown = render_comparison_markdown(session)

    # 出力先準備失敗とクリップボードコピーを独立させるため、mkdir 失敗を先に捕捉する
    output_paths: tuple[Path, Path, Path] | None = None
    try:
        output_paths = make_output_paths(args.output_dir)
    except Exception as error:
        print(f"error: 出力先準備失敗: {error}", file=sys.stderr)

    if output_paths is not None:
        json_path, md_path, csv_path = output_paths
        # MD / CSV を先に書き込み、クリップボードコピー後に全結果を JSON へ記録する
        md_ok = write_md(md_path, session)
        csv_ok = write_csv(csv_path, session)
    else:
        json_path = md_path = csv_path = None
        md_ok = csv_ok = False

    # 保存失敗・パス未確定でもクリップボードコピーは継続する
    clipboard_ok = copy_to_clipboard(markdown)

    if output_paths is not None:
        session.output_status = {
            "markdown": {"path": str(md_path), "success": md_ok},
            "csv": {"path": str(csv_path), "success": csv_ok},
            "clipboard": {"success": clipboard_ok},
        }
        json_ok = write_json(json_path, session)
        print(f"JSON:     {json_path}  ({'ok' if json_ok else 'failed'})")
        print(f"Markdown: {md_path}  ({'ok' if md_ok else 'failed'})")
        print(f"CSV:      {csv_path}  ({'ok' if csv_ok else 'failed'})")

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
