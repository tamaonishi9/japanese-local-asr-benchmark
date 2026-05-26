from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from pipeline import SessionResult


def render_comparison_markdown(session: SessionResult) -> str:
    # JSON と同一の session result から Markdown を生成し、保存版と貼り付け版の差異を防ぐ
    audio = session.source_audio
    lines: list[str] = [
        "# 音声文字起こし比較結果",
        "",
        f"- 録音時間: {audio['duration_seconds']:.3f} sec",
        f"- 実行日時: {session.created_at}",
        "",
    ]

    for result in session.engine_results:
        lines.append(f"## {result.engine_name} / {result.model_name}")
        lines.append("")
        if result.status == "success":
            settings_str = ", ".join(
                f"{k}={v}" for k, v in result.settings.items() if v is not None
            )
            lines += [
                f"- 推論時間: {result.inference_seconds:.3f} sec",
                f"- RTF: {result.inference_rtf:.3f}",
                f"- 計測スコープ: {result.measurement_scope}",
                f"- 設定: {settings_str}",
                "",
                result.transcript_raw or "(空の文字起こし結果)",
            ]
        else:
            lines += [
                f"- ステータス: {result.status}",
                f"- エラー: {result.error_message}",
            ]
        lines.append("")

    return "\n".join(lines)


def _session_to_json_dict(session: SessionResult) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for r in session.engine_results:
        run: dict[str, Any] = {
            "engine_name": r.engine_name,
            "model_name": r.model_name,
            "settings": r.settings,
            "status": r.status,
            "transcript_raw": r.transcript_raw,
            "model_load_seconds": round(r.model_load_seconds, 6),
            "preprocessing_seconds": round(r.preprocessing_seconds, 6),
            "inference_seconds": round(r.inference_seconds, 6),
            "execution_seconds": round(r.execution_seconds, 6),
            "inference_rtf": round(r.inference_rtf, 6),
            "end_to_end_rtf": round(r.end_to_end_rtf, 6),
            "measurement_scope": r.measurement_scope,
            "error_message": r.error_message,
        }
        if r.engine_input is not None:
            run["engine_input"] = {
                "input_kind": r.engine_input.input_kind,
                "sample_rate": r.engine_input.sample_rate,
                "channels": r.engine_input.channels,
                "dtype": r.engine_input.dtype,
                "resampled": r.engine_input.resampled,
                "channel_converted": r.engine_input.channel_converted,
                "serialized_to_wav": r.engine_input.serialized_to_wav,
                "preprocessing_seconds": round(r.engine_input.preprocessing_seconds, 6),
                "preprocessing_note": r.engine_input.preprocessing_note,
            }
        runs.append(run)

    result: dict[str, Any] = {
        "schema_version": session.schema_version,
        "created_at": session.created_at,
        "source_audio": session.source_audio,
        "execution_order": session.execution_order,
        "runs": runs,
    }
    if session.output_status:
        result["output_status"] = session.output_status
    if session.effective_configuration:
        result["effective_configuration"] = session.effective_configuration
    return result


def make_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    """タイムスタンプを共有した json / md / csv パスを確定する。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    prefix = output_dir / f"compare_{ts}"
    return prefix.with_suffix(".json"), prefix.with_suffix(".md"), prefix.with_suffix(".csv")


def write_md(path: Path, session: SessionResult) -> bool:
    try:
        path.write_text(render_comparison_markdown(session), encoding="utf-8", newline="\n")
        return True
    except Exception:
        return False


def write_csv(path: Path, session: SessionResult) -> bool:
    # 1 行 = 1 session × 1 engine result。settings は JSON 文字列列として全情報を保持する
    fieldnames = [
        "created_at",
        "duration_seconds",
        "engine_name",
        "model_name",
        "status",
        "inference_seconds",
        "inference_rtf",
        "measurement_scope",
        "settings",
        "transcript_raw",
        "error_message",
    ]
    rows = [
        {
            "created_at": session.created_at,
            "duration_seconds": session.source_audio["duration_seconds"],
            "engine_name": r.engine_name,
            "model_name": r.model_name,
            "status": r.status,
            "inference_seconds": round(r.inference_seconds, 6),
            "inference_rtf": round(r.inference_rtf, 6),
            "measurement_scope": r.measurement_scope,
            "settings": json.dumps(r.settings, ensure_ascii=False),
            "transcript_raw": r.transcript_raw,
            "error_message": r.error_message,
        }
        for r in session.engine_results
    ]
    try:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return True
    except Exception:
        return False


def write_json(path: Path, session: SessionResult) -> bool:
    # JSON は最後に書き込み、output_status（MD/CSV/クリップボード結果）を含めた完全記録にする
    try:
        path.write_text(
            json.dumps(_session_to_json_dict(session), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        return True
    except Exception:
        return False


def copy_to_clipboard(text: str) -> bool:
    # クリップボード失敗で推論結果や保存結果を破棄しないため、bool を返して呼び出し元で判断する
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False
