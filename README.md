# japanese-local-asr-benchmark

録音した音声を `faster-whisper` で文字起こしし、推論時間・RTF・transcript を Markdown / CSV / JSON で記録するツール。現在は `faster-whisper` 単体。将来的に `whisper.cpp` / `Kotoba-Whisper` / `ReazonSpeech` を追加し、同一音声での複数エンジン比較を行う予定。

## 動作環境

- Windows 10 / 11
- Python 3.10 以上

## セットアップ

### 1. 仮想環境の作成

```powershell
python -m venv .venv
```

### 2. 仮想環境の有効化

```powershell
.\.venv\Scripts\Activate.ps1
```

> PowerShell でスクリプト実行が禁止されている場合は先に実行:
>
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3. 依存パッケージのインストール

```powershell
pip install -r requirements.txt
```

## 実行

### 録音して比較する（`record_and_compare.py`）

```powershell
python record_and_compare.py
```

1. モデルが自動でロードされる（初回はダウンロードが発生する）
2. `Enter` キーで録音開始
3. 話し終えたら `Enter` キーで停止
4. 推論結果が `results/` に保存され、比較 Markdown がクリップボードにコピーされる

#### オプション（record_and_compare）

```text
--model         Whisper モデル名（デフォルト: small）
--language      言語コード（デフォルト: ja）
--device        cpu または cuda（デフォルト: cpu）
--compute-type  CTranslate2 計算精度（デフォルト: int8）
--cpu-threads   CPU 推論スレッド数（デフォルト: CTranslate2 自動）
--num-workers   WhisperModel worker 数（デフォルト: 1）
--output-dir    レポート出力先（デフォルト: results/）
```

実行例:

```powershell
python record_and_compare.py --model large-v3 --device cuda
```

## 出力ファイル

`results/` に以下が生成される（タイムスタンプ付きファイル名）:

- `.json` — 完全な実行記録（設定・推論時間・RTF・transcript・出力状態）
- `.md` — 人が読む比較レポート
- `.csv` — 複数回実行の一覧比較向け

## 用語

| 用語 | 説明 |
| --- | --- |
| RTF (Real-Time Factor) | 推論時間 / 音声長。1 未満なら実時間より高速 |
| model_load_seconds | モデル読込時間。RTF に含めない |
| inference_seconds | 推論時間。transcript 回収完了まで |
| measurement_scope | 計測範囲の種別。`inference_only` は純粋な推論区間 |
