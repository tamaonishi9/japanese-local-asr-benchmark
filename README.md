# japanese-local-asr-benchmark

録音した同一音声を複数のローカル ASR profile で文字起こしし、結果テキストと推論時間、RTF を比較するツールです。

現在の `v0.1` 対応範囲は、`faster-whisper` backend 上で動かす次のモデル比較です。

| Profile             | Model                                    | 用途                   |
| ------------------- | ---------------------------------------- | ---------------------- |
| `small`             | Whisper `small`                          | 軽量な動作確認向け     |
| `large_v3`          | Whisper `large-v3`                       | Whisper 系の高精度基準 |
| `kotoba_whisper_v2` | `kotoba-tech/kotoba-whisper-v2.0-faster` | 日本語特化モデル比較   |

結果は Markdown / CSV / JSON に保存でき、Markdown はクリップボードへコピーできます。

まず動かす場合は、[最短手順](docs/最短手順.md) を参照してください。CPU での初回比較と、CUDA を使う場合の追加手順だけをまとめています。

## 特徴

- 1 回の録音から、同一音声を複数 profile へ渡して比較
- profile ごとの文字起こし結果、推論時間、RTF を表示
- `profile_id` / `model_id` / `backend` を JSON に記録
- `config.toml` で利用モデルと出力方法を選択
- 出力を保存せず、クリップボードコピーだけで軽く使う設定にも対応

## 動作環境

- Python 3.10 以上
- マイク入力が使用できる環境

確認済み環境:

- Windows 10 / 11

想定対応環境:

- Windows
- macOS
- Linux

本ツールの現行コードに Windows 固有 API への依存はありません。録音には `sounddevice`、クリップボードコピーには `pyperclip` を使用しており、どちらも複数 OS に対応しています。ただし、macOS / Linux は本リポジトリでの動作確認前のため、確認済み環境とは分けて扱います。

OS ごとの注意:

- Linux では、マイク入力用の PortAudio 環境や、クリップボードコピー用の `xclip` / `xsel` 等が追加で必要になる場合があります。
- macOS / Linux では、仮想環境の有効化コマンドが Windows PowerShell と異なります。
- GPU 実行は OS に関係なく、CTranslate2 に対応する CUDA / cuDNN 構成の確認が必要です。

初期版で対応する音声条件:

```text
sample_rate = 16000
channels = 1
dtype = "float32"
```

現時点で利用できる backend は `faster_whisper` のみです。`backend` 設定へ別の値を書くだけでは、`whisper.cpp` や ReazonSpeech は利用できません。

## セットアップ

### 1. リポジトリを取得

```powershell
git clone <repository-url>
cd japanese-local-asr-benchmark
```

既に配置済みの場合は、プロジェクトディレクトリで以降の操作を行ってください。

### 2. 仮想環境を作成

```powershell
python -m venv .venv
```

### 3. 仮想環境を有効化

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

PowerShell でスクリプト実行が禁止されている場合:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 4. 依存パッケージをインストール

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

主要依存:

- `faster-whisper`: ASR 推論 backend
- `requests`: 初回モデル取得で必要となる依存
- `sounddevice`: マイク録音
- `pyperclip`: Markdown のクリップボードコピー

### 5. 初回モデル取得について

有効化した profile のモデルは、初回実行時に Hugging Face Hub からダウンロードされます。公開モデルは通常、token なしでも取得できます。`large-v3` や Kotoba-Whisper は `small` よりダウンロード容量とメモリ使用量が大きくなります。

初回取得の容量や実行時間を抑えたい場合は、`config.toml` で `small` のみを有効化して動作確認してください。

大容量モデルの取得を安定させたい場合は、任意で Hugging Face の `HF_TOKEN` を環境変数へ設定できます。

Windows PowerShell の現在セッションで設定する例:

```powershell
$env:HF_TOKEN = "hf_xxxxxxxxxxxxxxxxx"
python record_and_compare.py
```

`HF_TOKEN` は秘密情報です。`config.toml`、出力 JSON、README、コミット対象ファイルには記載しないでください。

#### Windows のキャッシュ警告

Windows では、Hugging Face のキャッシュが使用する symlink を作成できず、次の趣旨の警告が表示される場合があります。

```text
cache-system uses symlinks by default ... your machine does not support them
```

この警告が表示されても、モデル取得と推論は継続できます。ただし、キャッシュが重複ファイルを効率よく共有できず、複数モデル利用時のディスク使用量が増える場合があります。

一方、Kotoba-Whisper 等のモデル取得時に次のようなエラーで処理が停止する場合があります。

```text
[WinError 1314] クライアントは要求された特権を保有していません。
```

この場合は、現在の PowerShell セッションで symlink の使用を無効化してから再実行してください。キャッシュはファイルコピーを使うため、管理者権限や開発者モードなしでも取得を継続できます。

```powershell
$env:HF_HUB_DISABLE_SYMLINKS = "1"
python record_and_compare.py
```

`HF_HUB_DISABLE_SYMLINKS = "1"` を使用すると、複数モデル利用時のディスク使用量が増える場合があります。容量効率を優先する場合は、Windows の開発者モードを有効化すると symlink を利用しやすくなります。

警告のみを非表示にする場合は、次の環境変数を設定できます。

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

`HF_HUB_DISABLE_SYMLINKS_WARNING` は警告表示を抑えるだけであり、`WinError 1314` の回避にはなりません。

## 使い方

### 1. CPU / CUDA 設定を選ぶ

実行時に読み込まれる設定ファイルは `config.toml` です。比較用テンプレートとして、次の 2 ファイルを用意しています。

| ファイル | 用途 | 設定 |
| --- | --- | --- |
| `config.toml.cpu` | CPU で 3 モデル比較 | `device = "cpu"`, `compute_type = "int8"` |
| `config.toml.cuda` | NVIDIA GPU で 3 モデル比較 | `device = "cuda"`, `compute_type = "float16"` |

CPU 設定を利用する場合:

```powershell
Copy-Item .\config.toml.cpu .\config.toml -Force
python record_and_compare.py
```

CUDA 設定を利用する場合:

```powershell
Copy-Item .\config.toml.cuda .\config.toml -Force
python record_and_compare.py
```

`config.toml.cuda` の利用には、CTranslate2 に対応する NVIDIA CUDA / cuDNN 環境と十分な VRAM が必要です。GPU 環境が利用できない場合は `config.toml.cpu` を使用してください。

#### CUDA 実行環境

現在の依存構成で入る `CTranslate2 4.7.x` の GPU 実行には、次の NVIDIA ライブラリが必要です。

```text
- CUDA 12.x の cuBLAS
- CUDA 12 対応の cuDNN 9
```

CUDA Toolkit 12.x のインストール後、cuDNN 9 for CUDA 12 は仮想環境へ追加できます。

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install nvidia-cudnn-cu12
```

NVIDIA GPU や GPU ドライバが認識されていても、CUDA 13.x のみが導入されている環境では、CUDA 12 用 DLL が存在せず次のエラーで停止します。

```text
Library cublas64_12.dll is not found or cannot be loaded
```

CUDA 13.x を他用途で利用している場合は削除せず、CUDA 12.x と cuDNN 9 を追加導入して併存させる運用が可能です。このツールを実行する PowerShell では、CUDA 12 の `bin` ディレクトリを `PATH` の先頭へ追加してください。

例:

```powershell
$cudnnBin = (Resolve-Path .\.venv\Lib\site-packages\nvidia\cudnn\bin).Path
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin;$cudnnBin;$env:PATH"
where.exe cublas64_12.dll
where.exe cudnn*.dll
Copy-Item .\config.toml.cuda .\config.toml -Force
python record_and_compare.py
```

`v12.x` は導入した CUDA Toolkit の実際のバージョンへ置き換えてください。`cublas64_12.dll` だけを非公式配布サイトから取得して配置する方法は避け、NVIDIA 公式の CUDA Toolkit / cuDNN 配布物を使用してください。

### 2. 使用する profile を選ぶ

使用モデルは [config.toml](config.toml) の `enabled` で選択します。

CPU / CUDA テンプレートは、次の 3 profile を比較対象として有効化しています。

```toml
[profiles.small]
enabled = true

[profiles.large_v3]
enabled = true

[profiles.kotoba_whisper_v2]
enabled = true
```

#### 軽量な動作確認

```toml
[profiles.small]
enabled = true

[profiles.large_v3]
enabled = false

[profiles.kotoba_whisper_v2]
enabled = false
```

#### `small` と Kotoba-Whisper を比較

```toml
[profiles.small]
enabled = true

[profiles.large_v3]
enabled = false

[profiles.kotoba_whisper_v2]
enabled = true
```

#### 3 モデルを比較

```toml
[profiles.small]
enabled = true

[profiles.large_v3]
enabled = true

[profiles.kotoba_whisper_v2]
enabled = true
```

注意:

- 現在の実装は、録音完了後に有効な profile を順番に `モデル読込 -> 推論 -> 解放` します。
- 複数 profile を有効にしてもモデルを同時保持しない方針ですが、各モデル単体を動かすための RAM / VRAM は必要です。
- 3 モデル比較では profile 数に応じてモデル読込と推論を繰り返すため、完了までの時間が長くなります。
- `large-v3` は特に重いため、CPU のみの環境では時間がかかります。

### 3. 録音して比較

```powershell
python record_and_compare.py
```

実行の流れ:

1. `Enter` キーで録音開始
2. 話し終えたら `Enter` キーで録音停止
3. `enabled = true` の profile ごとに、モデル読込、文字起こし、モデル解放が順次行われる
4. 同じ録音音声に対する各 profile の結果が集計される
5. 結果が保存され、設定時は Markdown がクリップボードへコピーされる

### 4. 出力先だけ CLI で変更

現時点で CLI 上書きに対応しているのは出力先です。モデルやデバイスの切替は `config.toml` で行います。

```powershell
python record_and_compare.py --output-dir my_results
```

## config.toml

### 録音設定

```toml
[audio]
sample_rate = 16000
channels = 1
dtype = "float32"
```

初期版はこの組み合わせのみ対応しています。他の値は未対応設定として実行前に停止します。

### 出力設定

```toml
[output]
output_dir = "results"
copy_markdown_to_clipboard = true
save_markdown = true
save_csv = true
save_json = true
```

| 設定                         | 内容                                      |
| ---------------------------- | ----------------------------------------- |
| `output_dir`                 | 保存先ディレクトリ                        |
| `copy_markdown_to_clipboard` | Markdown 比較結果をクリップボードへコピー |
| `save_markdown`              | Markdown ファイルを保存                   |
| `save_csv`                   | CSV ファイルを保存                        |
| `save_json`                  | 完全記録用 JSON を保存                    |

クリップボードだけで使う例:

```toml
[output]
output_dir = "results"
copy_markdown_to_clipboard = true
save_markdown = false
save_csv = false
save_json = false
```

すべての出力を `false` にすると、結果を利用できないため実行前にエラー終了します。

### Profile 設定

```toml
[profiles.small]
enabled = true
model_id = "small"
backend = "faster_whisper"
language = "ja"
device = "cpu"
compute_type = "int8"
cpu_threads = 4
num_workers = 1
```

| 設定           | 内容                                          |
| -------------- | --------------------------------------------- |
| セクション名   | 結果で使用する `profile_id`。例: `small`      |
| `enabled`      | この比較対象を実行するか                      |
| `model_id`     | `faster-whisper` に渡すモデル識別子           |
| `backend`      | 実行 backend。現在は `faster_whisper` のみ    |
| `language`     | 文字起こし言語。日本語は `ja`                 |
| `device`       | `cpu` または `cuda`                           |
| `compute_type` | CTranslate2 の計算精度。CPU 初期設定は `int8` |
| `cpu_threads`  | CPU 実行時のスレッド数                        |
| `num_workers`  | `WhisperModel` の worker 数                   |

GPU 設定例:

```toml
[profiles.large_v3]
enabled = true
model_id = "large-v3"
backend = "faster_whisper"
language = "ja"
device = "cuda"
compute_type = "float16"
cpu_threads = 4
num_workers = 1
```

GPU 実行には、使用する `faster-whisper` / CTranslate2 のバージョンに対応した NVIDIA CUDA / cuDNN 環境が必要です。現在の依存構成では CUDA 12.x の cuBLAS と CUDA 12 対応の cuDNN 9 を使用します。詳細は「CUDA 実行環境」を参照してください。

## 出力

`results/` へ、同一タイムスタンプを持つファイルを生成します。

| 形式    | 用途                             |
| ------- | -------------------------------- |
| `.md`   | 人が読んで共有しやすい比較結果   |
| `.csv`  | 結果一覧を表形式で扱うための出力 |
| `.json` | 設定と結果を含む完全記録         |

JSON は `schema_version = 2` として、次の識別情報を各結果に含みます。

```text
profile_id
model_id
backend
settings
transcript_raw
model_load_seconds
inference_seconds
inference_rtf
measurement_scope
```

また、`effective_configuration.profiles` に実行時の profile 設定を記録し、`execution_order` に profile の実行順を記録します。

## 指標

| 用語                    | 説明                                                |
| ----------------------- | --------------------------------------------------- |
| `RTF`                   | 推論時間 / 音声長。`1` 未満なら音声の実時間より高速 |
| `model_load_seconds`    | モデル準備時間。CUDA profile では warm-up を含む。RTF には含めない |
| `preprocessing_seconds` | engine 投入前の音声前処理時間                       |
| `inference_seconds`     | 文字起こし結果の回収完了までの推論時間              |
| `execution_seconds`     | 前処理と推論を含む adapter 実行時間                 |
| `measurement_scope`     | 推論時間がどの範囲を表すか                          |

現在の `faster-whisper` profile は、モデル準備後の推論区間を `inference_only` として記録します。

CUDA profile では、GPU の初回初期化負担が先頭 profile の速度結果へ混入しないよう、モデル準備時に短い無音入力で warm-up 推論を 1 回行います。warm-up は `model_load_seconds` に含まれ、`inference_seconds` と RTF には含まれません。

## 他のモデル比較を追加する

### faster-whisper で読めるモデルを追加する場合

`faster-whisper` / CTranslate2 形式で利用できるモデルは、`config.toml` に profile を追加して比較候補にできます。

例:

```toml
[profiles.my_faster_whisper_model]
enabled = true
model_id = "モデル識別子またはローカルモデルパス"
backend = "faster_whisper"
language = "ja"
device = "cpu"
compute_type = "int8"
cpu_threads = 4
num_workers = 1
```

確認事項:

- モデルが `faster-whisper` からロード可能か
- 日本語文字起こしに対応しているか
- 使用端末の RAM / VRAM で、追加モデル単体を読み込んで推論できるか
- profile を増やした際のモデル読込を含む総実行時間が許容できるか
- モデル配布元のライセンスと再配布条件

### 別 backend を追加する場合

`whisper.cpp`、ReazonSpeech / sherpa-onnx、Vosk、NeMo 等は、`config.toml` の追記だけでは動きません。新しい実行 backend として adapter 追加が必要です。

追加時の基本境界:

1. `adapters/base.py` の `BaseAdapter` に従い、モデル準備と推論処理を分離する
2. `SourceAudio` の同一録音原本を入力起点にする
3. 必要な WAV 化やリサンプリングは adapter 内で行い、派生入力条件と前処理時間を記録する
4. `EngineResult` に `profile_id` / `model_id` / `backend` / 計測値 / transcript を返す
5. backend ごとの設定読込と adapter 選択を追加する
6. 追加依存、モデル入手手順、利用条件を README に明記する

速度比較で特に注意する点:

- Python 内で推論区間を計測できる backend と、CLI subprocess を呼び出す backend では計測範囲が異なりうる
- `measurement_scope` が異なる結果を、同じ推論 RTF として単純順位付けしない
- リサンプリングや WAV 化が必要な backend は、変換条件と `preprocessing_seconds` を残す

追加候補の位置付け:

| 候補                             | 想定する役割                           |
| -------------------------------- | -------------------------------------- |
| ReazonSpeech-k2-v2 / sherpa-onnx | Whisper 系以外の日本語 ASR 比較        |
| whisper.cpp                      | 同一 Whisper モデルの軽量 backend 比較 |
| Vosk                             | 軽量・組み込み寄りの比較               |
| NVIDIA NeMo / FastConformer 系   | GPU 高性能 ASR の比較                  |

## 現在の制限

- backend は `faster_whisper` のみ
- 有効 profile は録音後に順次処理されるため、profile 数に応じて総実行時間が増える
- モデルは処理後に解放するが、RAM / VRAM の実際の返却タイミングは推論 runtime と実行環境に依存する
- 初期版は 16 kHz / mono / float32 の録音入力のみ
- 精度評価用の正解テキストや CER / WER の集計は未対応
- 実行速度は CPU / GPU、メモリ、計算精度、スレッド数等に大きく依存する

## License

本リポジトリで作成したソースコード、設定テンプレート、ドキュメントは [MIT License](LICENSE) で公開しています。

本ツールが利用する外部ライブラリおよび ASR モデルの重みファイルは、本リポジトリには同梱しません。利用者が各配布元から取得し、それぞれのライセンスおよび利用条件に従って使用してください。

現在利用する主な外部成果物:

| 対象 | 用途 | ライセンス確認先 |
| --- | --- | --- |
| `faster-whisper` | ASR 推論 backend | [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/LICENSE) |
| `CTranslate2` | 推論 runtime | [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) |
| Whisper `small` / `large-v3` | 比較用モデル | [openai/whisper](https://github.com/openai/whisper/blob/main/LICENSE) |
| `kotoba-tech/kotoba-whisper-v2.0-faster` | 日本語特化比較用モデル | [Hugging Face model page](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-faster) |

将来、モデルファイル、依存ライブラリ、DLL、実行ファイル一式などを本リポジトリや配布物へ同梱する場合は、対象物ごとの再配布条件およびライセンス表示要件を改めて確認してください。
