# FlashVSR+ Turbo：アップスケール専用Pod一括導入

対象テンプレート：`runpod/comfyui:2.4.1-comfyuiv0.44.0-cuda13.2`。
**動画生成モデルは含みません。** ServerlessやAPIキーも使いません。
アプリと同じ公式FlashVSR+ Turboを、Pod内のComfyUIから実行する構成です。

## 前提

- 新しいGPU Pod（24GB以上のNVIDIA GPU推奨。最小検査は24GB級）。
- コンテナディスク80GB以上推奨。初回35GiB以上の空きが必要。
- テンプレートの初回起動を完了させ、ComfyUI・Jupyter・FileBrowserが使える状態。
- `COMFYUI_PASSWORD`などはテンプレート側で通常どおり設定。
- Jupyterの**Terminal**でrootとして実行。ノートブックのPythonセルではありません。

`install.sh`をダウンロードし、`bash install.sh`で実行します。
インストーラは次を行います。

1. `/opt/flashvsr-pod`のコンテナディスクに専用venvと公式ソースを配置。
2. モデル約7.02GBを公開配布元から取得し、サイズ・SHA256を検証。
3. 既存ComfyUIに操作用ノードとワークフローだけを追加。
4. ComfyUIがアイドルであることを確認して再起動。実行中なら停止しません。

テンプレートの既存ComfyUIコード・既存venvのパッケージは更新しません。
推論は**別venvのPyTorch2.7.1 CUDA12.8 + Triton3.3.1**で実行するため、
テンプレートのCUDA13.2用pip制約・PyTorchと混ざりません。
CコンパイラとPythonヘッダを使った共有ライブラリのコンパイル/ロード検査も行います。
テンプレートの公開8188認証プロキシを残し、内部18188のComfyUIだけを再起動します。
JupyterとFileBrowserは停止しません。Pod全体の再起動や起動スクリプト変更はしません。

## 操作

ComfyUIを再読み込みし、`FlashVSR-Pod-Turbo.json`を開きます。

- **元動画を読み込む**：動画をアップロード。
- **FlashVSR+ Turbo（動画拡大・保存）**：1.5倍/2倍/4倍、動きの保持、シードを選択。
- Queue：処理後のMP4を同じノードでプレビュー・保存。

最初は同梱の中立的なテストパターン動画（128x128・5フレーム）と2倍で確認します。
このサンプルはFFmpegだけで配置し、インストール時にGPU推論は実行しません。
出力は既存ComfyUIの`output/FlashVSR-Pod/`にMP4と処理情報JSONを保存します。

**初期対応範囲**：30秒まで、固定fps1～60、1800フレームまで、SDR8bit、正方画素。
入力各辺64～1920px・面積1920x1080以下、入力ファイル512MiBまで。
ComfyUIのアップロード上限も512MiBに設定しますが、外側のプロキシ制限がある場合は
Jupyter/FileBrowser経由でComfyUIの`input`へ配置してください。
出力各辺2048pxまで。128pxグリッドへエッジを補って処理後に補った部分だけ除去し、
元の構図・縦横比を保ちます。fps・フレーム数を確認し、音声はAACへ変換して引き継ぎます。

例：1280x720は1.5倍で1920x1080。800x800は2倍で1600x1600。4倍は出力上限を超えます。
4倍は公式モデルの推奨倍率。1.5倍・2倍は実用サイズで比較するための選択肢です。
動きの保持：標準=temporal compression4、他は2/1で、処理時間は増加します。
細部の質感や形を再構成する場合があるので、必ず元動画と比較してください。

Pod内でファイルを直接処理するため、アプリ版の64MiB転送制限・24時間リンクはありません。
ただしディスク空き容量は必要です。最大サイズの全動画が24GB GPUで処理できる保証はなく、
長さ・解像度・動き設定によりVRAM不足や30分の子プロセス時間切れになる場合があります。
各ジョブでモデルを読み込み、終了時にGPUプロセスと作業用ファイルを片付けます。
ComfyUIのキャンセル操作時にも推論子プロセスグループを停止します。

## 再起動・再利用

```bash
# ComfyUIだけ再起動
bash /opt/flashvsr-pod/restart-comfyui.sh

# ComfyUI起動ログ
 tail -n 100 /opt/flashvsr-pod/comfyui.log

# 導入のみ（再起動を別途行いたい場合）
FLASHVSR_NO_RESTART=1 bash install.sh
```

モデルと推論環境は`/opt/flashvsr-pod`です。PodをTerminateすると失われます。
ComfyUIへのアップロードと完成MP4はテンプレート標準の`/workspace/runpod-slim/ComfyUI`
側で保存され、その保持はPodのボリューム構成に従います。

Pod起動時の自動起動設定は変更しません。再起動後はテンプレートの通常ComfyUIが
追加ノードを読み込みます。専用モデルを使う時だけFlashVSR子プロセスを起動します。

## 固定バージョン / 配布元

- 公式ソース：OpenImagingLab/FlashVSR_Plus `b515ffc6dba57fdc9ad84c66cadd87d05f27590a`
- モデル：ruchangyao/FlashVSR_plus `46b79520946ac67dec94d1566490690953c7debc`
- Turbo DiT、LQ projection、TCDecoderのみ。モデル合計7,016,472,997bytes。
- 公式Triton block-sparse attentionを明示使用。利用できなければエラーにし、
  遅いdense attentionへ黙って切り替えません。
- CPU x264エンコード、CRF18。勝手な低ビットレート化やフレーム間引きはしません。

https://github.com/OpenImagingLab/FlashVSR_Plus
https://huggingface.co/ruchangyao/FlashVSR_plus

## 検証範囲

アプリ/Serverless版の同一モデルはA5000で実推論成功済みです。
この**Pod用ComfyUIアダプタ**は、シェル・埋め込みPython構文、ワークフロー接続、
実FFmpegによる寸法・fps・フレーム数・音声保持、原動画の保持、キャンセル経路、
認証プロキシを維持するポート選択など7件をCPUで確認しています。
新しいPod上の全体インストール・ComfyUI画面・GPU推論は未検証です。
インストール完了表示はノードの認識確認で、推論成功を意味しません。

```bash
python -m unittest -v test_pod
```

## FPS保存修正

公式ストリーミング出力のエンコード段階で小数FPSが丸められるケースに対応し、
最終保存時に元の正確な有理数FPS（例24000/1001）とフレーム番号からタイムスタンプ
を再設定します。フレーム数・音声・寸法の検査は継続します。GPU推論部分は同じです。
修正はengine.pyだけなので、既存Podでモデル再取得やComfyUI再起動は不要です。

## 1.5倍の追加

倍率に1.5倍を追加しました。1280x720→1920x1080は現在の2048px上限内です。
小数倍率はH.264/yuv420p用に最も近い偶数寸法へ丸めます（誤差最大1px）。
モデル内部の128pxグリッド補完と処理後の除去は従来どおりです。
小さい倍率でも画質は要確認です。1.5倍のGPU推論はまだ未検証です。
既存Podはupdate-scale-1_5.shでコードだけを更新し、ComfyUIを再起動します。
モデル・Pythonパッケージは再取得しません。FPS修正も更新に含みます。
