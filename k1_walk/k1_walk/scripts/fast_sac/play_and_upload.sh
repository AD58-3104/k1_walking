#!/bin/bash
# play_and_upload.sh
#
# 推論を実行してビデオを録画し、Google Drive にアップロードする
#
# 使い方:
#   ./play_and_upload.sh <checkpoint_path> [オプション]
#
# 必須:
#   <checkpoint_path>: チェックポイントファイルのパス
#
# オプション:
#   --video_length N       : 録画ステップ数 (デフォルト: 500)
#   --drive_remote NAME    : rclone のリモート名 (デフォルト: gdrive)
#   --drive_folder PATH    : Drive 上の保存先フォルダ (デフォルト: k1_walk_videos)
#   --num_envs N           : 環境数 (デフォルト: 1)
#   --task TASK            : タスク名 (デフォルト: K1-Walk-Play-fast-sac)
#   --no_upload            : アップロードせずローカル保存のみ
#
# 事前設定:
#   rclone が未設定の場合は以下を実行してください:
#     rclone config
#   → Google Drive を選んでリモート名を "gdrive" に設定
#
# 例:
#   ./play_and_upload.sh logs/fast_sac/k1_walk/model_1000.pt
#   ./play_and_upload.sh logs/fast_sac/k1_walk/model_1000.pt --video_length 1000 --drive_folder k1_walk/runs

set -e

# ------------------------------------------------------------------ #
# デフォルト値
# ------------------------------------------------------------------ #
CHECKPOINT=""
VIDEO_LENGTH=500
DRIVE_REMOTE="isaac_video"
DRIVE_FOLDER="k1_walk_videos"
NUM_ENVS=1
TASK="K1-Walk-Play-fast-sac"
NO_UPLOAD=false

# ------------------------------------------------------------------ #
# 引数パース
# ------------------------------------------------------------------ #
if [ $# -eq 0 ]; then
    echo "Usage: $0 <checkpoint_path> [options]"
    echo "  --video_length N    録画ステップ数 (デフォルト: 500)"
    echo "  --drive_remote NAME rclone リモート名 (デフォルト: gdrive)"
    echo "  --drive_folder PATH Drive上の保存先 (デフォルト: k1_walk_videos)"
    echo "  --num_envs N        環境数 (デフォルト: 1)"
    echo "  --task TASK         タスク名"
    echo "  --no_upload         ローカル保存のみ（アップロードしない）"
    exit 1
fi

CHECKPOINT="$1"
shift

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video_length)  VIDEO_LENGTH="$2";  shift 2 ;;
        --drive_remote)  DRIVE_REMOTE="$2";  shift 2 ;;
        --drive_folder)  DRIVE_FOLDER="$2";  shift 2 ;;
        --num_envs)      NUM_ENVS="$2";      shift 2 ;;
        --task)          TASK="$2";          shift 2 ;;
        --no_upload)     NO_UPLOAD=true;     shift   ;;
        *) echo "[ERROR] 不明なオプション: $1"; exit 1 ;;
    esac
done

# ------------------------------------------------------------------ #
# 入力チェック
# ------------------------------------------------------------------ #
if [ -z "$CHECKPOINT" ]; then
    echo "[ERROR] チェックポイントパスを指定してください"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "[ERROR] チェックポイントが見つかりません: $CHECKPOINT"
    exit 1
fi

if [ "$NO_UPLOAD" = false ] && ! command -v rclone &> /dev/null; then
    echo "[ERROR] rclone がインストールされていません"
    echo "  インストール: curl https://rclone.org/install.sh | sudo bash"
    echo "  設定:         rclone config"
    echo "  または --no_upload オプションでアップロードをスキップできます"
    exit 1
fi

# ------------------------------------------------------------------ #
# 設定表示
# ------------------------------------------------------------------ #
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "============================================"
echo " K1 Walk - Play & Upload"
echo "============================================"
echo "  checkpoint   : $CHECKPOINT"
echo "  task         : $TASK"
echo "  num_envs     : $NUM_ENVS"
echo "  video_length : $VIDEO_LENGTH ステップ"
echo "  timestamp    : $TIMESTAMP"
if [ "$NO_UPLOAD" = false ]; then
    echo "  upload先     : ${DRIVE_REMOTE}:${DRIVE_FOLDER}"
fi
echo "============================================"

# ------------------------------------------------------------------ #
# スクリプトのディレクトリへ移動（play.py の相対 import のため）
# ------------------------------------------------------------------ #
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ------------------------------------------------------------------ #
# 推論 + 録画実行
# ------------------------------------------------------------------ #
echo ""
echo "[1/3] 推論・録画を開始します..."

source ~/.bash_functions
# --video使用時はDISPLAYをunsetしてNVIDIA EGLレンダリングを使用
unset DISPLAY
_labpython play.py \
    --task "$TASK" \
    --num_envs "$NUM_ENVS" \
    --checkpoint "$CHECKPOINT" \
    --video \
    --video_length "$VIDEO_LENGTH" \
    --headless

echo "[1/3] 録画完了"

# ------------------------------------------------------------------ #
# 生成されたビデオファイルを特定
# ------------------------------------------------------------------ #
echo ""
echo "[2/3] ビデオファイルを検索中..."

# play.py は dirname(checkpoint)/videos/play/ に保存するので、そこを直接検索
# (CHECKPOINT が相対パスの場合は SCRIPT_DIR 基準で解決)
CHECKPOINT_ABS="$(cd "$(dirname "$CHECKPOINT")" && pwd)/$(basename "$CHECKPOINT")"
VIDEO_DIR="$(dirname "$CHECKPOINT_ABS")/videos/play"
VIDEO_FILE=$(find "$VIDEO_DIR" -name "*.mp4" 2>/dev/null | sort | tail -n 1)

if [ -z "$VIDEO_FILE" ]; then
    # フォールバック: SCRIPT_DIR 配下の logs を再帰的に検索（最近10分以内）
    VIDEO_FILE=$(find "$SCRIPT_DIR/logs" -name "*.mp4" -mmin -10 2>/dev/null | sort | tail -n 1)
fi

if [ -z "$VIDEO_FILE" ]; then
    echo "[ERROR] ビデオファイルが見つかりませんでした"
    echo "  検索パス: $VIDEO_DIR"
    exit 1
fi

echo "  ビデオファイル: $VIDEO_FILE"

# タイムスタンプ付きのファイル名でコピー
CHECKPOINT_BASENAME=$(basename "$CHECKPOINT" .pt)
UPLOAD_FILENAME="${TIMESTAMP}_${CHECKPOINT_BASENAME}.mp4"
UPLOAD_LOCAL_PATH="$(dirname "$VIDEO_FILE")/$UPLOAD_FILENAME"
cp "$VIDEO_FILE" "$UPLOAD_LOCAL_PATH"
echo "  保存先: $UPLOAD_LOCAL_PATH"

# ------------------------------------------------------------------ #
# Google Drive アップロード
# ------------------------------------------------------------------ #
if [ "$NO_UPLOAD" = false ]; then
    echo ""
    echo "[3/3] Google Drive にアップロード中..."
    echo "  リモート: ${DRIVE_REMOTE}:${DRIVE_FOLDER}/"

    rclone copy "$UPLOAD_LOCAL_PATH" "${DRIVE_REMOTE}:${DRIVE_FOLDER}/" \
        --progress \
        --stats-one-line

    echo ""
    echo "[3/3] アップロード完了"
    echo "  Drive パス: ${DRIVE_FOLDER}/$UPLOAD_FILENAME"

    # 共有リンクを取得（rclone がサポートしている場合）
    LINK=$(rclone link "${DRIVE_REMOTE}:${DRIVE_FOLDER}/$UPLOAD_FILENAME" 2>/dev/null || echo "")
    if [ -n "$LINK" ]; then
        echo "  共有リンク: $LINK"
    fi
else
    echo ""
    echo "[3/3] --no_upload 指定のためアップロードをスキップ"
fi

# ------------------------------------------------------------------ #
# 完了
# ------------------------------------------------------------------ #
echo ""
echo "============================================"
echo " 完了"
echo "  ローカル: $UPLOAD_LOCAL_PATH"
if [ "$NO_UPLOAD" = false ]; then
    echo "  Drive:    ${DRIVE_FOLDER}/$UPLOAD_FILENAME"
fi
echo "============================================"
