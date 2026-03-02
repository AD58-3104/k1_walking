"""
ランナースクリプト用の設定ファイル例
"""

BASE_COMMAND = "bash play.sh"
COMMON_ARGS = " --task k1-walk"
ADDITIONAL_ARGS = ""  # コマンドラインからの追加引数を受け取る

# ログディレクトリを指定するための引数名
# デフォルト: "--experiment_name"
# 空文字にするとログディレクトリの自動指定を無効化
LOG_DIR_ARG = "--experiment_name"

arg_list = [
    [""],
    [""],
    [""],
    [""],
]
