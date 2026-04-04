#!/usr/bin/env python3
"""
次のpython形式で書かれた設定に従って、BASE_COMMAND + COMMON_ARGS + ADDITIONAL_ARGS + arg_list[i]のコマンドを順番に実行する。

```python
BASE_COMMAND = "bash play.sh"
COMMON_ARGS = " --task k1-walk"
ADDITIONAL_ARGS = ""   # コマンドラインからの追加引数を受け取る

arg_list =[
    [""],
    [""],
    [""],
    [""],
]
```

引数は以下のものを受け取る。
- 読み込む対象の設定ファイルのファイル名
- 追加共通コマンド

引数がおかしいなどで、コマンドの実行が失敗した場合でもその場でスクリプトを停止させず、最後の実験まで実行する。
ログを保存するディレクトリの名前は、「設定ファイルのファイル名_exp_実験番号」とする。
"""

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


def load_config(config_file: str) -> dict:
    """設定ファイルを読み込む"""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_file}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    return {
        "BASE_COMMAND": getattr(config_module, "BASE_COMMAND", ""),
        "COMMON_ARGS": getattr(config_module, "COMMON_ARGS", ""),
        "ADDITIONAL_ARGS": getattr(config_module, "ADDITIONAL_ARGS", ""),
        "arg_list": getattr(config_module, "arg_list", []),
        "LOG_DIR_ARG": getattr(config_module, "LOG_DIR_ARG", "--experiment_name"),
    }


def run_experiments(config_file: str, additional_args: str = ""):
    """設定ファイルに従って実験を順番に実行する"""
    config = load_config(config_file)

    base_command = config["BASE_COMMAND"]
    common_args = config["COMMON_ARGS"]
    config_additional_args = config["ADDITIONAL_ARGS"]
    arg_list = config["arg_list"]
    log_dir_arg = config["LOG_DIR_ARG"]

    # コマンドラインからの追加引数を結合
    all_additional_args = config_additional_args
    if additional_args:
        all_additional_args = f"{all_additional_args} {additional_args}".strip()

    # 設定ファイル名（拡張子なし）を取得
    config_name = Path(config_file).stem

    results = []

    for i, args in enumerate(arg_list):
        exp_num = i + 1
        log_dir_name = f"{config_name}_exp_{exp_num}"

        # arg_listの要素がリストの場合は結合
        if isinstance(args, list):
            exp_args = " ".join(arg for arg in args if arg)
        else:
            exp_args = str(args).strip()

        # コマンドを構築（各部分をリストに追加し、空でないものだけ結合）
        # ログディレクトリを指定
        log_dir_override = f"{log_dir_arg} {log_dir_name}" if log_dir_arg else ""

        command_parts = [
            base_command.strip(),
            common_args.strip(),
            all_additional_args.strip(),
            exp_args,
            log_dir_override,
        ]
        command = " ".join(part for part in command_parts if part)

        print(f"\n{'='*60}")
        print(f"実験 {exp_num}/{len(arg_list)}: {log_dir_name}")
        print(f"コマンド: {command}")
        print(f"{'='*60}\n")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=Path(__file__).parent,
            )
            success = result.returncode == 0
            results.append({
                "exp_num": exp_num,
                "log_dir": log_dir_name,
                "command": command,
                "success": success,
                "returncode": result.returncode,
            })
            if success:
                print(f"\n✓ 実験 {exp_num} 完了")
            else:
                print(f"\n✗ 実験 {exp_num} 失敗 (終了コード: {result.returncode})")
        except Exception as e:
            print(f"\n✗ 実験 {exp_num} エラー: {e}")
            results.append({
                "exp_num": exp_num,
                "log_dir": log_dir_name,
                "command": command,
                "success": False,
                "error": str(e),
            })

    # 結果サマリーを表示
    print(f"\n{'='*60}")
    print("実験結果サマリー")
    print(f"{'='*60}")

    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    for r in results:
        status = "✓ 成功" if r["success"] else "✗ 失敗"
        print(f"  実験 {r['exp_num']}: {status} - {r['log_dir']}")

    print(f"\n合計: {success_count} 成功, {fail_count} 失敗")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="設定ファイルに従ってコマンドを順番に実行するランナースクリプト"
    )
    parser.add_argument(
        "config_file",
        help="読み込む対象の設定ファイルのファイル名",
    )
    parser.add_argument(
        "--additional-args",
        "-a",
        default="",
        help="追加共通コマンド",
    )

    args = parser.parse_args()

    try:
        results = run_experiments(args.config_file, args.additional_args)
        # 1つでも失敗があれば終了コード1
        if any(not r["success"] for r in results):
            sys.exit(1)
    except FileNotFoundError as e:
        print(f"エラー: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"予期しないエラー: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
