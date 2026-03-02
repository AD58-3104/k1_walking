以下の要件を満たすランナースクリプトをpythonで書いて下さい。

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