#!/usr/bin/env python3
"""
auto_loop.py
Geminiが動画とTensorBoardの数値を見て報酬重みを自動調整するループスクリプト
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types
from tensorboard.backend.event_processing import event_accumulator

# ────────────────────────────────────────────────────────────
# 引数解析 & 設定
# ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="前回の履歴を無視して新規セッションを開始")
    parser.add_argument("--fps", type=float, default=1.0, help="動画のサンプリングFPS (デフォルト: 1.0)")
    return parser.parse_args()

args = parse_args()

TRAIN_SCRIPT = "/workspace/k1_walking/k1_walk/k1_walk/scripts/fast_sac/train.py"
PLAY_SCRIPT  = "/workspace/k1_walking/k1_walk/k1_walk/scripts/fast_sac/play.py"
ISAACLAB_SH  = "/workspace/isaaclab/isaaclab.sh"

TASK           = "K1-Walk-Train-fast-sac"
NUM_ENVS_TRAIN = 2048
NUM_ENVS_PLAY  = 8
MAX_ITERATIONS = 10000

# ── ログディレクトリ構成 ──
BASE_LOG_ROOT = Path("logs/auto_loop")

def get_session_dir(reset_flag):
    if not reset_flag and BASE_LOG_ROOT.exists():
        sessions = sorted([d for d in BASE_LOG_ROOT.iterdir() if d.is_dir()])
        if sessions:
            print(f"[INFO] 既存セッションを継続します: {sessions[-1].name}")
            return sessions[-1]
    
    new_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = BASE_LOG_ROOT / new_name
    new_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 新規セッションを開始します: {new_name}")
    return new_dir

LOG_ROOT       = get_session_dir(args.reset)
VIDEO_DIR      = LOG_ROOT / "videos"
HISTORY_FILE   = LOG_ROOT / "history.json"
TRAIN_LOG_BASE = Path(__file__).parent / "logs" / "fast_sac"

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"

# subprocess に渡す環境変数
SUBPROCESS_ENV = {**os.environ, "TORCHDYNAMO_DISABLE": "1"}

# 重みの初期値
INITIAL_WEIGHTS = {
    "env.rewards.track_lin_vel_xy_exp.weight":           2.5,
    "env.rewards.track_ang_vel_z_exp.weight":            2.0,
    "env.rewards.feet_height_bezier.weight":            12.0,
    "env.rewards.feet_air_time.weight":                  2.5,
    "env.rewards.stride_length.weight":                  1.5,
    "env.rewards.bad_gait_penalty.weight":              -0.4,
    "env.rewards.orientation_potential.weight":           1.0,
    "env.rewards.joint_regularization_potential.weight":  2e-4,
    "env.rewards.feet_parallel_to_ground.weight":         3.5,
    "env.rewards.feet_slide.weight":                    -0.12,
    "env.rewards.alive_bonus.weight":                     0.7,
    "env.rewards.knee_limit_lower.weight":               -1.0,
    "env.rewards.dof_pos_limits.weight":                -10.0,
    "env.rewards.torque_limits.weight":                 -0.001,
    "env.rewards.base_jerk.weight":                     -0.005,
    "env.rewards.ang_vel_xy_l2.weight":                 -0.01,
    "env.rewards.joint_jerk.weight":                    -1e-7,
    "env.rewards.lin_vel_z_pen.weight":                  -0.25,
    "env.rewards.action_rate_l2.weight":                -0.1,
    "env.curriculum.action_rate_cur.params.target_weight": -1.2,
    "env.curriculum.base_jerk_cur.params.target_weight":   -0.03,
    "env.curriculum.bad_gait_cur.params.target_weight":    -1.0,
}

GEMINI_PROMPT = """\
<!---　まずここの文章いいのか？ --->
あなたはヒューマノイドロボットの強化学習の専門家です。
IsaacLabで歩行動作の強化学習を行ったBooster K1の動画と、学習結果の統計数値を見て、それぞれの報酬関数の重みを詳しく検討し、報酬関数の重みを改善してください。
目標は、**「指定された目標速度（緑矢印）に現在速度（青矢印）を正確に追従させつつ、転倒せず、足の引きずりや不自然な振動（Jerk）のない、滑らかで安定した二足歩行」**を実現することです。
動画は10000回の学習を終了したときのものです。
ロボット頭上にある矢印は，緑が目標速度で青が現在速度です。

## 各報酬の仕組み
### それぞれの報酬の仕組み
* **track_lin_vel_xy_exp**
    * 効果：重力方向に整列されたロボット座標系において，指定されたXY方向の移動速度を正確に維持させます。
    * 計算：目標速度と現在速度（Yaw角補正済み）の差の2乗を指数関数（$exp$）にかけます。誤差が小さいほど1に近づき，大きいと急激に0になります。

* **track_ang_vel_z_exp**
    * 効果：世界座標系において，指定された旋回速度（Yaw角速度）を正確に維持させます。
    * 計算：目標旋回速度と現在のルート角速度の差の2乗を用いた指数関数評価です。

* **feet_height_bezier**
    * 効果：遊脚がベジエ曲線に沿った理想的な高さ軌道を通るように促し，スムーズなスイング動作を実現します。
    * 計算：歩行フェーズから算出された目標高さ（ベジエ曲線）と実際の足の高さの誤差を指数関数で評価します。

* **feet_air_time**
    * 効果：足をしっかりと上げ，適切な滞空時間を確保した歩行を促します。
    * 計算：着地した瞬間に，それまでの滞空時間がしきい値を超えていた場合にその長さに応じた報酬を与えます。

* **stride_length_reward**（JSON内の `stride_length` に対応）
    * 効果：コマンド速度に応じた適切な歩幅（ストライド長）での歩行を促します。
    * 計算：着地瞬間の位置から歩幅を算出し，目標歩幅との誤差を評価します。ホッピング時は報酬をゼロにします。

* **bad_gait_penalty**
    * 効果：同じ足での連続接地（ホッピング）や，極端に短い遊脚時間による不自然な歩容を抑制します。
    * 計算：同一脚の連続着地判定，および着地時の滞空時間が規定値（min_air_time）以下である場合にペナルティを与えます。

* **orientation_potential**
    * 効果：ロボットの胴体（ルート）を常に垂直（直立）に近い状態に保たせます。
    * 計算：胴体のZ軸が世界座標の垂直方向からどれだけズレているかを評価するポテンシャル関数の，ステップ間増分（Shaped Reward）を計算します。

* **joint_regularization_potential**
    * 効果：各関節が不自然な角度にならないよう，初期姿勢（デフォルト）付近での動作を促します。
    * 計算：関節角度のデフォルトからの偏差をポテンシャル報酬として評価します。脚のピッチ，ロール（対称性），旋回時のヨーなど軸ごとに異なる緩和を適用します。

* **feet_parallel_to_ground**
    * 効果：足裏が常に地面と平行（水平）になるように制御し，安定した接地を促します。
    * 計算：足リンクのピッチ・ロール角の2乗誤差に基づくポテンシャル報酬です。

* **feet_slide**
    * 効果：接地中の足裏が地面を滑るスリップ現象を抑制し，効率的な蹴り出しを促します。
    * 計算：足が地面と接触している間の水平方向速度を検知し，速度がある場合にペナルティを与えます。

* **alive_bonus**
    * 効果：転倒によるリセット（エピソード終了）を避け，可能な限り長い時間歩行を継続させる動機付けを与えます。
    * 計算：ロボットが生存（転倒判定外）している間，ステップ毎に一定の正の報酬を加算します。

* **knee_limit_lower**
    * 効果：膝関節が逆折れ方向や限界角度を超えて動作するのを物理的に保護します。
    * 計算：規定の限界角度を超えた量に基づくペナルティです。

* **dof_pos_limits**
    * 効果：各関節が可動範囲の限界（ソフトリミット）に到達するのを防ぎ，スムーズな動作範囲を維持させます。
    * 計算：関節角度が規定の動作範囲を超えた際の超過量に基づくペナルティです。

* **torque_limits**
    * 効果：モーターの出力トルクが飽和するのを防ぎ，実機のハードウェアへの負荷を軽減します。
    * 計算：各関節に印加されたトルクが規定の制限値を超えた際の超過量に基づくペナルティです。

* **base_jerk**
    * 効果：胴体の加速度の急激な変化（カクつき）を抑え，滑らかな移動を実現します。
    * 計算：胴体線形加速度の微分値（加加速度）のノルムに対するペナルティです。

* **ang_vel_xy_l2**
    * 効果：胴体のロール方向およびピッチ方向の不要な揺れを抑制し，歩行中の上体の安定性を向上させます。
    * 計算：胴体のX軸およびY軸まわりの角速度の2乗和（L2ノルム）に対するペナルティです。

* **joint_jerk**
    * 効果：関節の動きの急変を抑え，モーターや減速機への負荷を軽減します。
    * 計算：関節加速度の差分から算出した加加速度のノルムに対するペナルティです。

* **lin_vel_z_pen**
    * 効果：胴体が上下に激しく揺れるのを抑制し，エネルギー効率と安定性を向上させます。
    * 計算：胴体の垂直方向（Z軸）速度の2乗に対するペナルティです。

* **action_rate_l2**
    * 効果：制御入力（アクション）がステップ間で激しく振動するのを防ぎ，滑らかな制御出力を促します。
    * 計算：前ステップと現ステップのアクション値の差の2乗和を計算し，ペナルティを与えます。

### カリキュラムがどうやってつくか
- env.rewards.* は学習初期から適用される重み（ベースライン）です。
- env.curriculum.*.target_weight は、エピソード長が伸びる（学習が進む）につれて徐々に近づいていく最終的な重みの目標値です。
- 学習初期は達成が難しい厳しいペナルティ（例: action_rateやjerkなど）は、初期のreward重みを0または小さくし、curriculumのtarget_weightで本来の厳しい値を設定することで、徐々に歩行を洗練させることができます。

## 重み調整のガイドライン
- 段階的な変更: 重みの変更は、極端な値のジャンプを避け、原則として現在の値から±10%〜30%程度の微調整にとどめてください（ただし、全く機能していないと判断されるパラメータは大胆に変更しても構いません）。
- ペナルティのバランス: 負の報酬（ペナルティ）が強すぎると、ロボットは「動かないのが一番マシ」と学習してしまいます。alive_bonusやtrack_velなどの正の報酬とのバランスを常に考慮してください。
- また，metricsとrewardsのweight，curriculamのtarget_weightは関係はありますが，必ずしも重みの値にmetricsが収束するわけではないことに注意してください。特に，負の値の報酬（ペナルティ）に関してはより小さい値（絶対値が大きく，マイナス符号がついた値）の重みをつけたほうがmetricsの値は大きく（0に近く）なります。避けたいペナルティに対して，より強い重みをかけるようにしてください。

## 現在の報酬重み
{current_weights}

## 学習統計 (TensorBoard)
{metrics}

## これまでの試行履歴
{history}

## 出力形式
以下のJSON形式のみで返してください。他の文章は不要です。
変更しない項目はそのままの値を返してください。

```json
{{
  "observations": "動画から観察できたロボットの具体的な挙動、問題点、およびTensorBoardの数値からの分析（日本語、300字以上）",
  "reasoning": "上記の観察に基づき、どの重みを、なぜ、どのように変更するかの論理的な理由（日本語、300字以上）",
  "weights": {{
    "env.rewards.track_lin_vel_xy_exp.weight": <float>,
    "env.rewards.track_ang_vel_z_exp.weight": <float>,
    "env.rewards.feet_height_bezier.weight": <float>,
    "env.rewards.feet_air_time.weight": <float>,
    "env.rewards.stride_length.weight": <float>,
    "env.rewards.bad_gait_penalty.weight": <float>,
    "env.rewards.orientation_potential.weight": <float>,
    "env.rewards.joint_regularization_potential.weight": <float>,
    "env.rewards.feet_parallel_to_ground.weight": <float>,
    "env.rewards.feet_slide.weight": <float>,
    "env.rewards.alive_bonus.weight": <float>,
    "env.rewards.knee_limit_lower.weight": <float>,
    "env.rewards.dof_pos_limits.weight": <float>,
    "env.rewards.torque_limits.weight": <float>,
    "env.rewards.base_jerk.weight": <float>,
    "env.rewards.ang_vel_xy_l2.weight": <float>,
    "env.rewards.joint_jerk.weight": <float>,
    "env.rewards.lin_vel_z_pen.weight": <float>,
    "env.rewards.action_rate_l2.weight": <float>,
    "env.curriculum.action_rate_cur.params.target_weight": <float>,
    "env.curriculum.base_jerk_cur.params.target_weight": <float>,
    "env.curriculum.bad_gait_cur.params.target_weight": <float>
  }}
}}
```
"""

# ────────────────────────────────────────────────────────────
# ユーティリティ
# ────────────────────────────────────────────────────────────

def setup_dirs():
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)


def load_history() -> list[dict]:
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE) as f:
            return json.load(f)
    return []


def save_history(history: list[dict]):
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def weights_to_args(weights: dict) -> list[str]:
    return [f"{k}={v}" for k, v in weights.items()]


def history_to_str(history: list[dict]) -> str:
    if not history:
        return "なし（初回）"
    lines = []
    for h in history[-3:]:
        obs = h.get('observations', 'なし')
        rea = h.get('reasoning', 'なし')
        entry = f"- イテレーション{h['iteration']}:\n  [観察]: {obs}\n  [考察]: {rea}"
        lines.append(entry)
    return "\n".join(lines)


def run_cmd(cmd: str):
    print(f"\n[CMD]\n{cmd}\n")
    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        env=SUBPROCESS_ENV,
        cwd=Path(__file__).parent,
    )
    if result.returncode != 0:
        raise RuntimeError(f"コマンドが失敗しました (code={result.returncode})")


def get_tensorboard_metrics(log_dir: Path) -> str:
    event_files = list(log_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return "TensorBoardログファイルが見つかりませんでした。"

    latest_event_file = sorted(event_files)[-1]
    ea = event_accumulator.EventAccumulator(str(latest_event_file))
    ea.Reload()

    metrics = {}
    tags = ea.Tags().get('scalars', [])
    target_prefixes = ["Train/mean_reward", "Rewards/", "Policy/"]

    for tag in tags:
        if any(tag.startswith(p) for p in target_prefixes):
            events = ea.Scalars(tag)
            if events:
                metrics[tag] = f"{events[-1].value:.4f}"

    if not metrics:
        return "有効なメトリクスがログから抽出できませんでした。"

    return json.dumps(metrics, indent=2, ensure_ascii=False)


def clip_weights(current: dict, suggested: dict) -> dict:
    """
    提案された重みを現在の値の0.5倍〜2.0倍の範囲にクリッピングする。
    """
    clipped = {}
    for key, suggested_val in suggested.items():
        if key in current:
            curr_val = current[key]
            
            # 元の値が0の場合はマージンを持たせて更新を許可
            if curr_val == 0:
                clipped[key] = suggested_val
                continue

            bound1 = curr_val * 0.5
            bound2 = curr_val * 2.0
            lower = min(bound1, bound2)
            upper = max(bound1, bound2)
            
            clipped_val = max(lower, min(suggested_val, upper))
            
            if abs(clipped_val - suggested_val) > 1e-6:
                print(f"  [CLIP] {key}: {suggested_val:.4f} -> {clipped_val:.4f} (range: {lower:.4f} ~ {upper:.4f})")
            
            clipped[key] = clipped_val
        else:
            clipped[key] = suggested_val
    return clipped


# ────────────────────────────────────────────────────────────
# 各ステップ
# ────────────────────────────────────────────────────────────

def find_latest_checkpoint(log_dir: Path) -> Path:
    candidates = sorted(log_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"チェックポイントが見つかりません: {log_dir}")
    return candidates[-1]


def run_train(iteration: int, weights: dict) -> Path:
    start_time = time.time()
    weight_args = " ".join(weights_to_args(weights))
    cmd = (
        f"{ISAACLAB_SH} -p -m torch.distributed.run --nproc_per_node=2 "
        f"{TRAIN_SCRIPT} --task {TASK} --num_envs {NUM_ENVS_TRAIN} "
        f"--headless --distributed --max_iterations {MAX_ITERATIONS} {weight_args}"
    )
    run_cmd(cmd)

    candidates = [
        d for d in TRAIN_LOG_BASE.rglob("*")
        if d.is_dir() and d.stat().st_mtime >= start_time
        and len(d.name) >= 19 and d.name[:4].isdigit()
    ]
    if not candidates:
        raise FileNotFoundError(f"学習後のログディレクトリが見つかりません")
    log_dir = max(candidates, key=lambda d: d.stat().st_mtime)
    print(f"[TRAIN] ログディレクトリ: {log_dir}")
    return log_dir


def run_play(iteration: int, log_dir: Path) -> Path:
    video_path = VIDEO_DIR / f"iter_{iteration:03d}.mp4"
    checkpoint = find_latest_checkpoint(log_dir)
    cmd = (
        f"{ISAACLAB_SH} -p {PLAY_SCRIPT} --task {TASK} --num_envs {NUM_ENVS_PLAY} "
        f"--headless --video --video_length 3000 --checkpoint {checkpoint}"
    )
    run_cmd(cmd)

    candidates = sorted(log_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"動画が見つかりません: {log_dir}")
    candidates[-1].rename(video_path)
    print(f"[PLAY] 動画を保存: {video_path}")
    return video_path


def ask_gemini(video_path: Path, current_weights: dict, history: list[dict], metrics: str, fps: float) -> dict:
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY が設定されていません")
    client = genai.Client(api_key=api_key)

    print(f"[GEMINI] 動画アップロード中: {video_path}")
    myfile = client.files.upload(file=str(video_path))

    while myfile.state.name == "PROCESSING":
        time.sleep(2)
        myfile = client.files.get(name=myfile.name)
    
    video_part = types.Part.from_uri(file_uri=myfile.uri, mime_type=myfile.mime_type)
    video_part.video_metadata = types.VideoMetadata(fps=fps)

    prompt = GEMINI_PROMPT.format(
        current_weights=json.dumps(current_weights, indent=2),
        history=history_to_str(history),
        metrics=metrics
    )

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[video_part, prompt],
            )
            break
        except Exception as e:
            if attempt == 4: raise
            wait = 300 * (attempt + 1)
            print(f"[GEMINI] エラー: {e}\n{wait}秒後にリトライ...")
            time.sleep(wait)

    text = response.text
    start = text.find("{")
    end = text.rfind("}") + 1
    result = json.loads(text[start:end])
    print(f"[GEMINI] 分析(観察): {result.get('observations', 'N/A')}")
    return result


# ────────────────────────────────────────────────────────────
# メインループ
# ────────────────────────────────────────────────────────────

def main():
    setup_dirs()
    history = load_history()

    if history and not args.reset:
        current_weights = history[-1]["next_weights"]
        start_iteration = history[-1]["iteration"] + 1
        print(f"[INFO] セッション継続: イテレーション {start_iteration} から開始")
    else:
        current_weights = INITIAL_WEIGHTS.copy()
        start_iteration = 0
        print("[INFO] 新規（またはリセット）セッションとして開始")

    iteration = start_iteration
    try:
        while True:
            print(f"\n{'='*60}\nイテレーション {iteration}\n{'='*60}")

            # 1. 学習
            log_dir = run_train(iteration, current_weights)

            # 2. TensorBoardログ解析
            metrics_str = get_tensorboard_metrics(log_dir)

            # 3. 動画生成
            video_path = run_play(iteration, log_dir)

            # 4. Geminiに推論依頼
            result = ask_gemini(video_path, current_weights, history, metrics_str, args.fps)
            suggested_weights = result["weights"]

            # ── クリッピング実行 ──
            print("[INFO] 重みをクリッピング中 (0.5x - 2.0x)...")
            next_weights = clip_weights(current_weights, suggested_weights)

            # 5. 履歴に保存
            history.append({
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(),
                "weights": current_weights,
                "metrics": json.loads(metrics_str) if not metrics_str.startswith("No") else metrics_str,
                "observations": result.get("observations", ""),
                "reasoning": result.get("reasoning", ""),
                "suggested_weights": suggested_weights,
                "next_weights": next_weights,
                "video": str(video_path),
                "log_dir": str(log_dir),
            })
            save_history(history)

            current_weights = next_weights
            iteration += 1

    except KeyboardInterrupt:
        print("\n[INFO] 手動停止されました。ログ: ", LOG_ROOT)


if __name__ == "__main__":
    main()