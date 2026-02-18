"""
_expected_foot_height_bezier のビジュアライザ

Usage:
    streamlit run plot_bezier.py
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Foot Height Bezier Visualizer", layout="wide")
st.title("_expected_foot_height_bezier ビジュアライザ")


def cubic_bezier_interpolation(y_start, y_end, x):
    y_diff = y_end - y_start
    bezier = x**3 + 3 * (x**2 * (1 - x))
    return y_start + y_diff * bezier


def expected_foot_height_bezier(phi, swing_height, stance_ratio=0.6):
    x = (phi + np.pi) / (2 * np.pi)  # [0, 1]
    t = np.clip((x - stance_ratio) / (1.0 - stance_ratio), 0.0, 1.0)
    up   = cubic_bezier_interpolation(0.0, swing_height, 2 * t)
    down = cubic_bezier_interpolation(swing_height, 0.0, 2 * t - 1)
    profile = np.where(t <= 0.5, up, down)
    return np.where(x <= stance_ratio, 0.0, profile)


# --- サイドバー: パラメータ設定 ---
st.sidebar.header("パラメータ設定")

swing_height = st.sidebar.slider(
    "swing_height [m]",
    min_value=0.01,
    max_value=0.30,
    value=0.09,
    step=0.005,
    format="%.3f",
)

stance_ratio = st.sidebar.slider(
    "stance_ratio（支持脚の割合）",
    min_value=0.0,
    max_value=0.9,
    value=0.6,
    step=0.05,
    format="%.2f",
)

phase_offset_deg = st.sidebar.slider(
    "左右の位相オフセット [°]",
    min_value=0,
    max_value=360,
    value=180,
    step=5,
)
phase_offset = np.deg2rad(phase_offset_deg)

frequency = st.sidebar.slider(
    "歩行周波数 [Hz]",
    min_value=0.5,
    max_value=4.0,
    value=1.6,
    step=0.1,
)

swing_ratio = 1.0 - stance_ratio

st.sidebar.markdown("---")
st.sidebar.markdown("**現在の設定**")
st.sidebar.markdown(f"- swing_height: `{swing_height:.3f}` m")
st.sidebar.markdown(f"- stance_ratio: `{stance_ratio:.2f}` → 支持脚 **{stance_ratio*100:.0f}%** / 遊脚 **{swing_ratio*100:.0f}%**")
st.sidebar.markdown(f"- 位相オフセット: `{phase_offset_deg}°`")
st.sidebar.markdown(f"- 周波数: `{frequency}` Hz")

# --- データ生成 ---
phi = np.linspace(-np.pi, np.pi, 1000)

height_left  = expected_foot_height_bezier(phi, swing_height, stance_ratio)
height_right = expected_foot_height_bezier(
    np.mod(phi + phase_offset + np.pi, 2 * np.pi) - np.pi,
    swing_height, stance_ratio
)

# 支持脚区間のマスク（height=0かつスタンス区間）
x = (phi + np.pi) / (2 * np.pi)
stance_mask = x <= stance_ratio

# --- グラフ1: 位相 vs 足高さ ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("位相 vs 足の高さ（1周期）")

    fig1 = go.Figure()

    # 支持脚区間をシェーディング
    fig1.add_vrect(
        x0=-180,
        x1=np.rad2deg(-np.pi + stance_ratio * 2 * np.pi),
        fillcolor="lightblue", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="支持脚区間", annotation_position="top left",
    )

    fig1.add_trace(go.Scatter(
        x=np.rad2deg(phi),
        y=height_left,
        mode="lines",
        name="左足 (phase[:,0])",
        line=dict(color="blue", width=2.5),
    ))
    fig1.add_trace(go.Scatter(
        x=np.rad2deg(phi),
        y=height_right,
        mode="lines",
        name=f"右足 (phase[:,1], +{phase_offset_deg}°)",
        line=dict(color="red", width=2.5, dash="dash"),
    ))
    fig1.add_hline(
        y=swing_height, line_dash="dot", line_color="green",
        annotation_text=f"swing_height={swing_height:.3f}m",
        annotation_position="top right",
    )
    fig1.update_layout(
        xaxis_title="位相 φ [°]",
        yaxis_title="目標足高さ [m]",
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
        xaxis=dict(tickmode="linear", tick0=-180, dtick=45, range=[-180, 180]),
        yaxis=dict(range=[-0.01, swing_height * 1.2]),
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("時間 vs 足の高さ（複数周期）")

    n_cycles = 3
    t_arr = np.linspace(0, n_cycles / frequency, 2000)
    phase_dt = 2 * np.pi * frequency

    phi_t_left  = np.mod(t_arr * phase_dt + np.pi, 2 * np.pi) - np.pi
    phi_t_right = np.mod(t_arr * phase_dt + phase_offset + np.pi, 2 * np.pi) - np.pi

    height_t_left  = expected_foot_height_bezier(phi_t_left,  swing_height, stance_ratio)
    height_t_right = expected_foot_height_bezier(phi_t_right, swing_height, stance_ratio)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=t_arr, y=height_t_left,
        mode="lines", name="左足",
        line=dict(color="blue", width=2.5),
    ))
    fig2.add_trace(go.Scatter(
        x=t_arr, y=height_t_right,
        mode="lines", name="右足",
        line=dict(color="red", width=2.5, dash="dash"),
    ))
    fig2.add_hline(
        y=swing_height, line_dash="dot", line_color="green",
        annotation_text=f"swing_height={swing_height:.3f}m",
        annotation_position="top right",
    )
    fig2.update_layout(
        xaxis_title="時間 [s]",
        yaxis_title="目標足高さ [m]",
        legend=dict(x=0.02, y=0.98),
        hovermode="x unified",
        yaxis=dict(range=[-0.01, swing_height * 1.2]),
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- 旧実装との比較 ---
st.subheader("旧実装（stance_ratio なし）との比較")

def old_expected_foot_height_bezier(phi, swing_height):
    x = (phi + np.pi) / (2 * np.pi)
    stance = cubic_bezier_interpolation(0.0, swing_height, 2 * x)
    swing  = cubic_bezier_interpolation(swing_height, 0.0, 2 * x - 1)
    return np.where(x <= 0.5, stance, swing)

height_old = old_expected_foot_height_bezier(phi, swing_height)
height_new = expected_foot_height_bezier(phi, swing_height, stance_ratio)

fig3 = go.Figure()
fig3.add_vrect(
    x0=-180,
    x1=np.rad2deg(-np.pi + stance_ratio * 2 * np.pi),
    fillcolor="lightblue", opacity=0.2,
    layer="below", line_width=0,
    annotation_text=f"支持脚 {stance_ratio*100:.0f}%",
    annotation_position="top left",
)
fig3.add_trace(go.Scatter(
    x=np.rad2deg(phi), y=height_old,
    mode="lines", name="旧実装（支持脚なし）",
    line=dict(color="gray", width=2, dash="dot"),
))
fig3.add_trace(go.Scatter(
    x=np.rad2deg(phi), y=height_new,
    mode="lines", name=f"新実装（stance_ratio={stance_ratio:.2f}）",
    line=dict(color="blue", width=2.5),
))
fig3.add_hline(
    y=swing_height, line_dash="dot", line_color="green",
    annotation_text=f"swing_height={swing_height:.3f}m",
    annotation_position="top right",
)
fig3.update_layout(
    xaxis_title="位相 φ [°]",
    yaxis_title="目標足高さ [m]",
    legend=dict(x=0.5, y=0.98),
    hovermode="x unified",
    xaxis=dict(tickmode="linear", tick0=-180, dtick=45, range=[-180, 180]),
    yaxis=dict(range=[-0.01, swing_height * 1.2]),
    height=350,
)
st.plotly_chart(fig3, use_container_width=True)

# --- 数式の説明 ---
with st.expander("実装の詳細"):
    st.markdown(r"""
### `_expected_foot_height_bezier(phi, swing_height, stance_ratio)`

位相 $\phi \in [-\pi, \pi]$ を $x \in [0, 1]$ に正規化:

$$x = \frac{\phi + \pi}{2\pi}$$

スウィング区間内での正規化 $t \in [0, 1]$:

$$t = \text{clip}\!\left(\frac{x - \text{stance\_ratio}}{1 - \text{stance\_ratio}},\ 0,\ 1\right)$$

**Bézier 補間関数:**
$$\text{bezier}(s) = s^3 + 3s^2(1-s) = 3s^2 - 2s^3 \quad (\text{cubic Hermite smoothstep})$$

**目標足高さ:**

$$h(\phi) = \begin{cases}
0 & (x \leq \text{stance\_ratio}) \\
\text{bezier}(2t) \cdot \text{swing\_height} & (x > \text{stance\_ratio},\ t \leq 0.5) \\
(1 - \text{bezier}(2t-1)) \cdot \text{swing\_height} & (x > \text{stance\_ratio},\ t > 0.5)
\end{cases}$$
""")
