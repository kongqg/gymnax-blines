# ===== learning curve plot (multiple PKLs on ONE figure) =====
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def load_pkl_object(pkl_path: str | os.PathLike):
    """Load pickle object from disk."""
    pkl_path = Path(pkl_path)
    with pkl_path.open("rb") as f:
        return pickle.load(f)


def plot_learning_curves(
    pkl_files,
    labels=None,
    key_steps="log_steps",
    key_return="log_return",
    title="Learning Curves",
    save_path=None,
):
    """
    pkl_files: list[str]  每个pkl里是dict，且包含 log_steps / log_return（和你repo里一样）
    labels:   list[str]  每条曲线的图例名；不传则用文件名
    """
    # --- style: mimic your notebook look ---
    mpl.rcParams["figure.dpi"] = 300
    plt.style.use("seaborn-v0_8-whitegrid")  # 不依赖 seaborn 包也能用
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    pkl_files = [str(p) for p in pkl_files]
    if labels is None:
        labels = [Path(p).stem for p in pkl_files]
    assert len(labels) == len(pkl_files), "labels长度必须和pkl_files一致"

    fig, ax = plt.subplots(figsize=(20, 12))

    for pkl_path, lab in zip(pkl_files, labels):
        data = load_pkl_object(pkl_path)
        if not isinstance(data, dict):
            raise TypeError(f"{pkl_path} 里不是 dict（拿到的是 {type(data)}），无法按log_steps/log_return画图")

        if key_steps not in data or key_return not in data:
            raise KeyError(
                f"{pkl_path} 缺少字段：需要 '{key_steps}' 和 '{key_return}'，"
                f"但实际 keys 有：{list(data.keys())[:30]}"
            )

        steps = np.asarray(data[key_steps])
        rets  = np.asarray(data[key_return])

        ax.plot(steps, rets, label=lab)
        print(f"[OK] {lab}: last_return={rets[-1]}")

    ax.set_xlabel("Env Steps")
    ax.set_ylabel("Return")
    ax.set_title(title)

    # 和你 notebook 一样：tick 不要太密
    ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300)
        print(f"[Saved] {save_path}")

    plt.show()


# ===== 你只要改这里：给一个list =====
pkl_files = [
    # "output/CartPole-v1/ppo.pkl",
    # "output/CartPole-v1/dhvl.pkl",
    # "output/CartPole-v1/dhvl1.pkl",
    # "output/CartPole-v1/dhvl2.pkl",
    # "output/CartPole-v1/dhvl8.pkl",
    # "output/CartPole-v1/dhvl6.pkl",
    # "output/CartPole-v1/dhvl7.pkl",
    "output/CartPole-v1/dhvl11.pkl",
    "output/CartPole-v1/dhvl12.pkl",
    "output/CartPole-v1/dhvl13.pkl",
]

labels = [
    # "PPO",
    # "dhvl==ppo",
    # "dhvl_0.6",
    # "dhvl_0.7",
    # "dhvl_0.4",
    # "dhvl_0.8",
    # "dhvl_0.3",
    "dhvl_0.52",
    "dhvl_0.50_1.82",
    "dhvl_0.6",
]

plot_learning_curves(
    pkl_files=pkl_files,
    labels=labels,
    title="CartPole-v1 Learning Curves",
    save_path="output/CartPole-v1/lcurves_single.png",  # 不想保存就设为 None
)
