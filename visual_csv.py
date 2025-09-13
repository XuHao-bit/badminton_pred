import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def visual_df(result_dir, df):
    save_path = './visualization'
    suffix = result_dir.split('/')[-1].split('.')[0]

    # 提取预测和真实的 xy
    pred_xy = df[["pred_x", "pred_y"]].values
    target_xy = df[["label_x", "label_y"]].values

    # 随机采样 30%
    N = pred_xy.shape[0]
    mask = np.random.rand(N) < 0.5
    pred_xy = pred_xy[mask]
    target_xy = target_xy[mask]

    # 绘图
    plt.figure(figsize=(10, 8))
    for (px, py), (tx, ty) in zip(pred_xy, target_xy):
        plt.scatter(tx, ty, c="blue", marker="o", s=15, label="True" if "True" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.scatter(px, py, c="red", marker="x", s=20, label="Pred" if "Pred" not in plt.gca().get_legend_handles_labels()[1] else "")
        plt.plot([tx, px], [ty, py], c="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Predicted vs True Landing Points (50% sampled)")
    plt.grid(True)
    plt.legend()
    # plt.savefig("pred_vs_true_xy.png", dpi=300)
    file_name = suffix + "_visual_result.pdf"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)


    # ===== 计算误差 =====
    x_error = df["pred_x"][mask].values - df["label_x"][mask].values
    y_error = df["pred_y"][mask].values - df["label_y"][mask].values
    time_error = df["pred_time"][mask].values - df["label_time"][mask].values

    # ===== 绘制落点误差 (X, Y) =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x_error, y_error, alpha=0.6, s=10)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlabel("X error")
    plt.ylabel("Y error")
    plt.title("Landing Point Error Distribution")
    plt.grid(True)

    # ===== 绘制时间误差直方图 =====
    plt.subplot(1, 2, 2)
    plt.hist(time_error, bins=50, alpha=0.7, color="steelblue")
    plt.axvline(0, color="red", linestyle="--", label="Zero Error")
    plt.xlabel("Time Error")
    plt.ylabel("Count")
    plt.title("Time Error Distribution")
    plt.legend()

    plt.tight_layout()
    file_name = suffix + "_error_distributions.pdf"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)

if __name__ == '__main__':
    from main import set_seed
    
    set_seed(seed=42)
    result_dir = "./results/SimplifiedLSTMRegressor_20250913_155153.csv"
    # 读取结果
    df = pd.read_csv(result_dir)
    visual_df(result_dir, df)