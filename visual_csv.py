import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 可视化函数
def visualize_group(group, group_name, log_time, avg_err, save_path):
    line_colors = [
        (0.7, 0.7, 0.7), # Darker Light Gray (比原 0.9 调深，更接近中灰)
        (0.4, 0.4, 0.8), # Darker Light Blue (比原 0.6 调深，蓝调更明显)
        (0.8, 0.4, 0.4) # Darker Light Red (比原 0.6 调深，绿调更明显)
        ]

    # 提取预测和真实的xy坐标
    pred_xy = group[["pred_x", "pred_y"]].values
    target_xy = group[["label_x", "label_y"]].values
    
    # 随机采样30%
    threshold = 1
    N = pred_xy.shape[0]
    if N > 0:  # 防止空组
        mask = np.random.rand(N) < threshold
        pred_xy = pred_xy[mask]
        target_xy = target_xy[mask]
        
        # 绘图
        plt.figure(figsize=(10, 8))
        for (px, py), (tx, ty) in zip(pred_xy, target_xy):
            random_color = np.random.choice(len(line_colors)) # Random index (0/1/2)
            # 避免重复添加图例
            plt.scatter(tx, ty, c="blue", marker="o", s=15, 
                    label="True" if "True" not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.scatter(px, py, c="red", marker="x", s=20, 
                    label="Predict" if "Predict" not in plt.gca().get_legend_handles_labels()[1] else "")
            # 绘制连接线表示误差
            plt.plot([tx, px], [ty, py], c=line_colors[random_color], linestyle="--", linewidth=0.5, alpha=0.6)
        
        plt.xlim(0, 600)  # X轴限制在±800cm
        plt.ylim(-300, 300)  # Y轴限制在±800cm
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.title(f"{group_name} (Avg L2 Dist: {avg_err:.2f}, Sampling Rate: {threshold})")
        plt.grid(True)
        plt.legend()
        
        # 保存图像
        file_name = f"{group_name}_{log_time}_visual_result.pdf"
        file_path = os.path.join(save_path, file_name)
        plt.savefig(file_path)
        plt.close()
        print(f"已保存{group_name}可视化结果至: {file_path}")


def visual_df(name, log_time, df):
    save_path = f'./visualization/{name}/'
    os.makedirs(save_path, exist_ok=True)

    # 计算每个轴的误差（预测值 - 真实值）
    df['err_x'] = df['pred_x'] - df['label_x']
    df['err_y'] = df['pred_y'] - df['label_y']
    df['err_z'] = df['pred_z'] - df['label_z']

    # 计算每个样本的欧氏距离误差（三维空间中两点间的直线距离）
    df['euclidean_error'] = np.sqrt(df['err_x']**2 + df['err_y']** 2 + df['err_z']**2)
    avg_euclidean_err, mid_eu_err = df['euclidean_error'].mean(), df['euclidean_error'].median()

    print("误差统计结果：")
    print(avg_euclidean_err, mid_eu_err)

    # 根据欧氏距离的平均误差分组
    group_high = df[df['euclidean_error'] > avg_euclidean_err]  # 高于平均误差的组
    group_low = df[df['euclidean_error'] <= avg_euclidean_err]  # 低于等于平均误差的组

    print(f"\n样本总数: {len(df)}")
    print(f"高于平均欧氏距离误差的样本数: {len(group_high)}")
    print(f"低于等于平均欧氏距离误差的样本数: {len(group_low)}")
    
    # 创建保存目录
    save_path = "./visualization/xyz_error_groups/"
    os.makedirs(save_path, exist_ok=True)

    # 可视化两组数据
    visualize_group(group_high, "Above_Avg_Err", log_time, avg_euclidean_err, save_path)
    visualize_group(group_low, "Below_Avg_Err", log_time, avg_euclidean_err, save_path)

    # # 提取预测和真实的 xy
    # pred_xy = df[["pred_x", "pred_y"]].values
    # target_xy = df[["label_x", "label_y"]].values

    # # 随机采样 30%
    # threshold = 0.3
    # N = pred_xy.shape[0]
    # mask = np.random.rand(N) < 0.3
    # pred_xy = pred_xy[mask]
    # target_xy = target_xy[mask]

    # # 绘图
    # plt.figure(figsize=(10, 8))
    # for (px, py), (tx, ty) in zip(pred_xy, target_xy):
    #     plt.scatter(tx, ty, c="blue", marker="o", s=15, label="True" if "True" not in plt.gca().get_legend_handles_labels()[1] else "")
    #     plt.scatter(px, py, c="red", marker="x", s=20, label="Pred" if "Pred" not in plt.gca().get_legend_handles_labels()[1] else "")
    #     plt.plot([tx, px], [ty, py], c="gray", linestyle="--", linewidth=0.5, alpha=0.6)

    # plt.xlabel("X Coordinate")
    # plt.ylabel("Y Coordinate")
    # plt.title(f"Predicted vs True Landing Points ({threshold} sampled)")
    # plt.grid(True)
    # plt.legend()
    # # plt.savefig("pred_vs_true_xy.png", dpi=300)
    # file_name = name + "_" + log_time + "_visual_result.pdf"
    # file_path = os.path.join(save_path, file_name)
    # plt.savefig(file_path)


    # ===== 计算误差 =====
    # x_error = df["pred_x"][mask].values - df["label_x"][mask].values
    # y_error = df["pred_y"][mask].values - df["label_y"][mask].values
    # time_error = df["pred_time"][mask].values - df["label_time"][mask].values
    x_error = df["pred_x"].values - df["label_x"].values
    y_error = df["pred_y"].values - df["label_y"].values
    time_error = df["pred_time"].values - df["label_time"].values

    # ===== 绘制落点误差 (X, Y) =====
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(x_error, y_error, alpha=0.6, s=10)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlim(-400, 400)  # X轴限制在±800cm
    plt.ylim(-400, 400)  # Y轴限制在±800cm
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
    file_name = name + "_" + log_time + "_error_distributions.pdf"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)

if __name__ == '__main__':
    from main import set_seed
    
    set_seed(seed=42)
    result_dir = "./results/TransformerModel/20250918_134619.csv"
    # 读取结果
    df = pd.read_csv(result_dir)
    visual_df('TransformerModel', '20250918_134619', df)