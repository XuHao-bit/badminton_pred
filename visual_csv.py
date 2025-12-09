import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
import seaborn as sns
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
    df['err_time'] = abs(df['pred_time'] - df['label_time'])

    # 计算每个样本的欧氏距离误差（三维空间中两点间的直线距离）
    df['err_euclidean'] = np.sqrt(df['err_x'] ** 2 + df['err_y'] ** 2 + df['err_z'] ** 2)
    avg_euclidean_err, mid_eu_err = df['err_euclidean'].mean(), df['err_euclidean'].median()
    p70_euclidean_err = df['err_euclidean'].quantile(0.7)
    p90_euclidean_err = df['err_euclidean'].quantile(0.9)
    avg_time_err = df['err_time'].mean()
    pred_dir = df[["pred_dir_x", "pred_dir_y"]].values
    label_dir = df[["label_dir_x", "label_dir_y"]].values
    cos_sim = np.sum(pred_dir * label_dir, axis=1) / (
            np.linalg.norm(pred_dir, axis=1) * np.linalg.norm(label_dir, axis=1)
    )
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 避免数值超出 [-1,1]
    df['err_dir'] = np.degrees(np.arccos(cos_sim))
    avg_dir_err = df['err_dir'].mean()

    df_xyz_sorted = df.sort_values(by='err_euclidean')
    df_time_sorted = df.sort_values(by='err_time')
    df_dir_sorted = df.sort_values(by='err_dir')

    print("误差统计结果：")
    print("落点预测误差: ", avg_euclidean_err)
    print("时间预测误差: ", avg_time_err)
    print("角度预测误差: ", avg_dir_err)

    # 根据欧氏距离的平均误差分组
    group_high = df[df['err_euclidean'] > avg_euclidean_err]  # 高于平均误差的组
    group_low = df[df['err_euclidean'] <= avg_euclidean_err]  # 低于等于平均误差的组

    print(f"\n样本总数: {len(df)}")
    print(f"高于平均欧氏距离误差的样本数: {len(group_high)}")
    print(f"低于等于平均欧氏距离误差的样本数: {len(group_low)}")

    # 创建保存目录
    save_path = "./visualization/xyz_error_groups/"
    os.makedirs(save_path, exist_ok=True)

    # 可视化两组数据
    visualize_group(group_high, "Above_Avg_Err", log_time, avg_euclidean_err, save_path)
    visualize_group(group_low, "Below_Avg_Err", log_time, avg_euclidean_err, save_path)

    # print("\n坐标预测精度最好的10个序列：")
    # print(df_xyz_sorted.head(10)[['file_name', 'err_euclidean']].to_string(index=False))
    #
    # print("\n时间预测精度最好的10个序列：")
    # print(df_time_sorted.head(10)[['file_name', 'err_time']].to_string(index=False))

    print("\n坐标预测精度最差的10个序列：")
    print(df_xyz_sorted.tail(10)[['file_name', 'err_euclidean']].to_string(index=False))

    print("\n时间预测精度最差的10个序列：")
    print(df_time_sorted.tail(10)[['file_name', 'err_time']].to_string(index=False))

    # print("\nError 200-250随机采样的10个序列：")
    # print(df_sampled[['file_name', 'euclidean_error']].to_string(index=False))

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

    # ===== 计算方向误差 (角度，单位：度) =====
    pred_dir = df[["pred_dir_x", "pred_dir_y"]].values
    true_dir = df[["label_dir_x", "label_dir_y"]].values
    cos_sim = np.sum(pred_dir * true_dir, axis=1) / (
            np.linalg.norm(pred_dir, axis=1) * np.linalg.norm(true_dir, axis=1)
    )
    cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 避免浮点误差
    direction_error = np.degrees(np.arccos(cos_sim))

    # ===== 绘制落点误差 (X, Y)、时间误差、方向误差 =====
    plt.figure(figsize=(12, 10))  # 更接近正方形的画布

    # (1) 落点误差散点
    plt.subplot(2, 2, 1)
    plt.scatter(x_error, y_error, alpha=0.6, s=10)
    plt.axhline(0, color="gray", linestyle="--")
    plt.axvline(0, color="gray", linestyle="--")
    plt.xlim(-400, 400)
    plt.ylim(-400, 400)
    plt.xlabel("X error")
    plt.ylabel("Y error")
    plt.title("Landing Point Error Distribution")
    plt.grid(True)

    # (3) 欧氏距离误差直方图
    plt.subplot(2, 2, 2)
    plt.hist(df['err_euclidean'].values, bins=50, alpha=0.7, color="steelblue")
    plt.axvline(mid_eu_err, color="red", linestyle="--", label="Mid Error")
    plt.axvline(avg_euclidean_err, color="green", linestyle="--", label="Avg Error")
    plt.axvline(p70_euclidean_err, color="blue", linestyle="--", label="70% Error")
    plt.axvline(p90_euclidean_err, color="yellow", linestyle="--", label="90% Error")
    plt.xlim(0, 500)
    plt.xlabel("Euclidean Error")
    plt.ylabel("Count")
    plt.title("Euclidean Error Distribution")
    plt.legend()

    # (3) 时间误差直方图
    plt.subplot(2, 2, 3)
    plt.hist(time_error, bins=50, alpha=0.7, color="steelblue")
    plt.axvline(0, color="red", linestyle="--", label="Zero Error")
    plt.xlim(-250, 250)
    plt.xlabel("Time Error")
    plt.ylabel("Count")
    plt.title("Time Error Distribution")
    plt.legend()

    # (4) 方向误差直方图
    plt.subplot(2, 2, 4)
    plt.hist(direction_error, bins=50, alpha=0.7, color="seagreen")
    plt.xlim(0, 200)
    plt.xlabel("Direction Error (°)")
    plt.ylabel("Count")
    plt.title("Direction Error Distribution")
    plt.grid(True)

    plt.tight_layout()
    file_name = name + "_" + log_time + "_error_distributions.pdf"
    file_path = os.path.join(save_path, file_name)
    plt.savefig(file_path)


def visual_uncertainty(df, threshold=75):
    print("\n======== 可视化不确定性估计 ========")
    # df['std_euclidean'] = np.sqrt(df['std_total_x'] ** 2 + df['std_total_y'] ** 2 + df['std_total_z'] ** 2)
    # df['std_euclidean'] = np.sqrt(df['std_epistemic_x'] ** 2 + df['std_epistemic_y'] ** 2 + df['std_epistemic_z'] ** 2)
    df['std_euclidean'] = np.sqrt(df['std_aleatoric_x'] ** 2 + df['std_aleatoric_y'] ** 2 + df['std_aleatoric_z'] ** 2)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='err_euclidean', y='std_euclidean', data=df, alpha=0.6, s=20)
    plt.title(f'Predictive Error vs. Predictive Uncertainty')
    plt.xlabel(f'Predictive Error')
    plt.ylabel(f'Predictive Uncertainty')
    plt.ylim(0, 225)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('visualization/Error-std aleatoric Correlation')
    # plt.savefig('visualization/Error-std epistemic Correlation')
    # plt.savefig('visualization/Error-std Correlation')
    print(f"\n成功保存预测误差-不确定性的散点图")

    df_less_1m = df[df['err_euclidean'] <= 100]
    df_more_1m = df[df['err_euclidean'] > 100]

    less_1m = len(df_less_1m[df_less_1m['std_euclidean'] < threshold])
    more_1m = len(df_more_1m[df_more_1m['std_euclidean'] > threshold])
    print(f"在设定不确定性的截断阈值为{threshold}的情况下：")
    print(f"保留的<1m样本占比 {less_1m / len(df_less_1m)}")
    print(f"丢弃的>1m样本占比 {more_1m / len(df_more_1m)}")
    print("\n================================\n")


def parse_pose_sequence(lines):
    """解析文本数据"""
    frames = []
    for line in lines:
        coords = np.array([float(num) for num in line.strip().split(':')[1].split(',')])
        coords = coords[:63]
        frames.append(coords.reshape(21, 3))
    return np.array(frames)  # Shape: (100, 21, 3)


def categorize_shot_custom(sequence):
    WRIST_IDX = 20  # 示例：右手手腕 (或球拍关键点)
    HIP_IDX = 12  # 示例：骨盆中心 (代表人的整体位置)

    # 坐标轴定义 (假设数据是 x, y, z)
    # 请确认哪个轴代表"高度"，哪个轴代表"前后场纵深"
    AXIS_HEIGHT = 2  # 示例：Z轴是高度
    AXIS_DEPTH = 0  # 示例：Y轴是纵深 (从网前到后场)

    # ================= 1. 配置区域 (请填入你直方图中观察到的数值) =================

    # A. 击球位置 (纵深) 阈值 -> 需要 2 个数值来切分 前/中/后
    # [前场与中场的分界线, 中场与后场的分界线]
    # 示例: 假设数据归一化在 -1~1 之间, 0是中心
    THRESH_DEPTH = [980, 1120]

    THRESH_HEIGHT = 175

    THRESH_VELOCITY = 8

    """
    根据: 位置(3) x 高度(2) x 速度(2) 进行分类
    """
    # 1. 获取关键数值
    hit_frame = sequence[-1]
    prev_frame = sequence[-10]

    h_val = hit_frame[WRIST_IDX, AXIS_HEIGHT]
    p_val = hit_frame[HIP_IDX, AXIS_DEPTH]

    # 计算速度 (最后10帧位移)
    disp = np.linalg.norm(hit_frame[WRIST_IDX] - prev_frame[WRIST_IDX])
    v_val = disp / 10.0

    # 2. 逻辑判定

    # --- 位置判定 (Front / Mid / Back) ---
    if p_val < THRESH_DEPTH[0]:
        pos_tag = "Front"
    elif p_val < THRESH_DEPTH[1]:
        pos_tag = "Mid"
    else:
        pos_tag = "Back"

    # --- 高度判定 (High / Low) ---
    if h_val > THRESH_HEIGHT:
        height_tag = "High"
    else:
        height_tag = "Low"

    # --- 速度判定 (Fast / Slow) ---
    if v_val > THRESH_VELOCITY:
        speed_tag = "Fast"
    else:
        speed_tag = "Slow"

    # 3. 生成组合标签 (例如: Back_High_Fast)
    full_tag = f"{pos_tag}_{height_tag}_{speed_tag}"

    # 4. 生成易读的羽毛球术语 (Optional: 辅助理解)
    # 你可以根据这个映射表来检查分类是否合理
    readable_name = full_tag  # 默认用标签

    if pos_tag == "Back":
        if height_tag == "High":
            readable_name = "Smash/Clear (杀/高)" if speed_tag == "Fast" else "Drop (吊球)"
        else:  # Back Low
            readable_name = "Low Defense (后场低手接杀)"

    elif pos_tag == "Front":
        if height_tag == "Low":
            readable_name = "Rush/Lift (扑/挑)" if speed_tag == "Fast" else "Net Spin (搓/放)"
        else:  # Front High
            readable_name = "Net Kill (网前封网)"

    elif pos_tag == "Mid":
        if speed_tag == "Fast":
            readable_name = "Drive (平抽挡)"
        else:
            readable_name = "Mid Transition (中场过渡)"

    return full_tag, readable_name, h_val, p_val, v_val


def visual_samples_distribution(df):
    WRIST_IDX = 20  # 示例：右手手腕 (或球拍关键点)
    HIP_IDX = 12  # 示例：骨盆中心 (代表人的整体位置)

    # 坐标轴定义 (假设数据是 x, y, z)
    # 请确认哪个轴代表"高度"，哪个轴代表"前后场纵深"
    AXIS_HEIGHT = 2  # 示例：Z轴是高度
    AXIS_DEPTH = 0  # 示例：Y轴是纵深 (从网前到后场)

    positions = []
    heights = []
    velocities = []

    print("正在读取文件并提取特征...")

    # 假设 df 是你之前加载好的 DataFrame
    for index, row in df.iterrows():
        file_name = row['file_name']
        file_path = os.path.join(data_folder, file_name)

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            del lines[-1]

            seq = parse_pose_sequence(lines)

            # 1. 击球高度 (取最后一帧)
            h = seq[-1, WRIST_IDX, AXIS_HEIGHT]

            # 2. 击球位置 (取最后一帧)
            p = seq[-1, HIP_IDX, AXIS_DEPTH]

            # 3. 挥拍速度 (取最后10帧的位移均值)
            # 计算第50帧和第40帧之间手腕的欧氏距离
            curr_pos = seq[-1, WRIST_IDX]
            prev_pos = seq[-10, WRIST_IDX]
            dist = np.linalg.norm(curr_pos - prev_pos)
            v = dist / 10.0  # 单位：距离/帧

            heights.append(h)
            positions.append(p)
            velocities.append(v)

        except Exception as e:
            # 忽略读取错误的样本，防止中断
            continue

    # 转换为 Numpy 数组方便绘图
    heights = np.array(heights)
    positions = np.array(positions)
    velocities = np.array(velocities)

    print(f"成功提取 {len(heights)} 个样本的特征。")

    # ================= 2. 绘图区域 =================

    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")  # 设置背景网格

    # --- 子图 1: 击球高度分布 ---
    plt.subplot(2, 2, 1)
    sns.histplot(positions, bins=50, kde=True, color='salmon')
    plt.title('Distribution of Court Position')
    plt.xlabel('Depth Coordinate (Normalized)')
    plt.ylabel('Count')

    # --- 子图 2: 场地位置分布 ---
    plt.subplot(2, 2, 2)
    sns.histplot(heights, bins=50, kde=True, color='skyblue')
    plt.title('Distribution of Hit Height')
    plt.xlabel('Height Coordinate (Normalized)')
    plt.ylabel('Count')

    # --- 子图 3: 挥拍速度分布 ---
    plt.subplot(2, 2, 3)
    sns.histplot(velocities, bins=50, kde=True, color='green')
    plt.title('Distribution of Hit Velocity')
    plt.xlabel('Velocity (Dist / Frame)')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig('visualization/Position, Height, Velocity Distribution')


def visual_shot_categories(df):
    results_list = []
    print("开始分类统计...")

    # 假设 df 是包含 file_name 和 err_euclidean 的 DataFrame
    # data_folder 是你的数据路径

    for index, row in df.iterrows():
        file_name = row['file_name']
        file_path = os.path.join(data_folder, file_name)
        original_err = row['err_euclidean']

        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            del lines[-1]

            seq = parse_pose_sequence(lines)

            # 调用新的分类函数
            tag, readable, h, p, v = categorize_shot_custom(seq)

            results_list.append({
                'file_name': file_name,
                'error': original_err,
                'tag': tag,  # 组合标签 (Back_High_Fast)
                'readable_type': readable,  # 易读术语
                'position': tag.split('_')[0],
                'height': tag.split('_')[1],
                'speed': tag.split('_')[2],
                'val_h': h,
                'val_p': p,
                'val_v': v
            })

        except Exception as e:
            print(f"Skipping {file_name}: {e}")

    # 生成结果 DataFrame，过滤掉个别数量极少的类别
    res_df = pd.DataFrame(results_list)
    res_df = res_df[res_df['readable_type'] != "Low Defense (后场低手接杀)"]
    res_df = res_df[res_df['readable_type'] != "Rush/Lift (扑/挑)"]
    res_df = res_df[res_df['readable_type'] != "Net Kill (网前封网)"]

    # ================= 结果分析输出 =================

    print("\n====== 1. 各组合类别的平均误差 (按误差从大到小排序) ======")
    # 重点看这里：哪个组合的 Error 最大？
    summary = res_df.groupby('readable_type')['error'].agg(['count', 'mean', 'std'])
    print(summary.sort_values(by='mean', ascending=False))

    print("\n====== 2. 维度拆解分析 (查看哪个单一因素影响最大) ======")
    print("位置影响:")
    print(res_df.groupby('position')['error'].mean())
    print("\n高度影响:")
    print(res_df.groupby('height')['error'].mean())
    print("\n速度影响:")
    print(res_df.groupby('speed')['error'].mean())

    # ============================================

    XLIM_MAX = 500  # 只显示 0 到 3 米误差范围内的分布
    BIN_WIDTH = 20  # 每一个直方条代表 10cm 的区间
    PALETTE = "tab10"
    # ===========================================

    sns.set_style("whitegrid")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    # sns.set_style("whitegrid")  # 设置白色网格背景，方便读数

    # 绘制堆叠直方图
    # data: 数据源
    # x: 误差数值列
    # hue: 分类列 (不同颜色)
    # multiple="stack": 堆叠模式 (如果不加这个，颜色会重叠覆盖)
    # binwidth: 设定柱子的宽度
    sns.violinplot(
        data=res_df,
        x='error',  # 误差在 X 轴
        y='readable_type',  # 类别在 Y 轴 (这样是水平排列，更易读)
        hue='readable_type',  # 颜色区分
        palette="tab10",
        order=["Drop (吊球)", "Smash/Clear (杀/高)", "Net Spin (搓/放)", "Mid Transition (中场过渡)"],
        inner="quartile",  # 【关键】在小提琴内部画出 四分位数线 (虚线)
        linewidth=1.5,
        cut=0  # 限制范围，不允许推测超出数据极值的范围
    )

    # --- 图表美化 ---
    plt.title('Prediction Error Distribution by Shot Type', fontsize=16, pad=20)
    plt.xlabel('Prediction Error', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xlim(0, XLIM_MAX)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)

    # 绘制一条竖线：显示整体的平均误差 (假设是 0.9m)
    # mean_err = res_df['error'].mean()
    # # plt.axvline(mean_err, color='red', linestyle='--')
    #
    # # 步骤 A: 获取 Seaborn 自动生成的分类图例 (它现在已经在图上了)
    # leg1 = ax.get_legend()
    # # 可以选择性地设置标题和位置
    # leg1.set_title("Shot Type")
    # # 将其移到图外右上方 (bbox_to_anchor 控制坐标)
    # leg1.set_bbox_to_anchor((1.02, 1))
    # leg1.set_loc("upper left")
    #
    # # 步骤 B: 手动创建一个新的图例项，用于表示均值线
    # # 创建一个红色的虚线图例句柄
    # mean_line_handle = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2,
    #                                  label=f'Overall Mean: {mean_err:.2f}cm')
    #
    # # 步骤 C: 创建第二个图例 (仅包含均值线)
    # # 将其放在第一个图例的下方 (通过调整 bbox_to_anchor 的纵坐标)
    # # leg2 = plt.legend(handles=[mean_line_handle], bbox_to_anchor=(1.02, 0.8), loc="upper left")
    #
    # # 步骤 D: 【最关键的一步】
    # # 当我们创建 leg2 时，Matplotlib 可能会把 leg1 抹掉。
    # # 我们需要把 leg1 作为“背景元素”加回来。
    # ax.add_artist(leg1)

    plt.tight_layout()
    plt.savefig('visualization/Prediction Error on different category')


if __name__ == '__main__':
    from main import set_seed
    data_folder = '../data/data_1204_infer'
    
    set_seed(seed=42)
    # result_dir = "./results/ImprovedTransformerModel/20251029_162224.csv"
    result_dir = "./results/ImprovedTransformerModel/20251204_222914_mc20_total_uncertainty.csv"
    df = pd.read_csv(result_dir)
    visual_df('ImprovedTransformerModel', '20251127_220434', df)

    # 可视化不确定性，设定不确定性的截断阈值
    visual_uncertainty(df, threshold=75)

    # 可视化测试样本关于各个指标的分布
    visual_samples_distribution(df)
    # 可视化不同类别球的预测精度
    visual_shot_categories(df)

    # 按照落地时间label取前30%快的球
    # quantile_30 = df['label_time'].quantile(0.3)
    # print(f'\n\nTop 30% 落地时间： {quantile_30}帧')
    # df_short_time = df.loc[df['label_time'] <= quantile_30].copy()
    # # df_short_time = df.loc[df['label_time'] < 200]
    # visual_df('ImprovedTransformerModel', '20251120_213417_short', df_short_time)


    # # 绘制 error-击球位置 关系图
    # df_sorted = df.sort_values(by='err_euclidean', ascending=True)
    #
    # # 2. 从排序后的 DataFrame 中提取数据
    # pos_x, pos_y = [], []
    # errors = df_sorted['err_euclidean']
    #
    # for index, row in df.iterrows():
    #     file_name = row['file_name']
    #     file_path = os.path.join(data_folder, file_name)
    #
    #     with open(file_path, 'r') as f:
    #         lines = f.readlines()
    #         line = lines[-2]
    #         coords = [float(num) for num in line.strip().split(':')[1].split(',')]
    #         points_array = np.array(coords).reshape(21, 3)
    #         mean_coords = np.mean(points_array, axis=0)
    #         mean_x = mean_coords[0]
    #         mean_y = mean_coords[1]
    #         pos_x.append(mean_x)
    #         pos_y.append(mean_y)
    #
    # # 3. 创建图形和子图对象
    # fig, ax = plt.subplots(figsize=(8, 6))
    #
    # # 4. 使用 scatter() 绘制散点图，并根据 'euclidean_error' 着色
    # # c=errors: 指定颜色依据，Matplotlib会自动将其映射到颜色
    # # cmap='viridis': 选择一个颜色映射，从紫色到黄色
    # # s=50: 设置点的大小
    # scatter = ax.scatter(pos_x, pos_y, c=df['err_euclidean'], cmap='viridis', norm=Normalize(vmin=0, vmax=400), s=30)
    #
    # # 5. 添加颜色条（Color Bar）
    # # 颜色条会显示颜色到误差数值的映射关系
    # cbar = fig.colorbar(scatter)
    # cbar.set_label('Euclidean Error')
    # plt.xlim(670, 1340)
    # plt.ylim(-305, 305)
    #
    # # 6. 设置图表标题和标签
    # ax.set_title("Hit Position Coordinates Colored by Euclidean Err")
    # ax.set_xlabel("Position X")
    # ax.set_ylabel("Position Y")
    # plt.tight_layout()
    # plt.savefig('visualization/Position-Error Correlation')
