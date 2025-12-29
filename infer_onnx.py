import onnxruntime as ort
import numpy as np
import os

def inference_onnx(model_path, seq_data, mask_data):
    """
    model_path: .onnx 文件的路径
    seq_data: shape 为 (batch_size, seq_len, 63) 的 numpy 数组
    mask_data: shape 为 (batch_size, seq_len) 的 numpy 数组 (dtype=bool)
    """

    # 1. 创建推理会话
    session = ort.InferenceSession(model_path)

    # 2. 准备输入数据
    # 注意：ONNX 期待的数据类型必须与导出时一致
    # seq 通常是 float32，mask 是 bool
    input_seq = seq_data.astype(np.float32)
    input_mask = mask_data.astype(bool)

    # 3. 运行推理
    # input_names 必须与导出时的 ['seq', 'mask'] 一致
    onnx_inputs = {
        'seq': input_seq,
        'mask': input_mask
    }

    # run 的第一个参数为 None 表示获取所有输出，也可以指定 ['output']
    outputs = session.run(['output'], onnx_inputs)

    # outputs 是一个列表，获取第一个输出
    return outputs[0]


def batch_onnx_inference_and_eval(model_path, folder_path):
    """
    加载模型，遍历文件夹，提取最后50帧进行推理，并与最后一行Label对比计算平均误差。
    """
    # 1. 初始化 ONNX 推理会话
    # 优先尝试 GPU，否则使用 CPU
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
        'CPUExecutionProvider']
    try:
        session = ort.InferenceSession(model_path, providers=providers)
    except Exception as e:
        return f"模型加载失败: {e}"

    all_distances = []
    file_list = [f for f in os.listdir(folder_path) if f.endswith('.txt') or '.' not in f]  # 兼容无后缀文件

    print(f"开始处理文件夹: {folder_path}，检测到 {len(file_list)} 个文件。")

    for filename in file_list:
        file_path = os.path.join(folder_path, filename)

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]

            # 数据校验：至少需要 51 行（50帧数据 + 1帧Label）
            if len(lines) < 51:
                print(f"跳过文件 {filename}: 行数不足51行")
                continue

            # 2. 提取 Label (最后一行)
            label_line = lines[-1]
            label_coords = np.array([float(x) for x in label_line.split(':')[1].split(',')], dtype=np.float32)

            # 3. 提取输入序列 (倒数第2行开始向上数50行)
            # 索引说明：lines[-51:-1] 包含倒数第51行到倒数第2行，共50行
            seq_lines = lines[-51:-1]
            seq_data = []
            for line in seq_lines:
                coords = [float(x) for x in line.split(':')[1].split(',')]
                seq_data.append(coords)

            # 转换为 numpy 数组并增加 Batch 维度: (1, 50, 63)
            input_seq = np.array(seq_data, dtype=np.float32).reshape(1, 50, 63)
            # 准备 Mask 数据: (1, 50)
            input_mask = np.ones((1, 50), dtype=bool)

            # 4. 模型推理
            onnx_inputs = {
                'seq': input_seq,
                'mask': input_mask
            }
            # 假设模型输出名称为 'output'
            prediction = session.run(['output'], onnx_inputs)[0]

            # 如果预测结果带 batch 维度 (1, 3)，则压平
            pred_coords = prediction.flatten()

            # 5. 计算欧氏距离
            distance = np.linalg.norm(pred_coords - label_coords)
            all_distances.append(distance)

        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # 6. 计算总平均值
    if not all_distances:
        return "没有成功处理任何文件，无法计算平均距离。"

    avg_distance = np.mean(all_distances)
    return avg_distance


# --- 使用示例 ---
model_path = "./models\ImprovedTransformerModel\ImprovedTransformerModel_20251225_212910.onnx"
data_dir = "../data/data_1225_test_ext5"
result = batch_onnx_inference_and_eval(model_path, data_dir)
print(f"所有文件的平均欧氏距离为: {result}")


# 模拟输入数据（支持 dynamic_axes 定义的动态尺寸）
# batch_size = 1
# seq_len = 50
# dummy_seq_np = np.random.randn(batch_size, seq_len, 63).astype(np.float32)
# dummy_mask_np = np.ones((batch_size, seq_len), dtype=bool)
#
# # 执行
# result = inference_onnx(model_path, dummy_seq_np, dummy_mask_np)
#
# print(f"输出形状: {result.shape}")
# print(f"推理结果预览: \n{result}")