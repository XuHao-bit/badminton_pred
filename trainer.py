# trainer.py
import os
import torch
import random
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
from dataset import collate_fn_dynamic  # 你的动态 padding collate_fn


def nll_loss(pred_mu, pred_log_var, target, lambda_nll=1.0):
    # 钳位 log_var 以避免数值不稳定 (可选，但推荐)
    pred_log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)

    # 方差 sigma^2
    pred_var = torch.exp(pred_log_var)

    # 损失项 1: log(sigma^2)
    loss_term_1 = lambda_nll * pred_log_var

    # 损失项 2: (y - mu)^2 / sigma^2 (带权重的 MSE)
    loss_term_2 = torch.square(target - pred_mu) / pred_var

    # 整体损失
    loss = 0.5 * (loss_term_1 + loss_term_2)

    return loss.mean()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, args, logger, model, train_dataset, test_dataset, device=None, batch_size=32, lr=1e-3, seed=42, save_dir="models"):
        set_seed(seed)
        self.logger = logger
        self.args = args
        self.lambda_time = args.lambda_time
        self.lambda_direction = args.lambda_direction
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")

        self.model = model.to(self.device)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                       collate_fn=lambda batch: collate_fn_dynamic(batch, max_len=args.max_len), num_workers=0)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                      collate_fn=lambda batch: collate_fn_dynamic(batch, max_len=args.max_len), num_workers=0)

        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = torch.nn.MSELoss()
        self.criterion_cos = torch.nn.CosineEmbeddingLoss()
        self.l1_criterion = torch.nn.L1Loss()
        self.regression_criterion_xyz = nll_loss

        self.save_dir = save_dir
        self.best_model_path = os.path.join(self.save_dir, f"{self.model.name}_{self.logger.time}.pt")
        os.makedirs(save_dir, exist_ok=True)

    def cal_xyz_loss(self, pred_xyz, labels_xyz):
        # 1. 计算元素级的 error（向量/张量）
        error = pred_xyz - labels_xyz  # 形状与 y_pred 相同，如 [batch_size, dim]
        
        # 2. 元素级判断：|error| ≤ delta → 掩码为 True，否则为 False
        # abs(error) 是元素级绝对值，self.delta 会广播到与 error 同形状
        mask = torch.abs(error) <= self.args.delta
        
        # 3. 元素级计算分段损失
        # 掩码为 True 的位置：用 MSE 项；False 的位置：用 MAE 项
        loss = torch.where(
            mask,
            0.5 * torch.square(error),  # MSE 分支（元素级平方）
            self.args.delta * (torch.abs(error) - 0.5 * self.args.delta)  # MAE 分支
        )
        return loss.mean()
        # return self.criterion(pred_xyz, labels_xyz)

    def train(self, num_epochs=50):
        best_epoch, best_loss, best_xyz_err, best_time_err = 0, float("inf"), float("inf"), float("inf")

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            # xyz_alpha = (5 - epoch*0.5//10) if epoch <= 30 else 2
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in batch[:-1]]

                self.optimizer.zero_grad()
                pred_xyz, pred_xyz_var, pred_time, pred_direction = self.model(seqs, masks)

                # loss_xyz = self.cal_xyz_loss(pred_xyz, labels_xyz)
                # loss_xyz = self.criterion(pred_xyz, labels_xyz)
                loss_xyz = self.regression_criterion_xyz(pred_xyz, pred_xyz_var, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                # loss_time = self.l1_criterion(pred_time, labels_time)
                targets = torch.ones(labels_direction.size(0), device=self.device)
                loss_direction = self.criterion_cos(pred_direction, labels_direction, targets)
                loss = loss_xyz + self.lambda_time * loss_time + self.lambda_direction * loss_direction

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(self.train_loader)
            res_str = f"Epoch {epoch+1}: [Train] Loss = {avg_loss:.4f}|"

            # validation
            avg_xyz_loss, avg_time_loss, avg_dir_loss, avg_xyz_err, avg_xy_err, avg_time_err, avg_direction_err = self.evaluate()
            res_str += f"\t[Valid] XYZ Loss = {avg_xyz_loss:.4f}| Time Loss = {avg_time_loss:.4f}| Dir Loss = {avg_dir_loss:.4f}|"
            res_str += f"\tXYZ Err(L2 Distance): {avg_xyz_err:.4f}|\tXY Err: {avg_xy_err:.4f}|\tTime Err: {avg_time_err:.4f}|\tDirection Err: {avg_direction_err:.4f}"
            self.logger.info(res_str)

            # save best model
            # if avg_xyz_loss + self.lambda_time * avg_time_loss < best_loss:
            if avg_xyz_err < best_xyz_err:
                best_epoch = epoch
                best_loss = avg_xyz_loss
                best_xyz_err = avg_xyz_err
                best_time_err = avg_time_err
                best_direction_err = avg_direction_err
                torch.save(self.model.state_dict(), self.best_model_path)

                # save model as onnx
                # self.model.to('cpu')
                # dummy_seq = torch.randn(1, 50, 63)
                # dummy_mask = torch.ones(1, 50, dtype=torch.bool)
                # torch.onnx.export(
                #     self.model,
                #     (dummy_seq, dummy_mask),
                #     self.best_model_path.replace('.pt', '.onnx'),
                #     input_names=['seq', 'mask'],
                #     output_names=['output'],
                #     dynamic_axes={"seq": {0: "batch_size", 1: "seq_len", 2: "feature_dim"}, "mask": {0: "batch_size", 1: "seq_len"}}
                # )
                # self.model.to(self.device)

                self.logger.info(f"✅ Saved best model to {self.best_model_path}")

        self.logger.info(
            f"Training finished. Best epoch: {best_epoch}|\tBest [Valid] loss: {best_loss:.4f}|\tXYZ Err: {best_xyz_err:.4f}|\tTime Err: {best_time_err:.4f}|\tDir Err: {best_direction_err:.2f}")
    def evaluate(self):
        self.model.eval()
        xyz_loss, time_loss, dir_loss = 0.0, 0.0, 0.0
        preds, labels = [], []
        dir_preds, dir_labels = [], []
        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std

        with torch.no_grad():
            for batch in self.test_loader:
                seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in batch[:-1]]
                pred_xyz, pred_xyz_var, pred_time, pred_direction = self.model(seqs, masks)

                # loss_xyz = self.criterion(pred_xyz, labels_xyz)
                loss_xyz = self.regression_criterion_xyz(pred_xyz, pred_xyz_var, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                targets = torch.ones(labels_direction.size(0), device=self.device)
                loss_dir = self.criterion_cos(pred_direction, labels_direction, targets)

                xyz_loss += loss_xyz.item()
                time_loss += loss_time.item()
                dir_loss += loss_dir.item()

                preds.append(torch.cat([pred_xyz, pred_time.unsqueeze(1)], dim=-1).cpu().numpy())
                labels.append(torch.cat([labels_xyz, labels_time.unsqueeze(1)], dim=-1).cpu().numpy())
                dir_preds.append(pred_direction.cpu().numpy())
                dir_labels.append(labels_direction.cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        dir_preds = np.concatenate(dir_preds, axis=0)
        dir_labels = np.concatenate(dir_labels, axis=0)
        # ===== 反归一化 =====
        preds = preds * label_std + label_mean
        labels = labels * label_std + label_mean
        xyz_preds, time_preds = preds[:, :3], preds[:, -1]
        xyz_labels, time_labels = labels[:, :3], labels[:, -1]

        avg_xyz_loss = xyz_loss / len(self.test_loader)
        avg_time_loss = time_loss / len(self.test_loader)
        avg_dir_loss = dir_loss / len(self.test_loader)
        avg_xyz_dist = np.mean(np.linalg.norm(xyz_preds - xyz_labels, axis=1))
        avg_xy_dist = np.mean(np.linalg.norm(xyz_preds[:, :2] - xyz_labels[:, :2], axis=1))
        avg_time_err = np.mean(np.abs(time_preds - time_labels))

        # === 方向误差（角度，单位：度） ===
        cos_sim = np.sum(dir_preds * dir_labels, axis=1) / (
                np.linalg.norm(dir_preds, axis=1) * np.linalg.norm(dir_labels, axis=1)
        )
        cos_sim = np.clip(cos_sim, -1.0, 1.0)  # 避免数值超出 [-1,1]
        avg_direction_err = np.mean(np.degrees(np.arccos(cos_sim)))

        return avg_xyz_loss, avg_time_loss, avg_dir_loss, avg_xyz_dist, avg_xy_dist, avg_time_err, avg_direction_err

    # def test_and_save(self, save_dir="./results"):
    #     self.model.eval()
    #     preds, labels, filenames = [], [], []
    # 
    #     label_mean = self.test_loader.dataset.label_mean
    #     label_std = self.test_loader.dataset.label_std
    #     with torch.no_grad():
    #         for batch in self.test_loader:
    #             seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [b.to(self.device) for b in
    #                                                                                batch[:-1]]
    #             fn = batch[-1]
    #             best_param = torch.load(self.best_model_path)
    #             self.model.load_state_dict(best_param)
    #             pred_xyz, pred_time, pred_direction = self.model(seqs, masks)
    # 
    #             preds.append(torch.cat(
    #                 [pred_xyz, pred_time.unsqueeze(1), pred_direction], dim=-1
    #             ).cpu().numpy())
    #             labels.append(torch.cat(
    #                 [labels_xyz, labels_time.unsqueeze(1), labels_direction], dim=-1
    #             ).cpu().numpy())
    #             filenames.append(fn)
    # 
    #     preds = np.concatenate(preds, axis=0)
    #     labels = np.concatenate(labels, axis=0)
    #     filenames = [fn for fnames in filenames for fn in fnames]
    #     # ===== 反归一化 (只对 xyz,time 进行，direction 是单位向量不需要归一化) =====
    #     preds[:, :4] = preds[:, :4] * label_std + label_mean
    #     labels[:, :4] = labels[:, :4] * label_std + label_mean
    # 
    #     df = pd.DataFrame({
    #         "pred_x": preds[:, 0],
    #         "pred_y": preds[:, 1],
    #         "pred_z": preds[:, 2],
    #         "pred_time": preds[:, 3],
    #         "pred_dir_x": preds[:, 4],
    #         "pred_dir_y": preds[:, 5],
    #         "label_x": labels[:, 0],
    #         "label_y": labels[:, 1],
    #         "label_z": labels[:, 2],
    #         "label_time": labels[:, 3],
    #         "label_dir_x": labels[:, 4],
    #         "label_dir_y": labels[:, 5],
    #         "file_name": filenames,
    #     })
    # 
    #     # 拼接文件名
    #     os.makedirs(save_dir, exist_ok=True)
    #     filename = f"{self.logger.time}.csv"
    #     csv_file = os.path.join(save_dir, filename)
    #     df.to_csv(csv_file, index=False)
    #     self.logger.info(f"📄 Saved test results to {csv_file}")
    #     return df

    def test_and_save(self, save_dir="./results", mc_samples=20):
        # 加载最佳模型参数
        best_param = torch.load(self.best_model_path)
        self.model.load_state_dict(best_param)
        self.model.to(self.device)

        # 启用 Dropout 层的训练模式以激活 MC Dropout
        self.model.train()

        # 用于存储 T 次 MC 采样的结果
        mc_mu_xyz_all = []  # 存储 T 次采样的 XYZ 均值 (mu)
        mc_log_var_xyz_all = []  # 存储 T 次采样的 XYZ Log-方差 (log(sigma^2_A))
        mc_time_all = []  # 存储 T 次采样的 Time 均值
        mc_dir_all = []  # 存储 T 次采样的 Direction 均值

        labels_list, filenames_list = [], []

        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std
        std_xyz = label_std[0][:3]
        std_xyz_sq = std_xyz ** 2  # 用于方差反归一化

        # 🎯 外层循环：MC 采样 T 次
        for mc_iter in tqdm(range(mc_samples), desc="MC Dropout Sampling (Total Uncertainty)"):
            mu_xyz_iter, log_var_xyz_iter, time_iter, dir_iter = [], [], [], []

            with torch.no_grad():
                for batch in self.test_loader:
                    seqs, lengths, masks, labels_xyz, labels_time, labels_direction = [
                        b.to(self.device) for b in batch[:-1]
                    ]

                    # 运行模型获取预测
                    output = self.model(seqs, masks)
                    pred_xyz_mu, pred_xyz_log_var, pred_time, pred_direction = output

                    # 收集本次采样的结果
                    mu_xyz_iter.append(pred_xyz_mu.cpu().numpy())
                    log_var_xyz_iter.append(pred_xyz_log_var.cpu().numpy())
                    time_iter.append(pred_time.cpu().numpy())
                    dir_iter.append(pred_direction.cpu().numpy())

                    # 仅在第一次 MC 采样时收集标签和文件名
                    if mc_iter == 0:
                        fn = batch[-1]
                        labels = torch.cat(
                            [labels_xyz, labels_time.unsqueeze(1), labels_direction], dim=-1
                        ).cpu().numpy()
                        labels_list.append(labels)
                        filenames_list.append(fn)

            # 将本轮 MC 采样的所有 batch 预测结果合并
            mc_mu_xyz_all.append(np.concatenate(mu_xyz_iter, axis=0))
            mc_log_var_xyz_all.append(np.concatenate(log_var_xyz_iter, axis=0))
            mc_time_all.append(np.concatenate(time_iter, axis=0))
            mc_dir_all.append(np.concatenate(dir_iter, axis=0))

        # 将 MC 样本的结果堆叠成张量: [MC_SAMPLES, Total_Samples, Features]
        mc_mu_xyz_tensor = np.stack(mc_mu_xyz_all, axis=0)
        mc_log_var_xyz_tensor = np.stack(mc_log_var_xyz_all, axis=0)
        mc_time_tensor = np.stack(mc_time_all, axis=0)
        mc_dir_tensor = np.stack(mc_dir_all, axis=0)

        labels = np.concatenate(labels_list, axis=0)
        filenames = [fn for fnames in filenames_list for fn in fnames]

        # ==========================================================
        # 1. 均值预测 (Final Prediction)
        # ==========================================================
        final_preds_mu_xyz = np.mean(mc_mu_xyz_tensor, axis=0)
        final_preds_mu_time = np.mean(mc_time_tensor, axis=0)
        final_preds_mu_dir = np.mean(mc_dir_tensor, axis=0)

        # 拼接 XYZT 均值
        final_preds_mu_xyzt = np.concatenate(
            [final_preds_mu_xyz, final_preds_mu_time[:, np.newaxis]], axis=1
        )

        # ==========================================================
        # 2. Epistemic Uncertainty (模型不确定性)
        # Var_E = Var(mu_t)
        # ==========================================================
        # 注意: 只需要对 XYZ (前3维) 计算 Epistemic
        var_epistemic_xyz_norm = np.var(mc_mu_xyz_tensor, axis=0)

        # 3. Aleatoric Uncertainty (数据不确定性)
        # Var_A = Mean(exp(log_var_t))
        # ==========================================================
        var_aleatoric_xyz_norm = np.mean(np.exp(mc_log_var_xyz_tensor), axis=0)

        # 4. Total Uncertainty (总体不确定性)
        # Var_T = Var_E + Var_A
        # ==========================================================
        var_total_xyz_norm = var_epistemic_xyz_norm + var_aleatoric_xyz_norm

        # ==========================================================
        # 5. 反归一化 (针对 XYZ)
        # ==========================================================
        # 均值反归一化 (前 4 列)
        final_preds_mu_xyzt_denorm = final_preds_mu_xyzt * label_std + label_mean
        labels_denorm = labels[:, :4] * label_std + label_mean

        # 方差反归一化: Var_denorm = Var_norm * std_label^2
        var_epistemic_xyz_denorm = var_epistemic_xyz_norm * std_xyz_sq
        var_aleatoric_xyz_denorm = var_aleatoric_xyz_norm * std_xyz_sq
        var_total_xyz_denorm = var_total_xyz_norm * std_xyz_sq

        # 转化为标准差 (Std = sqrt(Var))
        std_epistemic_xyz = np.sqrt(var_epistemic_xyz_denorm)
        std_aleatoric_xyz = np.sqrt(var_aleatoric_xyz_denorm)
        std_total_xyz = np.sqrt(var_total_xyz_denorm)

        # ==========================================================
        # 6. 保存到 DataFrame
        # ==========================================================

        df = pd.DataFrame({
            "pred_x": final_preds_mu_xyzt_denorm[:, 0],
            "pred_y": final_preds_mu_xyzt_denorm[:, 1],
            "pred_z": final_preds_mu_xyzt_denorm[:, 2],
            "pred_time": final_preds_mu_xyzt_denorm[:, 3],
            "pred_dir_x": final_preds_mu_dir[:, 0],
            "pred_dir_y": final_preds_mu_dir[:, 1],

            # 🎯 Epistemic Uncertainty (模型不确定性)
            "std_epistemic_x": std_epistemic_xyz[:, 0],
            "std_epistemic_y": std_epistemic_xyz[:, 1],
            "std_epistemic_z": std_epistemic_xyz[:, 2],

            # 🎯 Aleatoric Uncertainty (数据不确定性)
            "std_aleatoric_x": std_aleatoric_xyz[:, 0],
            "std_aleatoric_y": std_aleatoric_xyz[:, 1],
            "std_aleatoric_z": std_aleatoric_xyz[:, 2],

            # 🎯 Total Uncertainty (总体不确定性)
            "std_total_x": std_total_xyz[:, 0],
            "std_total_y": std_total_xyz[:, 1],
            "std_total_z": std_total_xyz[:, 2],

            "label_x": labels_denorm[:, 0],
            "label_y": labels_denorm[:, 1],
            "label_z": labels_denorm[:, 2],
            "label_time": labels_denorm[:, 3],
            "label_dir_x": labels[:, 4],
            "label_dir_y": labels[:, 5],
            "file_name": filenames,
        })

        # 拼接文件名
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.logger.time}_mc{mc_samples}.csv"
        csv_file = os.path.join(save_dir, filename)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"📄 Saved test results to {csv_file}")
        return df
    