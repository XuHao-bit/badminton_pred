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
        self.l1_criterion = torch.nn.L1Loss()

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
                seqs, lengths, masks, labels_xyz, labels_time = [b.to(self.device) for b in batch[:-1]]

                self.optimizer.zero_grad()
                pred_xyz, pred_time = self.model(seqs, masks)

                # loss_xyz = self.cal_xyz_loss(pred_xyz, labels_xyz)
                loss_xyz = self.criterion(pred_xyz, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                # loss_time = self.l1_criterion(pred_time, labels_time)
                loss = loss_xyz + self.lambda_time * loss_time

                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()

            avg_loss = train_loss / len(self.train_loader)
            res_str = f"Epoch {epoch+1}: [Train] Loss = {avg_loss:.4f}|"

            # validation
            avg_xyz_loss, avg_time_loss, avg_xyz_err, avg_xy_err, avg_time_err = self.evaluate()
            res_str += f"\t[Valid] Loss = {avg_xyz_loss+self.lambda_time*avg_time_loss:.4f}(XYZ:{avg_xyz_loss:.2f}; Time:{avg_time_loss:.2f})|"
            res_str += f"\tXYZ Err(L2 Distance): {avg_xyz_err:.4f}|\tXY Err: {avg_xy_err:.4f}|\tTime Err: {avg_time_err:.4f}"
            self.logger.info(res_str)

            # save best model
            # if avg_xyz_loss + self.lambda_time * avg_time_loss < best_loss:
            if avg_xyz_err < best_xyz_err:
                best_epoch = epoch
                best_loss = avg_xyz_loss
                best_xyz_err = avg_xyz_err
                best_time_err = avg_time_err
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

        self.logger.info(f"Training finished. Best epoch: {best_epoch}|\tBest [Valid] loss: {best_loss:.4f}|\tXYZ Err: {best_xyz_err:.4f}|\tTime Err: {best_time_err:.4f}")

    def evaluate(self):
        self.model.eval()
        xyz_loss, time_loss = 0.0, 0.0
        preds, labels = [], []
        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std

        with torch.no_grad():
            for batch in self.test_loader:
                seqs, lengths, masks, labels_xyz, labels_time = [b.to(self.device) for b in batch[:-1]]
                pred_xyz, pred_time = self.model(seqs, masks)

                loss_xyz = self.criterion(pred_xyz, labels_xyz)
                loss_time = self.criterion(pred_time, labels_time)
                xyz_loss += loss_xyz.item()
                time_loss += loss_time.item()
                preds.append(torch.cat([pred_xyz, pred_time.unsqueeze(1)], dim=-1).cpu().numpy())
                labels.append(torch.cat([labels_xyz, labels_time.unsqueeze(1)], dim=-1).cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        # ===== 反归一化 =====
        preds = preds * label_std + label_mean
        labels = labels * label_std + label_mean
        xyz_preds, time_preds = preds[:, :3], preds[:, -1]
        xyz_labels, time_labels = labels[:, :3], labels[:, -1]

        avg_xyz_loss = xyz_loss / len(self.test_loader)
        avg_time_loss = time_loss / len(self.test_loader)
        avg_xyz_dist = np.mean(np.linalg.norm(xyz_preds - xyz_labels, axis=1))
        avg_xy_dist = np.mean(np.linalg.norm(xyz_preds[:, :2] - xyz_labels[:, :2], axis=1))
        avg_time_err = np.mean(np.abs(time_preds - time_labels))

        return avg_xyz_loss, avg_time_loss, avg_xyz_dist, avg_xy_dist, avg_time_err

    def test_and_save(self, save_dir="./results"):
        self.model.eval()
        preds, labels, filenames = [], [], []

        label_mean = self.test_loader.dataset.label_mean
        label_std = self.test_loader.dataset.label_std
        with torch.no_grad():
            for batch in self.test_loader:
                seqs, lengths, masks, labels_xyz, labels_time = [b.to(self.device) for b in batch[:-1]]
                fn = batch[-1]
                best_param = torch.load(self.best_model_path)
                self.model.load_state_dict(best_param)
                pred_xyz, pred_time = self.model(seqs, masks)

                preds.append(torch.cat([pred_xyz, pred_time.unsqueeze(1)], dim=-1).cpu().numpy())
                labels.append(torch.cat([labels_xyz, labels_time.unsqueeze(1)], dim=-1).cpu().numpy())
                filenames.append(fn)

        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0)
        filenames = [fn for fnames in filenames for fn in fnames]
        # ===== 反归一化 =====
        preds = preds * label_std + label_mean
        labels = labels * label_std + label_mean
        
        df = pd.DataFrame({
            "pred_x": preds[:, 0],
            "pred_y": preds[:, 1],
            "pred_z": preds[:, 2],
            "pred_time": preds[:, 3],
            "label_x": labels[:, 0],
            "label_y": labels[:, 1],
            "label_z": labels[:, 2],
            "label_time": labels[:, 3],
            "file_name": filenames,
        })

        # 拼接文件名
        os.makedirs(save_dir, exist_ok=True)
        filename = f"{self.logger.time}.csv"
        csv_file = os.path.join(save_dir, filename)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"📄 Saved test results to {csv_file}")
        return df
