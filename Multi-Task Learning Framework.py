# -*- coding: utf-8 -*-
"""
Environment-Adaptive Multi-Task Learning (MTL) Framework
For simultaneous prediction of crop traits (e.g., LAI and Yield) across multi-environment trials.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# Configuration & Hyperparameters (配置区)
# ==========================================
CONFIG = {
    'input_file': './data/dataset.xlsx',  # 请确保数据存放在 data 文件夹下
    'output_dir': './output/',  # 结果输出目录
    'lai_loss_weight': 0.5,  # LAI 损失权重 (User can tune this)
    'yield_loss_weight': 0.5,  # Yield 损失权重 (User can tune this)
    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'k_folds': 10,
    'random_seed': 42
}

# Auto-detect GPU computation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")


# ==========================================
# 1. Dataset Definition
# ==========================================
class MultiEnvDataset(Dataset):
    def __init__(self, X, Y_lai, Y_yield, env_labels, family_names):
        self.X = X
        self.Y_lai = Y_lai
        self.Y_yield = Y_yield
        self.env_labels = env_labels
        self.family_names = family_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y_lai[idx], self.Y_yield[idx], self.env_labels[idx], self.family_names[idx]


# ==========================================
# 2. Model Architecture
# ==========================================
class MultiTaskModel(nn.Module):
    """ Base Shared Network for Feature Extraction """

    def __init__(self, input_dim):
        super(MultiTaskModel, self).__init__()
        self.shared_layer1 = nn.Linear(input_dim, 256)
        self.shared_layer2 = nn.Linear(256, 128)
        self.shared_layer3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.lai_output = nn.Linear(64, 1)
        self.yield_output = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu(self.shared_layer1(x))
        x = self.relu(self.shared_layer2(x))
        x = self.relu(self.shared_layer3(x))
        lai_pred = self.lai_output(x)
        yield_pred = self.yield_output(x)
        return lai_pred, yield_pred


class MultiEnvMultiTaskModel(nn.Module):
    """ Environment-Specific Adaptation Routing """

    def __init__(self, base_model):
        super(MultiEnvMultiTaskModel, self).__init__()
        self.base_model = base_model

        # Environment 1 specific branches
        self.env1_layer = nn.Linear(64, 32)
        self.env1_lai_output = nn.Linear(32, 1)
        self.env1_yield_output = nn.Linear(32, 1)

        # Environment 2 specific branches
        self.env2_layer = nn.Linear(64, 32)
        self.env2_lai_output = nn.Linear(32, 1)
        self.env2_yield_output = nn.Linear(32, 1)

    def forward(self, x, env):
        # Extract shared features
        shared_features = self.base_model.relu(self.base_model.shared_layer3(
            self.base_model.relu(self.base_model.shared_layer2(
                self.base_model.relu(self.base_model.shared_layer1(x))))))

        lai_pred, yield_pred = self.base_model(x)

        mask_env1 = (env == 1)
        mask_env2 = (env == 2)

        # Route to Env 1 branch if present in batch
        if mask_env1.any():
            env1_features = self.base_model.relu(self.env1_layer(shared_features[mask_env1]))
            lai_pred[mask_env1] += self.env1_lai_output(env1_features)
            yield_pred[mask_env1] += self.env1_yield_output(env1_features)

        # Route to Env 2 branch if present in batch
        if mask_env2.any():
            env2_features = self.base_model.relu(self.env2_layer(shared_features[mask_env2]))
            lai_pred[mask_env2] += self.env2_lai_output(env2_features)
            yield_pred[mask_env2] += self.env2_yield_output(env2_features)

        return lai_pred, yield_pred


# ==========================================
# 3. Training Loop
# ==========================================
def train_multi_task_model(model, train_loader, lai_weight, yield_weight, epochs=50):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    model.to(device)

    for epoch in range(epochs):
        model.train()
        for x_batch, y_lai_batch, y_yield_batch, env_batch, _ in train_loader:
            x_batch, y_lai_batch, y_yield_batch, env_batch = \
                x_batch.to(device), y_lai_batch.to(device), y_yield_batch.to(device), env_batch.to(device)

            optimizer.zero_grad()
            lai_pred, yield_pred = model(x_batch, env_batch)

            loss_lai = criterion(lai_pred, y_lai_batch.unsqueeze(1))
            loss_yield = criterion(yield_pred, y_yield_batch.unsqueeze(1))

            # Weighted loss optimization
            total_loss = lai_weight * loss_lai + yield_weight * loss_yield
            total_loss.backward()
            optimizer.step()
    return model


# ==========================================
# 4. Main Execution
# ==========================================
def main():
    os.makedirs(CONFIG['output_dir'], exist_ok=True)
    print(f"Loading data from {CONFIG['input_file']}...")

    try:
        df = pd.read_excel(CONFIG['input_file'])
    except FileNotFoundError:
        print(f"Error: Dataset not found at {CONFIG['input_file']}. Please check the path.")
        return

    # Data imputation
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].mean())

    # Data Structure (Assumed: Col 0: Family, Col 1: Env, Col 2: Yield, Col 3: LAI, Col 4+: Features)
    family_names = df.iloc[:, 0].values
    env_labels = df.iloc[:, 1].values
    yield_data = df.iloc[:, 2].values
    lai_data = df.iloc[:, 3].values
    X = df.iloc[:, 4:].values

    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)

    X_scaled = torch.tensor(X_scaled, dtype=torch.float32)
    Y_lai = torch.tensor(lai_data, dtype=torch.float32)
    Y_yield = torch.tensor(yield_data, dtype=torch.float32)
    env_labels = torch.tensor(env_labels, dtype=torch.int64)

    kf = KFold(n_splits=CONFIG['k_folds'], shuffle=True, random_state=CONFIG['random_seed'])
    results, metrics = [], []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
        print(f"Processing Fold {fold + 1}/{CONFIG['k_folds']}...")

        train_dataset = MultiEnvDataset(X_scaled[train_idx], Y_lai[train_idx], Y_yield[train_idx],
                                        env_labels[train_idx], family_names[train_idx])
        test_dataset = MultiEnvDataset(X_scaled[test_idx], Y_lai[test_idx], Y_yield[test_idx], env_labels[test_idx],
                                       family_names[test_idx])

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

        # Initialize and train model
        base_model = MultiTaskModel(input_dim=X_scaled.shape[1])
        multi_env_model = MultiEnvMultiTaskModel(base_model)
        multi_env_model = train_multi_task_model(multi_env_model, train_loader, CONFIG['lai_loss_weight'],
                                                 CONFIG['yield_loss_weight'], epochs=CONFIG['epochs'])

        # Evaluation
        multi_env_model.eval()
        train_res, test_res = [], []
        t_lai_true, t_lai_pred, t_yield_true, t_yield_pred = [], [], [], []
        v_lai_true, v_lai_pred, v_yield_true, v_yield_pred = [], [], [], []

        with torch.no_grad():
            for x, y_lai, y_yield, env, fam in train_loader:
                x, env = x.to(device), env.to(device)
                l_pred, y_pred = multi_env_model(x, env)
                train_res.extend(zip(fam, y_lai.numpy(), l_pred.cpu().numpy().flatten(), y_yield.numpy(),
                                     y_pred.cpu().numpy().flatten()))
                t_lai_true.extend(y_lai.numpy())
                t_yield_true.extend(y_yield.numpy())
                t_lai_pred.extend(l_pred.cpu().numpy().flatten())
                t_yield_pred.extend(y_pred.cpu().numpy().flatten())

            for x, y_lai, y_yield, env, fam in test_loader:
                x, env = x.to(device), env.to(device)
                l_pred, y_pred = multi_env_model(x, env)
                test_res.extend(zip(fam, y_lai.numpy(), l_pred.cpu().numpy().flatten(), y_yield.numpy(),
                                    y_pred.cpu().numpy().flatten()))
                v_lai_true.extend(y_lai.numpy())
                v_yield_true.extend(y_yield.numpy())
                v_lai_pred.extend(l_pred.cpu().numpy().flatten())
                v_yield_pred.extend(y_pred.cpu().numpy().flatten())

        # Metrics Calculation
        metrics.append((
            fold + 1,
            r2_score(t_lai_true, t_lai_pred), np.sqrt(mean_squared_error(t_lai_true, t_lai_pred)),
            r2_score(t_yield_true, t_yield_pred), np.sqrt(mean_squared_error(t_yield_true, t_yield_pred)),
            r2_score(v_lai_true, v_lai_pred), np.sqrt(mean_squared_error(v_lai_true, v_lai_pred)),
            r2_score(v_yield_true, v_yield_pred), np.sqrt(mean_squared_error(v_yield_true, v_yield_pred))
        ))
        results.append((train_res, test_res))

    # Save Results
    results_path = os.path.join(CONFIG['output_dir'], 'multi_task_predictions.xlsx')
    metrics_path = os.path.join(CONFIG['output_dir'], 'multi_task_metrics.xlsx')

    with pd.ExcelWriter(results_path) as writer:
        for fold, (train_res, test_res) in enumerate(results):
            pd.DataFrame(train_res,
                         columns=['Family', 'True LAI', 'Predicted LAI', 'True Yield', 'Predicted Yield']).to_excel(
                writer, sheet_name=f'Fold_{fold + 1}_Train', index=False)
            pd.DataFrame(test_res,
                         columns=['Family', 'True LAI', 'Predicted LAI', 'True Yield', 'Predicted Yield']).to_excel(
                writer, sheet_name=f'Fold_{fold + 1}_Test', index=False)

    pd.DataFrame(metrics, columns=[
        'Fold', 'Train LAI R2', 'Train LAI RMSE', 'Train Yield R2', 'Train Yield RMSE',
        'Test LAI R2', 'Test LAI RMSE', 'Test Yield R2', 'Test Yield RMSE'
    ]).to_excel(metrics_path, index=False)

    print("Training complete! All results and metrics have been saved to the output directory.")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {str(e)}")