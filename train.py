# 训练: 数据加载 → 训练 MLP
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from util import read_csv
from dataset import prepare_data
from model import MLP

# 英文key → 中文列名映射
TARGET_MAP = {
    'ttfb': '首帧的平均',
    'stall_count': 'pwc卡顿数',
    'stall_rate': 'pwc卡顿率',
}
TARGETS = list(TARGET_MAP.keys())

MLP_CONFIG = {
    'epochs': 100,
    'lr': 1e-3,
    'batch_size': 64,
    'patience': 15,
    'weight_decay': 1e-4,
    'hidden_dims': [128, 64],
    'dropout': 0.2,
}


def train(model, train_set, val_set, config, save_path=None, scaler_X=None, scaler_y=None):
    train_dl = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True)
    val_dl = DataLoader(val_set, batch_size=config['batch_size'], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()
    device = next(model.parameters()).device

    best_val, best_state, patience_cnt = float('inf'), None, 0
    train_losses, val_losses = [], []

    for epoch in range(1, config['epochs'] + 1):
        # train
        model.train()
        t_loss, n = 0.0, 0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            t_loss += loss.item()
            n += 1
        t_loss /= n

        # val
        model.eval()
        v_loss, n = 0.0, 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                v_loss += criterion(model(X), y).item()
                n += 1
        v_loss /= n

        scheduler.step(v_loss)
        train_losses.append(t_loss)
        val_losses.append(v_loss)

        print(f"  Epoch {epoch:3d}/{config['epochs']} | Train: {t_loss:.6f} | Val: {v_loss:.6f}")

        # early stop
        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config['patience']:
                print(f"  Early stop @ Epoch {epoch}")
                break

    model.load_state_dict(best_state)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val,
            'config': config,
            'scaler_X': scaler_X,
            'scaler_y': scaler_y,
        }, save_path)

    return model, train_losses, val_losses


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default='./checkpoints')
    args = parser.parse_args()

    targets = [args.target] if args.target else list(TARGET_MAP.keys())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    df = read_csv(args.data)
    exclude = set(TARGETS)
    feature_cols = [c for c in df.select_dtypes(include='number').columns if c not in exclude]

    for target in targets:
        col = TARGET_MAP.get(target)
        if not col or col not in df.columns:
            continue

        print(f"\n{'=' * 50}")
        print(f"Target: {target} ({col})")
        print(f"{'=' * 50}")

        train_set, val_set, test_set, input_dim, scaler_X, scaler_y = prepare_data(df, feature_cols, col)
        model = MLP(input_dim, MLP_CONFIG['hidden_dims'], MLP_CONFIG['dropout']).to(device)

        save_path = os.path.join(args.save_dir, f'mlp_{target}.pt')
        model, _, _ = train(model, train_set, val_set, MLP_CONFIG, save_path=save_path, scaler_X=scaler_X, scaler_y=scaler_y)
