import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

TARGETS = ['PTS', 'REB', 'AST', 'FG_PCT']

class NBAPlayerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NBAMultiOutputModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NBAMultiOutputModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    def forward(self, x):
        return self.net(x)

def main():
    df1 = pd.read_parquet('parquet/final_database_2024-25.parquet')
    df2 = pd.read_parquet('parquet/final_database_2023-24.parquet')
    df3 = pd.read_parquet('parquet/final_database_2025-26.parquet')
    df1 = df1.dropna()
    df2 = df2.dropna()
    df3 = df3.dropna()

    place_df = df1.sort_values('GAME_DATE')
    place_df2 = df2.sort_values('GAME_DATE')
    place_df3 = df3.sort_values('GAME_DATE')
    split_index1 = int(len(place_df) * 0.6)

    train_df = pd.concat([place_df, place_df2, place_df3], ignore_index=True)
    test_df = place_df.iloc[split_index1:]

    features = ['MIN_last5', 'PTS_last5', 'REB_last5', 'AST_last5', 'FG_PCT_last5', 'USAGE_last5', 'IS_HOME', 'DAYS_REST', 'PLUS_MINUS_last5', 'offensiveRating_last5', 'defensiveRating_last5', 'pace_last5', 'OPP_offensiveRating_last5', 'OPP_defensiveRating_last5', 'OPP_pace_last5']

    X_train = train_df[features].values.astype(np.float32)
    y_train = train_df[TARGETS].values.astype(np.float32)

    X_test = test_df[features].values.astype(np.float32)
    y_test = test_df[TARGETS].values.astype(np.float32)

    train_ds = NBAPlayerDataset(X_train, y_train)
    test_ds = NBAPlayerDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)

    model = NBAMultiOutputModel(input_dim=len(features), output_dim=len(TARGETS))

    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 50
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}')

    model.eval()
    preds_list, true_list = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            preds_list.append(preds.numpy())
            true_list.append(y_batch.numpy())

    preds_arr = np.vstack(preds_list)
    true_arr = np.vstack(true_list)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Predicted vs Actual', fontsize=14)

    for i, (target, ax) in enumerate(zip(TARGETS, axes.flatten())):
        actual = true_arr[:, i]
        predicted = preds_arr[:, i]
        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        print(f"{target} — MAE: {mae:.4f}, R²: {r2:.4f}")

        lo, hi = min(actual.min(), predicted.min()), max(actual.max(), predicted.max())
        ax.scatter(actual, predicted, alpha=0.3, s=10)
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(f'{target}  |  MAE: {mae:.3f}  R²: {r2:.3f}')

    plt.tight_layout()
    plt.savefig('models/predicted_vs_actual.png', dpi=150)
    plt.show()

    torch.save(model, 'models/multi_output_model.pth')

    print_feature_importance(model, X_test, y_test, features)

def print_feature_importance(model, X_test, y_test, features):
    model.eval()
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        baseline_preds = model(X_tensor).numpy()
    baseline_mae = np.mean([
        mean_absolute_error(y_test[:, i], baseline_preds[:, i])
        for i in range(len(TARGETS))
    ])

    importance = {}
    rng = np.random.default_rng(42)
    for j, feature in enumerate(features):
        X_permuted = X_test.copy()
        X_permuted[:, j] = rng.permutation(X_permuted[:, j])
        with torch.no_grad():
            permuted_preds = model(torch.tensor(X_permuted, dtype=torch.float32)).numpy()
        permuted_mae = np.mean([
            mean_absolute_error(y_test[:, i], permuted_preds[:, i])
            for i in range(len(TARGETS))
        ])
        importance[feature] = permuted_mae - baseline_mae

    ranked = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature Importance (permutation, avg MAE increase across all targets):")
    for feature, delta in ranked:
        print(f"  {feature:<35} +{delta:.4f}")

if __name__ == "__main__":
    main()
