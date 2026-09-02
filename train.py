import argparse
from asyncio import run
from gradio import server
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from data import generate_sine_dataset, TimeSeriesDataset, load_csv_time_series
from models import LSTMForecaster, CNN1DForecaster, StateSpaceForecaster, TransformerForecaster


def get_model(name, seq_len=50, **kwargs):
    if name == 'lstm':
        return LSTMForecaster(input_size=kwargs.get('input_size', 1),
                               hidden_size=kwargs.get('hidden_size', 64),
                               num_layers=kwargs.get('num_layers', 2),
                               dropout=kwargs.get('dropout', 0.1))
    if name == 'cnn':
        return CNN1DForecaster(input_channels=kwargs.get('input_channels', 1),
                                channels=kwargs.get('channels', [16, 32]),
                                kernel_size=kwargs.get('kernel_size', 3),
                                seq_len=seq_len)
    if name == 'ssm':
        return StateSpaceForecaster(state_dim=kwargs.get('state_dim', 32),
                                    input_dim=kwargs.get('input_dim', 1))
    if name == 'transformer':
        return TransformerForecaster(input_dim=kwargs.get('input_dim', 1),
                                      d_model=kwargs.get('d_model', 64),
                                      nhead=kwargs.get('nhead', 4),
                                      num_layers=kwargs.get('num_layers', 3),
                                      dim_feedforward=kwargs.get('dim_feedforward', 128),
                                      dropout=kwargs.get('dropout', 0.1))
    raise ValueError('unknown model')


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_path:
        # expect comma-separated feature column names if provided via CLI
        feat_cols = args.feature_cols.split(',') if args.feature_cols else None
        if feat_cols is None:
            raise ValueError('When using --data-path you must pass --feature-cols (comma-separated) and --target-col')
        X, y = load_csv_time_series(args.data_path, feature_cols=feat_cols, target_col=args.target_col, seq_len=args.seq_len)
    else:
        X, y = generate_sine_dataset(n_series=2000, seq_len=args.seq_len, noise=args.noise)
    ds = TimeSeriesDataset(X, y)
    n_val = int(len(ds) * 0.1)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = get_model(args.model, input_size=1, input_dim=1, seq_len=args.seq_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        train_loss = total / n_train

        model.eval()
        total = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb).squeeze(-1)
                loss = loss_fn(pred, yb)
                total += loss.item() * xb.size(0)
        val_loss = total / n_val

        print(f"Epoch {epoch:3d} — Train MSE: {train_loss:.6f}  Val MSE: {val_loss:.6f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['lstm', 'cnn', 'ssm', 'transformer'], default='lstm')
    p.add_argument('--seq-len', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--noise', type=float, default=0.05)
    p.add_argument('--data-path', type=str, default=None, help='Optional CSV file path or URL for a real dataset')
    p.add_argument('--feature-cols', type=str, default=None, help='Comma-separated feature column names (required with --data-path)')
    p.add_argument('--target-col', type=str, default=None, help='Target column name (required with --data-path)')
    args = p.parse_args()
    train(args)
    # Exits after training. Use --data-path plus --feature-cols and --target-col to train on real datasets.
    