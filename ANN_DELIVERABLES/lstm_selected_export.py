"""Train and export the selected standard LSTM model.

This script keeps the LSTM setup used for the best validation simulation result:

    seq_len = 100, hidden_size = 64, num_layers = 1, batch_size = 64

The model selection and reported metrics are computed only on the chronological
training/validation split from training-val-test-data.npz. The hidden submission
templates are only used at the end to create files with the expected structure.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class LSTMConfig:
    seq_len: int = 100
    hidden_size: int = 64
    num_layers: int = 1
    dense_size: int = 32
    dropout: float = 0.0
    batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 20
    gradient_clip: float = 1.0
    seed: int = 7


class LSTMModel(nn.Module):
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.dense_size),
            nn.ReLU(),
            nn.Linear(config.dense_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def angle_error(theta_true: np.ndarray, theta_pred: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta_true - theta_pred), np.cos(theta_true - theta_pred))


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def make_lstm_dataset(u_norm: np.ndarray, theta: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    x_data: list[np.ndarray] = []
    y_data: list[np.ndarray] = []

    for k in range(seq_len, len(theta)):
        features = np.stack(
            [
                u_norm[k - seq_len : k],
                sin_theta[k - seq_len : k],
                cos_theta[k - seq_len : k],
            ],
            axis=1,
        )
        target = np.array([sin_theta[k], cos_theta[k]])
        x_data.append(features)
        y_data.append(target)

    return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)


def sincos_to_angle(y_pred: torch.Tensor) -> torch.Tensor:
    norm = torch.sqrt(y_pred[:, 0] ** 2 + y_pred[:, 1] ** 2).unsqueeze(1) + 1e-8
    y_unit = y_pred / norm
    return torch.atan2(y_unit[:, 0], y_unit[:, 1])


def evaluate_prediction_rmse(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            theta_pred = sincos_to_angle(model(x_batch))
            theta_true = torch.atan2(y_batch[:, 0], y_batch[:, 1])
            err = torch.atan2(torch.sin(theta_true - theta_pred), torch.cos(theta_true - theta_pred))
            errors.append(err.cpu().numpy())
    return rmse(np.concatenate(errors))


def predict_dataset_angles(model: nn.Module, x_data: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    x_tensor = torch.tensor(x_data, dtype=torch.float32)
    preds: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            x_batch = x_tensor[start : start + batch_size].to(device)
            theta_pred = sincos_to_angle(model(x_batch))
            preds.append(theta_pred.cpu().numpy())
    return np.concatenate(preds)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LSTMConfig,
    device: torch.device,
) -> tuple[nn.Module, dict[str, list[float] | float | int]]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-5,
    )

    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: dict[str, list[float] | float | int] = {
        "train_loss": [],
        "val_prediction_rmse": [],
        "best_epoch": 0,
        "best_val_prediction_rmse": math.inf,
    }

    for epoch in range(config.epochs):
        model.train()
        losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            loss = criterion(model(x_batch), y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            losses.append(float(loss.item()))

        train_loss = float(np.sqrt(np.mean(losses)))
        val_rmse = evaluate_prediction_rmse(model, val_loader, device)
        scheduler.step(val_rmse)

        history["train_loss"].append(train_loss)  # type: ignore[index]
        history["val_prediction_rmse"].append(val_rmse)  # type: ignore[index]

        if val_rmse < best_val:
            best_val = val_rmse
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_prediction_rmse"] = val_rmse
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"epoch={epoch:04d} train_loss={train_loss:.5f} "
                f"val_pred_rmse={val_rmse:.5f} lr={current_lr:.6f}"
            )

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history


def simulate_lstm(
    model: nn.Module,
    u: np.ndarray,
    theta_init: np.ndarray,
    u_mean: float,
    u_std: float,
    max_history: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    u_norm = (u - u_mean) / u_std
    theta_generated = list(theta_init.astype(float))

    with torch.no_grad():
        for k in range(len(theta_generated), len(u)):
            start = max(0, k - max_history)
            theta_hist = np.asarray(theta_generated[start:k], dtype=float)
            u_hist = u_norm[start:k]
            x = np.stack([u_hist, np.sin(theta_hist), np.cos(theta_hist)], axis=1)
            x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x_tensor).cpu().numpy()[0]
            pred_norm = np.sqrt(pred[0] ** 2 + pred[1] ** 2) + 1e-8
            theta_next = np.arctan2(pred[0] / pred_norm, pred[1] / pred_norm)
            theta_generated.append(float(theta_next))

    return np.asarray(theta_generated)


def predict_sequences(
    model: nn.Module,
    u_histories: np.ndarray,
    theta_histories: np.ndarray,
    u_mean: float,
    u_std: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    u_norm = (u_histories - u_mean) / u_std
    x = np.stack([u_norm, np.sin(theta_histories), np.cos(theta_histories)], axis=2)
    x_tensor = torch.tensor(x, dtype=torch.float32).to(device)

    preds: list[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, len(x_tensor), batch_size):
            y_pred = model(x_tensor[start : start + batch_size])
            theta_pred = sincos_to_angle(y_pred)
            preds.append(theta_pred.cpu().numpy())
    return np.concatenate(preds)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    config: LSTMConfig,
    u_mean: float,
    u_std: float,
    history: dict[str, list[float] | float | int],
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "u_mean": u_mean,
            "u_std": u_std,
            "history": history,
            "metrics": metrics,
        },
        path,
    )


def export_prediction_submission(
    model: nn.Module,
    template_path: Path,
    output_path: Path,
    u_mean: float,
    u_std: float,
    device: torch.device,
) -> None:
    data = np.load(template_path)
    upast = data["upast"]
    thpast = data["thpast"]
    thnow = predict_sequences(model, upast, thpast, u_mean, u_std, device)
    np.savez(output_path, upast=upast, thpast=thpast, thnow=thnow)
    print(f"Saved prediction submission: {output_path}")


def export_simulation_submission(
    model: nn.Module,
    template_path: Path,
    output_path: Path,
    u_mean: float,
    u_std: float,
    config: LSTMConfig,
    init_steps: int,
    device: torch.device,
) -> None:
    data = np.load(template_path)
    u = data["u"]
    theta_template = data["th"]
    theta_init = theta_template[:init_steps]
    theta_sim = simulate_lstm(model, u, theta_init, u_mean, u_std, config.seq_len, device)
    np.savez(output_path, u=u, th=theta_sim)
    print(f"Saved simulation submission: {output_path}")


def save_report_figures(
    figures_dir: Path,
    history: dict[str, list[float] | float | int],
    theta_val: np.ndarray,
    theta_val_pred: np.ndarray,
    theta_val_sim: np.ndarray,
    config: LSTMConfig,
) -> None:
    import os
    import tempfile

    plot_cache = Path(tempfile.gettempdir()) / "lstm_matplotlib_cache"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    train_loss = np.asarray(history["train_loss"], dtype=float)
    val_prediction_rmse = np.asarray(history["val_prediction_rmse"], dtype=float)
    epochs = np.arange(len(train_loss))

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_prediction_rmse, label="val prediction RMSE")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.title("LSTM training")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "LSTM_training_plot_lines.png", dpi=300)
    plt.close()

    pred_index = np.arange(config.seq_len, len(theta_val))
    theta_val_true_pred = theta_val[config.seq_len :]
    pred_residual = angle_error(theta_val_true_pred, theta_val_pred)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(pred_index, theta_val_true_pred, label="measured", linewidth=1.0)
    plt.plot(pred_index, theta_val_pred, label="LSTM one-step prediction", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("LSTM one-step validation prediction")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "LSTM_validation_prediction.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 2.8))
    plt.plot(pred_index, pred_residual, label="prediction residual", linewidth=1.0)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("error [rad]")
    plt.title("LSTM one-step prediction residual")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "LSTM_prediction_residual.png", dpi=300)
    plt.close()

    sim_index = np.arange(len(theta_val_sim))
    sim_residual = angle_error(theta_val[: len(theta_val_sim)], theta_val_sim)

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(sim_index, theta_val[: len(theta_val_sim)], label="measured", linewidth=1.0)
    plt.plot(sim_index, theta_val_sim, label="LSTM simulation", linewidth=1.0)
    plt.axvline(config.seq_len, color="black", linestyle="--", linewidth=1.0, label="end of init")
    plt.xlabel("k")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("LSTM full validation simulation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "LSTM_full_validation_simulation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 2.8))
    plt.plot(sim_index, sim_residual, label="simulation residual", linewidth=1.0)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    plt.axvline(config.seq_len, color="black", linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("error [rad]")
    plt.title("LSTM simulation residual")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "LSTM_simulation_residual.png", dpi=300)
    plt.close()

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True)
    axes[0].plot(sim_index, theta_val[: len(theta_val_sim)], label="measured", linewidth=1.0)
    axes[0].plot(sim_index, theta_val_sim, label="LSTM simulation", linewidth=1.0)
    axes[0].axvline(config.seq_len, color="black", linestyle="--", linewidth=1.0, label="end of init")
    axes[0].set_ylabel(r"$\theta$ [rad]")
    axes[0].set_title("LSTM full validation simulation")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(sim_index, sim_residual, label="simulation residual", linewidth=1.0)
    axes[1].axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    axes[1].axvline(config.seq_len, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("error [rad]")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(figures_dir / "LSTM_simulation_and_residual.png", dpi=300)
    plt.close(fig)

    print(f"Saved report figures in: {figures_dir}")


def evaluate_split(
    model: nn.Module,
    u_split: np.ndarray,
    theta_split: np.ndarray,
    u_mean: float,
    u_std: float,
    config: LSTMConfig,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    u_norm = (u_split - u_mean) / u_std
    x_split, y_split = make_lstm_dataset(u_norm, theta_split, config.seq_len)
    loader = DataLoader(
        TensorDataset(torch.tensor(x_split), torch.tensor(y_split)),
        batch_size=config.batch_size,
        shuffle=False,
    )
    prediction_rmse = evaluate_prediction_rmse(model, loader, device)
    theta_pred = predict_dataset_angles(model, x_split, device)
    theta_sim = simulate_lstm(
        model,
        u_split,
        theta_split[: config.seq_len],
        u_mean,
        u_std,
        config.seq_len,
        device,
    )
    sim_error_all = angle_error(theta_split[: len(theta_sim)], theta_sim)
    sim_error_after_init = angle_error(theta_split[config.seq_len :], theta_sim[config.seq_len :])
    return prediction_rmse, rmse(sim_error_all), rmse(sim_error_after_init), theta_pred, theta_sim


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Train/export the selected standard LSTM model.")
    parser.add_argument("--data", type=Path, default=script_dir / "training-val-test-data.npz")
    parser.add_argument("--prediction-template", type=Path, default=script_dir / "hidden-test-prediction-submission-file.npz")
    parser.add_argument("--simulation-template", type=Path, default=script_dir / "hidden-test-simulation-submission-file.npz")
    parser.add_argument("--checkpoint", type=Path, default=script_dir / "selected_lstm_best.pt")
    parser.add_argument("--prediction-output", type=Path, default=script_dir / "lstm_prediction_submission.npz")
    parser.add_argument("--simulation-output", type=Path, default=script_dir / "lstm_simulation_submission.npz")
    parser.add_argument("--metrics-output", type=Path, default=script_dir / "selected_lstm_metrics.json")
    parser.add_argument("--figures-dir", type=Path, default=script_dir / "Figures")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--no-export", action="store_true", help="Train/evaluate without writing submission files.")
    parser.add_argument("--no-figures", action="store_true", help="Train/evaluate without saving report figures.")
    parser.add_argument("--simulation-init-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=LSTMConfig.epochs)
    parser.add_argument("--seed", type=int, default=LSTMConfig.seed)
    args = parser.parse_args()

    config = LSTMConfig(epochs=args.epochs, seed=args.seed)
    set_seed(config.seed)
    device = select_device(args.device)
    print(f"Using device: {device}")
    print(f"Config: {config}")

    out = np.load(args.data)
    theta = out["th"]
    u = out["u"]

    n_samples = len(u)
    n_train = int(0.70 * n_samples)
    n_val = int(0.15 * n_samples)

    u_train = u[:n_train]
    theta_train = theta[:n_train]
    u_val = u[n_train : n_train + n_val]
    theta_val = theta[n_train : n_train + n_val]
    u_test = u[n_train + n_val :]
    theta_test = theta[n_train + n_val :]

    u_mean = float(u_train.mean())
    u_std = float(u_train.std())
    u_train_norm = (u_train - u_mean) / u_std
    u_val_norm = (u_val - u_mean) / u_std

    x_train, y_train = make_lstm_dataset(u_train_norm, theta_train, config.seq_len)
    x_val, y_val = make_lstm_dataset(u_val_norm, theta_val, config.seq_len)

    train_loader = DataLoader(
        TensorDataset(torch.tensor(x_train), torch.tensor(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(x_val), torch.tensor(y_val)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = LSTMModel(config)
    model, history = train_model(model, train_loader, val_loader, config, device)

    val_prediction_rmse = evaluate_prediction_rmse(model, val_loader, device)
    theta_val_pred = predict_dataset_angles(model, x_val, device)
    theta_val_sim = simulate_lstm(
        model,
        u_val,
        theta_val[: config.seq_len],
        u_mean,
        u_std,
        config.seq_len,
        device,
    )
    sim_error_all = angle_error(theta_val[: len(theta_val_sim)], theta_val_sim)
    sim_error_after_init = angle_error(theta_val[config.seq_len :], theta_val_sim[config.seq_len :])
    test_prediction_rmse, test_sim_rmse_all, test_sim_rmse_after_init, _, _ = evaluate_split(
        model,
        u_test,
        theta_test,
        u_mean,
        u_std,
        config,
        device,
    )

    metrics = {
        "validation_prediction_rmse_rad": val_prediction_rmse,
        "validation_prediction_rmse_deg": val_prediction_rmse * 180.0 / np.pi,
        "validation_simulation_rmse_rad_including_init": rmse(sim_error_all),
        "validation_simulation_rmse_deg_including_init": rmse(sim_error_all) * 180.0 / np.pi,
        "validation_simulation_rmse_rad_after_init": rmse(sim_error_after_init),
        "validation_simulation_rmse_deg_after_init": rmse(sim_error_after_init) * 180.0 / np.pi,
        "test_prediction_rmse_rad": test_prediction_rmse,
        "test_prediction_rmse_deg": test_prediction_rmse * 180.0 / np.pi,
        "test_simulation_rmse_rad_including_init": test_sim_rmse_all,
        "test_simulation_rmse_deg_including_init": test_sim_rmse_all * 180.0 / np.pi,
        "test_simulation_rmse_rad_after_init": test_sim_rmse_after_init,
        "test_simulation_rmse_deg_after_init": test_sim_rmse_after_init * 180.0 / np.pi,
    }

    print("\n--- Validation and held-out test metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")

    save_checkpoint(args.checkpoint, model, config, u_mean, u_std, history, metrics)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Saved metrics: {args.metrics_output}")

    if not args.no_figures:
        save_report_figures(
            args.figures_dir,
            history,
            theta_val,
            theta_val_pred,
            theta_val_sim,
            config,
        )

    if not args.no_export:
        export_prediction_submission(
            model,
            args.prediction_template,
            args.prediction_output,
            u_mean,
            u_std,
            device,
        )
        export_simulation_submission(
            model,
            args.simulation_template,
            args.simulation_output,
            u_mean,
            u_std,
            config,
            args.simulation_init_steps,
            device,
        )


if __name__ == "__main__":
    main()
