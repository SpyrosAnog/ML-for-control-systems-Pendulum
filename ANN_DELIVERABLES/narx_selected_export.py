"""Train, evaluate, plot, and export the selected NARX model.

This script keeps the selected NARX setup used in the report:

    n_a = 6, n_b = 2, hidden_size = 32, activation = ReLU

The model selection and reported metrics are computed on the chronological
training/validation/test split from training-val-test-data.npz. The hidden
submission templates are only used at the end to create files with the
expected structure.
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


@dataclass
class NARXConfig:
    na: int = 6
    nb: int = 2
    hidden_size: int = 32
    epochs: int = 6000
    learning_rate: float = 1e-3
    patience: int = 100
    seed: int = 7

    @property
    def n0(self) -> int:
        return max(self.na, self.nb)


class NARXNet(nn.Module):
    def __init__(self, config: NARXConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.na + config.nb, config.hidden_size)
        self.fc2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.fc3 = nn.Linear(config.hidden_size, 1)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        return self.fc3(out)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values**2)))


def degrees(rad: float) -> float:
    return rad * 180.0 / np.pi


def create_io_data(u_norm: np.ndarray, theta_norm: np.ndarray, config: NARXConfig) -> tuple[np.ndarray, np.ndarray]:
    x_data: list[np.ndarray] = []
    y_data: list[float] = []
    for k in range(config.n0, len(theta_norm)):
        x_data.append(np.concatenate([u_norm[k - config.nb : k], theta_norm[k - config.na : k]]))
        y_data.append(float(theta_norm[k]))
    return np.asarray(x_data, dtype=np.float32), np.asarray(y_data, dtype=np.float32)


def train_model(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    theta_std: float,
    config: NARXConfig,
) -> tuple[nn.Module, dict[str, list[float] | float | int]]:
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    x_val_t = torch.tensor(x_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()

    best_val = math.inf
    best_state = copy.deepcopy(model.state_dict())
    patience_counter = 0
    history: dict[str, list[float] | float | int] = {
        "train_rmse_rad": [],
        "val_rmse_rad": [],
        "best_epoch": 0,
        "best_val_rmse_rad": math.inf,
    }

    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        train_pred = model(x_train_t).squeeze()
        train_loss = torch.sqrt(criterion(train_pred, y_train_t))
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(x_val_t).squeeze()
            val_loss = torch.sqrt(criterion(val_pred, y_val_t)).item()

        train_rmse_rad = float(train_loss.item() * theta_std)
        val_rmse_rad = float(val_loss * theta_std)
        history["train_rmse_rad"].append(train_rmse_rad)  # type: ignore[index]
        history["val_rmse_rad"].append(val_rmse_rad)  # type: ignore[index]

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            history["best_epoch"] = epoch
            history["best_val_rmse_rad"] = val_rmse_rad
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 100 == 0:
            print(f"epoch={epoch:04d} train_rmse={train_rmse_rad:.5f} val_rmse={val_rmse_rad:.5f}")

        if patience_counter >= config.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history


def one_step_prediction(
    model: nn.Module,
    u_norm: np.ndarray,
    theta_norm: np.ndarray,
    theta_mean: float,
    theta_std: float,
    config: NARXConfig,
) -> tuple[np.ndarray, np.ndarray]:
    x_data, _ = create_io_data(u_norm, theta_norm, config)
    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(x_data, dtype=torch.float32)).squeeze().numpy()
    theta_pred = y_pred_norm * theta_std + theta_mean
    theta_true = theta_norm[config.n0 :] * theta_std + theta_mean
    return theta_pred, theta_true


def free_run_simulation(
    model: nn.Module,
    u_norm: np.ndarray,
    theta_norm: np.ndarray,
    theta_mean: float,
    theta_std: float,
    config: NARXConfig,
    init_steps: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    init = config.n0 if init_steps is None else max(init_steps, config.n0)
    u_t = torch.tensor(u_norm, dtype=torch.float32)
    generated = [torch.tensor(theta_norm[i], dtype=torch.float32) for i in range(init)]

    with torch.no_grad():
        for k in range(init, len(theta_norm)):
            u_lags = u_t[k - config.nb : k].reshape(1, -1)
            y_lags = torch.stack(generated[k - config.na : k]).reshape(1, -1)
            x = torch.cat([u_lags, y_lags], dim=1)
            generated.append(model(x).squeeze())

    theta_pred = torch.stack(generated).numpy() * theta_std + theta_mean
    theta_true = theta_norm * theta_std + theta_mean
    return theta_pred, theta_true


def split_metrics(
    model: nn.Module,
    u_split: np.ndarray,
    theta_split: np.ndarray,
    u_mean: float,
    u_std: float,
    theta_mean: float,
    theta_std: float,
    config: NARXConfig,
    split_name: str,
) -> dict[str, float]:
    u_norm = (u_split - u_mean) / u_std
    theta_norm = (theta_split - theta_mean) / theta_std

    theta_pred, theta_true_pred = one_step_prediction(model, u_norm, theta_norm, theta_mean, theta_std, config)
    pred_error = theta_pred - theta_true_pred

    theta_sim, theta_true_sim = free_run_simulation(model, u_norm, theta_norm, theta_mean, theta_std, config)
    sim_error = theta_sim[config.n0 :] - theta_true_sim[config.n0 :]

    return {
        f"{split_name}_prediction_rmse_rad": rmse(pred_error),
        f"{split_name}_prediction_rmse_deg": degrees(rmse(pred_error)),
        f"{split_name}_simulation_rmse_rad": rmse(sim_error),
        f"{split_name}_simulation_rmse_deg": degrees(rmse(sim_error)),
    }


def save_report_figures(
    figures_dir: Path,
    history: dict[str, list[float] | float | int],
    theta_val: np.ndarray,
    theta_val_pred: np.ndarray,
    theta_val_true_pred: np.ndarray,
    theta_val_sim: np.ndarray,
    theta_val_true_sim: np.ndarray,
    config: NARXConfig,
) -> None:
    import os
    import tempfile

    plot_cache = Path(tempfile.gettempdir()) / "narx_matplotlib_cache"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)

    train_loss = np.asarray(history["train_rmse_rad"], dtype=float)
    val_loss = np.asarray(history["val_rmse_rad"], dtype=float)
    epochs = np.arange(len(train_loss))

    plt.figure(figsize=(6.4, 4.0))
    plt.plot(epochs, train_loss, label="train loss")
    plt.plot(epochs, val_loss, label="val loss")
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("RMSE [rad]")
    plt.title("NARX training")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "NARX_training_plot_lines.png", dpi=300)
    plt.close()

    pred_residual = theta_val_pred - theta_val_true_pred
    sim_residual = theta_val_sim - theta_val_true_sim

    plt.figure(figsize=(7.2, 4.2))
    plt.plot(theta_val_true_sim, label="measured", linewidth=1.0)
    plt.plot(theta_val_sim, label="NARX simulation", linewidth=1.0)
    plt.axvline(config.n0, color="black", linestyle="--", linewidth=1.0, label="end of init")
    plt.xlabel("k")
    plt.ylabel(r"$\theta$ [rad]")
    plt.title("NARX full validation simulation")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Narx_full_validation_simulation.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 2.8))
    plt.plot(sim_residual, label="simulation residual", linewidth=1.0)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("error [rad]")
    plt.title("NARX simulation residual")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "NARX_simulation_resildual.png", dpi=300)
    plt.close()

    plt.figure(figsize=(7.2, 2.8))
    plt.plot(np.arange(config.n0, len(theta_val)), pred_residual, label="prediction residual", linewidth=1.0)
    plt.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    plt.xlabel("k")
    plt.ylabel("error [rad]")
    plt.title("NARX one-step prediction residual")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "NARX_prediction_residual.png", dpi=300)
    plt.close()

    print(f"Saved report figures in: {figures_dir}")


def export_prediction_submission(
    model: nn.Module,
    template_path: Path,
    output_path: Path,
    u_mean: float,
    u_std: float,
    theta_mean: float,
    theta_std: float,
    config: NARXConfig,
) -> None:
    data = np.load(template_path)
    upast = data["upast"]
    thpast = data["thpast"]

    upast_norm = (upast - u_mean) / u_std
    thpast_norm = (thpast - theta_mean) / theta_std
    x_test = np.concatenate([upast_norm[:, -config.nb :], thpast_norm[:, -config.na :]], axis=1)

    model.eval()
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(x_test, dtype=torch.float32)).squeeze().numpy()
    thnow = y_pred_norm * theta_std + theta_mean

    np.savez(output_path, upast=upast, thpast=thpast, thnow=thnow)
    print(f"Saved prediction submission: {output_path}")


def export_simulation_submission(
    model: nn.Module,
    template_path: Path,
    output_path: Path,
    u_mean: float,
    u_std: float,
    theta_mean: float,
    theta_std: float,
    config: NARXConfig,
    init_steps: int,
) -> None:
    data = np.load(template_path)
    u = data["u"]
    theta_template = data["th"]
    u_norm = (u - u_mean) / u_std
    theta_norm = (theta_template - theta_mean) / theta_std
    theta_sim, _ = free_run_simulation(
        model,
        u_norm,
        theta_norm,
        theta_mean,
        theta_std,
        config,
        init_steps=init_steps,
    )
    np.savez(output_path, u=u, th=theta_sim)
    print(f"Saved simulation submission: {output_path}")


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train/evaluate/export the selected NARX model.")
    parser.add_argument("--data", type=Path, default=script_dir / "training-val-test-data.npz")
    parser.add_argument("--checkpoint", type=Path, default=script_dir / "selected_narx_best.pt")
    parser.add_argument("--metrics-output", type=Path, default=script_dir / "selected_narx_metrics.json")
    parser.add_argument("--figures-dir", type=Path, default=script_dir / "Figures")
    parser.add_argument("--prediction-template", type=Path, default=script_dir / "hidden-test-prediction-submission-file.npz")
    parser.add_argument("--simulation-template", type=Path, default=script_dir / "hidden-test-simulation-submission-file.npz")
    parser.add_argument("--prediction-output", type=Path, default=script_dir / "narx_prediction_submission.npz")
    parser.add_argument("--simulation-output", type=Path, default=script_dir / "narx_simulation_submission.npz")
    parser.add_argument("--simulation-init-steps", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=NARXConfig.epochs)
    parser.add_argument("--seed", type=int, default=NARXConfig.seed)
    parser.add_argument("--no-export", action="store_true")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()

    config = NARXConfig(epochs=args.epochs, seed=args.seed)
    set_seed(config.seed)
    print(f"Config: {config}")

    data = np.load(args.data)
    theta = data["th"]
    u = data["u"]

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
    theta_mean = float(theta_train.mean())
    theta_std = float(theta_train.std())

    u_train_norm = (u_train - u_mean) / u_std
    theta_train_norm = (theta_train - theta_mean) / theta_std
    u_val_norm = (u_val - u_mean) / u_std
    theta_val_norm = (theta_val - theta_mean) / theta_std

    x_train, y_train = create_io_data(u_train_norm, theta_train_norm, config)
    x_val, y_val = create_io_data(u_val_norm, theta_val_norm, config)

    model = NARXNet(config)
    model, history = train_model(model, x_train, y_train, x_val, y_val, theta_std, config)

    val_pred, val_true_pred = one_step_prediction(model, u_val_norm, theta_val_norm, theta_mean, theta_std, config)
    val_sim, val_true_sim = free_run_simulation(model, u_val_norm, theta_val_norm, theta_mean, theta_std, config)

    metrics = {
        "config": asdict(config),
        **split_metrics(model, u_val, theta_val, u_mean, u_std, theta_mean, theta_std, config, "validation"),
        **split_metrics(model, u_test, theta_test, u_mean, u_std, theta_mean, theta_std, config, "test"),
    }

    print("\n--- NARX validation and held-out test metrics ---")
    for key, value in metrics.items():
        if key == "config":
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.6f}")

    torch.save(model.state_dict(), args.checkpoint)
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Saved metrics: {args.metrics_output}")

    if not args.no_figures:
        save_report_figures(
            args.figures_dir,
            history,
            theta_val,
            val_pred,
            val_true_pred,
            val_sim,
            val_true_sim,
            config,
        )

    if not args.no_export:
        export_prediction_submission(
            model,
            args.prediction_template,
            args.prediction_output,
            u_mean,
            u_std,
            theta_mean,
            theta_std,
            config,
        )
        export_simulation_submission(
            model,
            args.simulation_template,
            args.simulation_output,
            u_mean,
            u_std,
            theta_mean,
            theta_std,
            config,
            args.simulation_init_steps,
        )


if __name__ == "__main__":
    main()
