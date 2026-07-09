from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_training_config, resolve_project_path


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        return {k: 1.0 / len(weights) for k in weights}
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


def apply_compression_prior(
    weights: dict[str, float],
    manifest_path: Path,
    strength: float = 0.35,
) -> dict[str, float]:
    if not manifest_path.exists():
        return normalize_weights(weights)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    avg = manifest.get("avg_gzip_ratio_by_category", {})
    adjusted = {}
    for category, weight in weights.items():
        ratio = float(avg.get(category, 0.5))
        # Favor the middle band. Very low ratios are often repeated boilerplate;
        # very high ratios can be noisy or code-like entropy.
        quality = max(0.05, 1.0 - abs(ratio - 0.52) / 0.42)
        adjusted[category] = weight * ((1.0 - strength) + strength * quality)
    return normalize_weights(adjusted)


def sample_candidate_mixtures(categories: list[str], count: int, seed: int) -> list[dict[str, float]]:
    rng = np.random.default_rng(seed)
    mixtures = []
    for _ in range(count):
        alpha = np.ones(len(categories))
        weights = rng.dirichlet(alpha)
        mixtures.append({cat: float(w) for cat, w in zip(categories, weights)})
    return mixtures


def fit_ridge_regression(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    reg = alpha * np.eye(x_aug.shape[1])
    reg[0, 0] = 0.0
    return np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y)


def predict_ridge(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
    return x_aug @ coef


def select_best_from_proxy_results(results_path: Path, ridge_alpha: float) -> dict:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    categories = rows["categories"]
    observations = rows["observations"]
    x = np.array([[obs["mixture"][cat] for cat in categories] for obs in observations], dtype=np.float64)
    # Lower validation loss is better, so regress negative loss as a score.
    y = -np.array([obs["val_loss"] for obs in observations], dtype=np.float64)
    coef = fit_ridge_regression(x, y, alpha=ridge_alpha)
    candidate_x = np.array([[cand[cat] for cat in categories] for cand in rows["candidates"]], dtype=np.float64)
    scores = predict_ridge(coef, candidate_x)
    best_idx = int(np.argmax(scores))
    return {
        "categories": categories,
        "best_mixture": rows["candidates"][best_idx],
        "predicted_score": float(scores[best_idx]),
        "coefficients": coef.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    parser.add_argument("--write-candidates", action="store_true")
    parser.add_argument("--select", type=str, help="Path to proxy_results.json.")
    args = parser.parse_args()

    config = load_training_config(args.config)
    regmix = config["regmix"]
    out_dir = resolve_project_path(config, regmix["proxy_output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    categories = list(regmix["categories"])

    if args.write_candidates:
        candidates = sample_candidate_mixtures(
            categories,
            int(regmix["candidate_mixtures"]),
            int(config["training"].get("seed", 42)),
        )
        out_path = out_dir / "candidate_mixtures.json"
        out_path.write_text(json.dumps({"categories": categories, "candidates": candidates}, indent=2), encoding="utf-8")
        print(f"[regmix] wrote {out_path}")

    if args.select:
        result = select_best_from_proxy_results(Path(args.select), float(regmix.get("ridge_alpha", 0.1)))
        out_path = out_dir / "selected_mixture.json"
        out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
