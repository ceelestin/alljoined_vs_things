"""Per-timepoint linear decoding of image features from EEG.

Segments EEG around Image stimulus onsets, extracts DINOv2 image
embeddings as targets, and fits a Ridge regression at every timepoint.
Evaluation metric: Pearson correlation between predicted and true features.

Requires:
    pip install neuralset scikit-learn matplotlib torch

Usage:
    python linear_decoding.py --study_dir /path/to/studies --cache_dir /tmp/cache
    python linear_decoding.py --study_dir /path/to/studies --cache_dir /tmp/cache \
        --n_subjects 1 --n_images 2000
    python linear_decoding.py --study_dir /path/to/studies --cache_dir /tmp/cache \
        --n_subjects 5 --n_images 2000 --window_start -0.05 --window_duration 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from exca import MapInfra
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import neuralset as ns

# ---- monkey-patch for neuralhub Xu2025._get_fname bug ----
# iter_timelines yields subject as str, but _get_fname expects int.
# from neuralhub.internal import xu2025 as _xu2025

_orig_get_fname = _xu2025.Xu2025._get_fname


@staticmethod  # type: ignore[misc]
def _patched_get_fname(path, subject, session, run):  # type: ignore[no-untyped-def]
    return _orig_get_fname(path, int(subject), int(session), int(run))


_xu2025.Xu2025._get_fname = _patched_get_fname  # type: ignore[assignment]
# -----------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-timepoint linear decoding: EEG → DINOv2"
    )
    parser.add_argument(
        "--study_dir", type=str, required=True,
        help="Root directory containing the Xu2025 study data",
    )
    parser.add_argument(
        "--cache_dir", type=str, required=True,
        help="Directory for caching extracted features",
    )
    parser.add_argument(
        "--n_subjects", type=int, default=None,
        help="Number of subjects to use (default: all available)",
    )
    parser.add_argument(
        "--n_images", type=int, default=None,
        help="Max image trials per subject (default: all available)",
    )
    parser.add_argument(
        "--window_start", type=float, default=-0.05,
        help="Window start relative to stimulus onset in seconds (default: -0.05)",
    )
    parser.add_argument(
        "--window_duration", type=float, default=0.5,
        help="Total window length in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--ridge_alpha", type=float, default=1.0,
        help="Ridge regularisation alpha (default: 1.0)",
    )
    parser.add_argument(
        "--test_size", type=float, default=0.2,
        help="Fraction of trials for the test set (default: 0.2)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split (default: 42)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results",
        help="Directory to save figures (default: results/)",
    )
    return parser.parse_args()


def pearson_r(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Pearson R across feature dimensions."""
    y_pred_c = y_pred - y_pred.mean(axis=0, keepdims=True)
    y_true_c = y_true - y_true.mean(axis=0, keepdims=True)
    num = (y_pred_c * y_true_c).sum(axis=0)
    den = np.sqrt((y_pred_c**2).sum(axis=0) * (y_true_c**2).sum(axis=0))
    return float(np.nanmean(num / np.maximum(den, 1e-8)))


def load_events(study_dir: str, cache_dir: str, n_subjects: int | None) -> "pd.DataFrame":
    """Load the Xu2025 study events, optionally keeping only n_subjects."""
    study_cfg: dict = {
        "name": "Xu2025",
        "path": study_dir,
        "infra_timelines": {"cluster": None},
    }
    if n_subjects is not None:
        study_cfg["query"] = f"subject_index < {n_subjects}"
    loader = ns.StudyLoader(
        study=study_cfg,
        infra={"backend": "Cached", "folder": cache_dir},
    )
    return loader.build()


def filter_images_per_subject(
    events: "pd.DataFrame", n_images: int | None
) -> "pd.DataFrame":
    """Keep at most n_images Image events per subject."""
    import pandas as pd  # noqa: F811

    if n_images is None:
        return events
    non_image = events[events["type"] != "Image"]
    image_subset = (
        events[events["type"] == "Image"]
        .groupby("subject", sort=False)
        .head(n_images)
    )
    return pd.concat([non_image, image_subset]).sort_index().reset_index(drop=True)


def decode_subject(
    events: "pd.DataFrame",
    features: dict,
    subject: str,
    *,
    window_start: float,
    window_duration: float,
    ridge_alpha: float,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Run per-timepoint decoding for a single subject.

    Returns (times, scores) or None if not enough segments.
    """
    mask = (events["type"] == "Image") & (events["subject"] == subject)
    segments = ns.segments.list_segments(
        events, mask, duration=window_duration, start=window_start,
    )
    if len(segments) < 10:
        print(f"  Skipping {subject}: only {len(segments)} segments")
        return None

    dataset = ns.SegmentDataset(features, segments)
    dataloader = DataLoader(
        dataset,
        batch_size=len(segments),
        num_workers=4,
        collate_fn=dataset.collate_fn,
    )
    batch = next(iter(dataloader))

    X = batch.data["eeg"].numpy().astype(np.float32)
    y = batch.data["image"].numpy().astype(np.float32)
    if y.ndim > 2:
        y = y.reshape(y.shape[0], -1)

    n_trials, n_channels, n_times = X.shape
    print(f"  EEG: {X.shape}, Image features: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed,
    )

    scores = np.zeros(n_times)
    for t in range(n_times):
        clf = Ridge(alpha=ridge_alpha)
        clf.fit(X_train[:, :, t], y_train)
        y_pred = clf.predict(X_test[:, :, t])
        scores[t] = pearson_r(y_pred, y_test)

    times = np.linspace(window_start, window_start + window_duration, n_times)
    return times, scores


def plot_results(
    all_scores: dict[str, np.ndarray],
    times: np.ndarray,
    *,
    n_images: int | None,
) -> plt.Figure:
    """Plot temporal decoding curves per subject + mean ± std."""
    n_subj = len(all_scores)
    fig, ax = plt.subplots(figsize=(8, 4))

    scores_mat = np.array(list(all_scores.values()))
    mean_scores = scores_mat.mean(axis=0)

    if n_subj > 1:
        std_scores = scores_mat.std(axis=0)
        for subj, sc in all_scores.items():
            ax.plot(times, sc, alpha=0.15, lw=0.8, color="steelblue")
        ax.fill_between(
            times,
            mean_scores - std_scores,
            mean_scores + std_scores,
            color="steelblue",
            alpha=0.25,
            label="± 1 std",
        )
        ax.plot(times, mean_scores, color="k", lw=2.5, label="mean")
    else:
        subj_name = next(iter(all_scores.keys())).split("/")[-1]
        ax.plot(times, mean_scores, color="steelblue", lw=2, label=subj_name)

    ax.axvline(0, color="gray", ls="--")
    ax.axhline(0, color="gray", ls=":", alpha=0.5)

    images_str = f", {n_images} trials each" if n_images else ""
    ax.set(
        xlabel="Time relative to stimulus (s)",
        ylabel="Pearson R",
        title=f"Temporal decoding — {n_subj} subject(s){images_str}",
    )
    ax.legend(fontsize=8, loc="upper right")
    plt.tight_layout()

    peak_t = times[mean_scores.argmax()]
    print(f"\nMean peak R = {mean_scores.max():.4f} at t = {peak_t:.3f}s")

    return fig


if __name__ == "__main__":
    args = parse_args()

    cache_dir = args.cache_dir
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # --- load events ---
    events = load_events(args.study_dir, cache_dir, args.n_subjects)
    print(f"Loaded {len(events)} events, {events['subject'].nunique()} subjects")

    # --- filter images per subject ---
    events = filter_images_per_subject(events, args.n_images)
    img_per_subj = events[events["type"] == "Image"].groupby("subject").size()
    print(f"Image trials per subject:\n{img_per_subj.to_string()}")

    # --- extractors ---
    cache_infra = MapInfra(folder=cache_dir, cluster=None)

    eeg = ns.extractors.EegExtractor(
        frequency=100,
        filter=(0.1, 100.0),
        scaler="RobustScaler",
        infra=cache_infra,
    )
    image_feat = ns.extractors.HuggingFaceImage(
        model_name="facebook/dinov2-base",
        aggregation="trigger",
        infra=cache_infra,
    )
    eeg.prepare(events)
    image_feat.prepare(events)
    features = {"eeg": eeg, "image": image_feat}

    # --- per-subject decoding ---
    subjects = sorted(events["subject"].unique())
    all_scores: dict[str, np.ndarray] = {}
    times_arr: np.ndarray | None = None

    for subject in subjects:
        print(f"\n{'='*60}\nSubject: {subject}")
        result = decode_subject(
            events,
            features,
            subject,
            window_start=args.window_start,
            window_duration=args.window_duration,
            ridge_alpha=args.ridge_alpha,
            test_size=args.test_size,
            seed=args.seed,
        )
        if result is None:
            continue
        times_arr, scores = result
        all_scores[subject] = scores
        peak_t = times_arr[scores.argmax()]
        print(f"  Peak R = {scores.max():.4f} at t = {peak_t:.3f}s")

    if not all_scores:
        print("No subjects decoded — exiting.")
        raise SystemExit(1)

    assert times_arr is not None

    # --- summary plot ---
    fig = plot_results(all_scores, times_arr, n_images=args.n_images)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_subj = len(all_scores)
    n_img_tag = f"_{args.n_images}img" if args.n_images else "_allimg"
    out_path = out_dir / f"linear_decoding_{n_subj}subj{n_img_tag}.png"
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close(fig)
