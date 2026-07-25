"""EXP-1 — DS1 in-distribution sweep + best_encoder selection.

Per ``docs/experiment_matrix.md`` EXP-1 / §9 and ``docs/final_protocol.md``:

* Dataset: DS1 ``vDEDUP``, split ``S-RAND``
* Models: ``LR, SVM, RF, LGBM`` (classical / TF-IDF on ``vDEDUP`` title+text)
  + ``BERT, ROBERTA, DEBERTA, DISTIL, ALBERT`` (encoders / ``vCLEAN-N`` via
  ``pre_dedup_index``)
* Seeds: primary set from ``configs/protocol.yaml`` (default {13,21,42,87,100})
* Metrics: METRICS-CORE + ECE; pairwise encoder STATS-CONF (H1)
* Outputs: ``results/ds1_indist.csv``, ``results/ds1_encoder_pairwise_stats.csv``,
  ``results/best_encoder.txt``

``best_encoder`` tie-break (deterministic, pre_implementation_review W5):
    highest mean macro-F1 → lowest mean ECE → fewest parameters.
    Never best-run-only.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from fnb.config import config_hash, load_config
from fnb.config.schema import PROTOCOL_VERSION
from fnb.data.dedup import combine_title_text
from fnb.data.splits import index_path
from fnb.evaluation.metrics import core_metrics, ece
from fnb.evaluation.results_io import make_result_row, write_result_rows
from fnb.evaluation.stats import (
    assemble_confirmatory,
    cliffs_delta,
    mcnemar_test,
    paired_bootstrap_diff,
)
from fnb.models.classical import CLASSICAL_MODEL_IDS, train_classical
from fnb.models.encoder import build_encoder, predict_from_loader, resolve_encoder_name
from fnb.training.datasets import WindowedTextDataset, collate_windows
from fnb.training.loop import train as train_encoder
from fnb.utils.io import ensure_dir, load_indices
from fnb.utils.run_registry import finish_run, start_run
from fnb.utils.seeding import set_global_seed

logger = logging.getLogger("fnb.experiments.exp1")

EXPERIMENT_ID = "EXP-1"
DATASET_ID = "DS1"
DATASET_VERSION_TAG = "vDEDUP"
SPLIT_TYPE = "S-RAND"

ENCODER_MODEL_IDS: tuple[str, ...] = ("BERT", "ROBERTA", "DEBERTA", "DISTIL", "ALBERT")
EXP1_MODEL_IDS: tuple[str, ...] = (*CLASSICAL_MODEL_IDS, *ENCODER_MODEL_IDS)

CONFIG_NAMES_CLASSICAL = ("protocol", "classical_grids", "preprocessing", "metrics", "datasets")
CONFIG_NAMES_ENCODER = ("protocol", "encoder", "preprocessing", "metrics", "datasets")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SplitBundle:
    """Train/val/test texts + labels for one seed (one preprocessing track)."""

    texts_train: list[str]
    y_train: np.ndarray
    texts_val: list[str]
    y_val: np.ndarray
    texts_test: list[str]
    y_test: np.ndarray
    idx_train: np.ndarray
    idx_val: np.ndarray
    idx_test: np.ndarray


def primary_seeds(config_dir: str | Path | None = None) -> list[int]:
    """Primary endpoint seeds from ``configs/protocol.yaml`` (§9)."""
    cfg = load_config("protocol", config_dir)
    return [int(s) for s in cfg.seeds.primary]


def _composite_config_hash(names: Sequence[str], config_dir: str | Path | None) -> str:
    parts = [f"{n}:{config_hash(n, config_dir)}" for n in names]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _texts_from_frame(df: pd.DataFrame) -> list[str]:
    return combine_title_text(df).tolist()


def load_ds1_split(
    seed: int,
    *,
    track: str,
    processed_dir: str | Path = "data/processed",
    splits_dir: str | Path = "data/splits",
) -> SplitBundle:
    """Load DS1 train/val/test for classical or neural track.

    Split indices address **vDEDUP** row positions. For the neural track, texts
    are resolved via ``pre_dedup_index`` into the row-aligned ``vCLEAN-N`` frame
    (same dedup survivors, neural preprocessing).
    """
    processed = Path(processed_dir)
    vdedup_path = processed / f"{DATASET_ID}_vDEDUP.parquet"
    if not vdedup_path.is_file():
        raise FileNotFoundError(f"missing {vdedup_path} — run data pipeline P2 first")

    vdedup = pd.read_parquet(vdedup_path)
    if "label" not in vdedup.columns:
        raise KeyError(f"{vdedup_path}: missing 'label'")

    parts: dict[str, np.ndarray] = {}
    for part in ("train", "val", "test"):
        path = index_path(DATASET_ID, SPLIT_TYPE, part, seed, splits_dir=splits_dir)
        if not path.is_file():
            raise FileNotFoundError(f"missing split index {path} — run data pipeline P3")
        parts[part] = np.asarray(load_indices(path), dtype=int)

    labels = vdedup["label"].to_numpy(dtype=int)

    if track == "classical":
        all_texts = _texts_from_frame(vdedup)
    elif track == "neural":
        if "pre_dedup_index" not in vdedup.columns:
            raise KeyError(f"{vdedup_path}: missing 'pre_dedup_index'")
        pre = vdedup["pre_dedup_index"].to_numpy(dtype=int)
        n_path = processed / f"{DATASET_ID}_vCLEAN-N.parquet"
        if not n_path.is_file():
            raise FileNotFoundError(f"missing {n_path} — run data pipeline P1b first")
        vclean_n = pd.read_parquet(n_path)
        neural_all = _texts_from_frame(vclean_n)
        all_texts = [neural_all[int(i)] for i in pre]
    else:
        raise ValueError(f"track must be 'classical' or 'neural'; got {track!r}")

    def _slice(idxs: np.ndarray) -> tuple[list[str], np.ndarray]:
        return [all_texts[int(i)] for i in idxs], labels[idxs]

    tr_t, tr_y = _slice(parts["train"])
    va_t, va_y = _slice(parts["val"])
    te_t, te_y = _slice(parts["test"])
    return SplitBundle(
        texts_train=tr_t, y_train=tr_y,
        texts_val=va_t, y_val=va_y,
        texts_test=te_t, y_test=te_y,
        idx_train=parts["train"], idx_val=parts["val"], idx_test=parts["test"],
    )


# ---------------------------------------------------------------------------
# Prediction artifact I/O (consumed by pairwise stats)
# ---------------------------------------------------------------------------

def pred_artifact_path(
    model_id: str, seed: int, *, artifacts_dir: str | Path,
) -> Path:
    return Path(artifacts_dir) / "exp1_preds" / f"{model_id}_seed{seed}.npz"


def save_predictions(
    path: str | Path,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_indices: np.ndarray,
    model_id: str,
    seed: int,
) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    np.savez_compressed(
        p,
        y_true=np.asarray(y_true, dtype=int),
        y_pred=np.asarray(y_pred, dtype=int),
        y_prob=np.asarray(y_prob, dtype=float),
        test_indices=np.asarray(test_indices, dtype=int),
        model_id=np.asarray(str(model_id)),
        seed=np.asarray(int(seed)),
    )
    return p


def load_predictions(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "y_true": z["y_true"],
            "y_pred": z["y_pred"],
            "y_prob": z["y_prob"],
            "test_indices": z["test_indices"],
        }


# ---------------------------------------------------------------------------
# Metrics helper
# ---------------------------------------------------------------------------

def _metrics_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    *,
    model_id: str,
    config_dir: str | Path | None,
) -> dict[str, float]:
    m = core_metrics(y_true, y_pred, y_prob)
    m["ece"] = ece(y_true, y_prob, model_id=model_id, config_dir=config_dir)
    return m


# ---------------------------------------------------------------------------
# Train-and-evaluate cells (one model × one seed)
# ---------------------------------------------------------------------------

def _existing_result_keys(csv_path: Path) -> set[tuple[str, int]]:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return set()
    df = pd.read_csv(csv_path)
    if "model_id" not in df.columns or "seed" not in df.columns:
        return set()
    return {(str(r.model_id), int(r.seed)) for r in df.itertuples(index=False)}


def run_classical_cell(
    model_id: str,
    seed: int,
    *,
    processed_dir: str | Path,
    splits_dir: str | Path,
    artifacts_dir: str | Path,
    results_dir: str | Path,
    config_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Train/eval one classical model × seed; append row + save predictions."""
    set_global_seed(seed)
    split = load_ds1_split(
        seed, track="classical", processed_dir=processed_dir, splits_dir=splits_dir,
    )
    art = Path(artifacts_dir)
    out_csv = Path(results_dir) / "ds1_indist.csv"
    ctx = start_run(
        EXPERIMENT_ID, model_id, DATASET_VERSION_TAG, SPLIT_TYPE, seed,
        config_names=list(CONFIG_NAMES_CLASSICAL),
        base_dir=base_dir, results_dir="results", logs_dir="logs",
    )
    try:
        result = train_classical(
            model_id,
            split.texts_train, split.y_train,
            split.texts_val, split.y_val,
            seed=seed,
            texts_test=split.texts_test,
            artifacts_dir=art / "classical",
            config_dir=config_dir,
            save_model=True,
        )
        assert result.y_pred_test is not None and result.y_prob_test is not None
        metrics = _metrics_dict(
            split.y_test, result.y_pred_test, result.y_prob_test,
            model_id=model_id, config_dir=config_dir,
        )
        save_predictions(
            pred_artifact_path(model_id, seed, artifacts_dir=art),
            y_true=split.y_test, y_pred=result.y_pred_test,
            y_prob=result.y_prob_test, test_indices=split.idx_test,
            model_id=model_id, seed=seed,
        )
        cfg_hash = _composite_config_hash(CONFIG_NAMES_CLASSICAL, config_dir)
        row = make_result_row(
            dataset_version_tag=DATASET_VERSION_TAG, split_type=SPLIT_TYPE,
            model_id=model_id, seed=seed, run_id=ctx.run_id,
            config_hash=cfg_hash, git_sha=ctx.git_commit,
            metrics=metrics,
            val_macro_f1=result.val_macro_f1,
            best_params_json=json.dumps(result.best_params, sort_keys=True),
            family="classical",
        )
        write_result_rows(out_csv, [row])
        finish_run(ctx, status="completed", metrics=metrics, artifact_dir=art / "classical")
        return row
    except Exception:
        finish_run(ctx, status="failed")
        raise


def run_encoder_cell(
    model_id: str,
    seed: int,
    *,
    processed_dir: str | Path,
    splits_dir: str | Path,
    artifacts_dir: str | Path,
    results_dir: str | Path,
    config_dir: str | Path | None = None,
    base_dir: str | Path | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Train/eval one encoder × seed; append row + save predictions + checkpoint."""
    from transformers import AutoTokenizer

    set_global_seed(seed)
    enc_cfg = load_config("encoder", config_dir)
    batch_size = int(enc_cfg.training.batch_size)

    split = load_ds1_split(
        seed, track="neural", processed_dir=processed_dir, splits_dir=splits_dir,
    )
    art = Path(artifacts_dir)
    ckpt_dir = art / "checkpoints" / f"{EXPERIMENT_ID}_{model_id}_seed{seed}"
    ensure_dir(ckpt_dir)
    out_csv = Path(results_dir) / "ds1_indist.csv"

    ctx = start_run(
        EXPERIMENT_ID, model_id, DATASET_VERSION_TAG, SPLIT_TYPE, seed,
        config_names=list(CONFIG_NAMES_ENCODER),
        base_dir=base_dir, results_dir="results", logs_dir="logs",
    )
    try:
        hub_name = resolve_encoder_name(model_id, config_dir=config_dir)
        tokenizer = AutoTokenizer.from_pretrained(hub_name)
        model = build_encoder(model_id, config_dir=config_dir, pretrained=True)

        train_ds = WindowedTextDataset(
            split.texts_train, split.y_train.tolist(), tokenizer,
            config_dir=config_dir,
        )
        val_ds = WindowedTextDataset(
            split.texts_val, split.y_val.tolist(), tokenizer,
            config_dir=config_dir,
        )
        test_ds = WindowedTextDataset(
            split.texts_test, split.y_test.tolist(), tokenizer,
            config_dir=config_dir,
        )
        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_windows,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_windows,
        )
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_windows,
        )

        train_result = train_encoder(
            model, train_loader, val_loader,
            seed=seed, train_labels=split.y_train,
            config_dir=config_dir, device=device, checkpoint_dir=ckpt_dir,
        )
        pred = predict_from_loader(model, test_loader, device=device)
        y_prob_pos = pred.y_prob[:, 1] if pred.y_prob.ndim == 2 else pred.y_prob
        metrics = _metrics_dict(
            split.y_test, pred.y_pred, y_prob_pos,
            model_id=model_id, config_dir=config_dir,
        )
        n_params = int(sum(p.numel() for p in model.parameters()))
        save_predictions(
            pred_artifact_path(model_id, seed, artifacts_dir=art),
            y_true=split.y_test, y_pred=pred.y_pred,
            y_prob=y_prob_pos, test_indices=split.idx_test,
            model_id=model_id, seed=seed,
        )
        cfg_hash = _composite_config_hash(CONFIG_NAMES_ENCODER, config_dir)
        row = make_result_row(
            dataset_version_tag=DATASET_VERSION_TAG, split_type=SPLIT_TYPE,
            model_id=model_id, seed=seed, run_id=ctx.run_id,
            config_hash=cfg_hash, git_sha=ctx.git_commit,
            metrics=metrics,
            val_macro_f1=train_result.best_val_macro_f1,
            best_epoch=train_result.best_epoch,
            n_params=n_params,
            family="encoder",
            checkpoint_path=str(train_result.checkpoint_path or ""),
        )
        write_result_rows(out_csv, [row])
        finish_run(ctx, status="completed", metrics=metrics, artifact_dir=ckpt_dir)
        return row
    except Exception:
        finish_run(ctx, status="failed")
        raise


# ---------------------------------------------------------------------------
# best_encoder selection (deterministic tie-break)
# ---------------------------------------------------------------------------

def select_best_encoder(
    ds1_indist: pd.DataFrame,
    *,
    encoder_ids: Sequence[str] = ENCODER_MODEL_IDS,
    param_counts: Mapping[str, int] | None = None,
) -> str:
    """Deterministic best_encoder: max mean macro-F1 → min mean ECE → min params.

    Aggregates **across seeds** (never best-run-only). ``param_counts`` may be
    supplied; missing counts sort last.
    """
    df = ds1_indist[ds1_indist["model_id"].isin(list(encoder_ids))].copy()
    if df.empty:
        raise ValueError("no encoder rows in ds1_indist for best_encoder selection")

    rows: list[tuple[float, float, int, str]] = []
    for mid in encoder_ids:
        sub = df[df["model_id"] == mid]
        if sub.empty:
            continue
        f1 = float(pd.to_numeric(sub["macro_f1"], errors="coerce").mean())
        ece_vals = pd.to_numeric(sub["ece"], errors="coerce")
        ece_mean = float(ece_vals.mean()) if ece_vals.notna().any() else float("inf")
        if param_counts is not None and mid in param_counts:
            n_p = int(param_counts[mid])
        elif "n_params" in sub.columns and sub["n_params"].notna().any():
            n_p = int(pd.to_numeric(sub["n_params"], errors="coerce").dropna().iloc[0])
        else:
            n_p = 2**62
        # Sort key: higher F1 → negate; lower ECE; lower params.
        rows.append((-f1, ece_mean, n_p, str(mid)))

    if not rows:
        raise ValueError("no complete encoder groups for best_encoder selection")
    rows.sort()
    return rows[0][3]


def write_best_encoder(path: str | Path, model_id: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(f"{model_id.strip()}\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Pairwise encoder statistics (H1)
# ---------------------------------------------------------------------------

def compute_encoder_pairwise_stats(
    *,
    ds1_indist: pd.DataFrame,
    artifacts_dir: str | Path,
    seeds: Sequence[int],
    encoder_ids: Sequence[str] = ENCODER_MODEL_IDS,
    config_dir: str | Path | None = None,
    n_resamples: int | None = None,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Pairwise STATS-CONF among encoders (H1); BH within this EXP-1 subfamily.

    * Paired bootstrap + McNemar: stacked per-seed test predictions (same
      seed → same test set; seeds stacked for a single instance-level test).
    * Cliff's delta: on the **per-seed** macro-F1 vectors (protocol §10).
    * ``p_adjusted`` here is BH within the pairwise subfamily; EXP-G4
      re-applies BH across the full confirmatory family using ``p_raw``.
    """
    art = Path(artifacts_dir)

    f1_by_model: dict[str, list[float]] = {m: [] for m in encoder_ids}
    for mid in encoder_ids:
        sub = ds1_indist[
            (ds1_indist["model_id"] == mid) & (ds1_indist["seed"].isin(list(seeds)))
        ].sort_values("seed")
        if len(sub) != len(seeds):
            raise ValueError(
                f"{mid}: expected {len(seeds)} seed rows, got {len(sub)}"
            )
        f1_by_model[mid] = [
            float(x) for x in pd.to_numeric(sub["macro_f1"], errors="coerce")
        ]

    stacked: dict[str, dict[str, np.ndarray]] = {}
    for mid in encoder_ids:
        yt_parts: list[np.ndarray] = []
        yp_parts: list[np.ndarray] = []
        for seed in seeds:
            path = pred_artifact_path(mid, int(seed), artifacts_dir=art)
            if not path.is_file():
                raise FileNotFoundError(f"missing prediction artifact: {path}")
            blob = load_predictions(path)
            yt_parts.append(blob["y_true"])
            yp_parts.append(blob["y_pred"])
        stacked[mid] = {
            "y_true": np.concatenate(yt_parts),
            "y_pred": np.concatenate(yp_parts),
        }

    family: list[dict[str, Any]] = []
    pair_meta: list[dict[str, Any]] = []
    y_true = stacked[encoder_ids[0]]["y_true"]

    for a, b in itertools.combinations(encoder_ids, 2):
        boot = paired_bootstrap_diff(
            y_true, stacked[a]["y_pred"], stacked[b]["y_pred"],
            n_resamples=n_resamples, seed=bootstrap_seed, config_dir=config_dir,
        )
        mc = mcnemar_test(y_true, stacked[a]["y_pred"], stacked[b]["y_pred"])
        delta = cliffs_delta(f1_by_model[a], f1_by_model[b])
        name = f"{a}_vs_{b}"
        family.append({
            "name": name,
            "p_raw": boot.p_value,
            "effect_size": delta,
            "ci_low": boot.ci_low,
            "ci_high": boot.ci_high,
            "mean_diff": boot.mean_diff,
        })
        pair_meta.append({
            "name": name,
            "model_a": a,
            "model_b": b,
            "observed_diff": boot.observed_diff,
            "mcnemar_statistic": mc.statistic,
            "mcnemar_p_value": mc.p_value,
            "n_resamples": boot.n_resamples,
            "n_seeds": len(seeds),
            "n_paired_examples": int(len(y_true)),
        })

    confirmed = assemble_confirmatory(family, config_dir=config_dir)
    records: list[dict[str, Any]] = []
    for meta, cmp_ in zip(pair_meta, confirmed, strict=True):
        records.append({
            "protocol_version": PROTOCOL_VERSION,
            "experiment": EXPERIMENT_ID,
            "hypothesis": "H1",
            "comparison": meta["name"],
            "model_a": meta["model_a"],
            "model_b": meta["model_b"],
            "mean_diff": cmp_.mean_diff,
            "observed_diff": meta["observed_diff"],
            "ci_low": cmp_.ci_low,
            "ci_high": cmp_.ci_high,
            "p_raw": cmp_.p_raw,
            "p_adjusted": cmp_.p_adjusted,
            "cliffs_delta": cmp_.effect_size,
            "mcnemar_statistic": meta["mcnemar_statistic"],
            "mcnemar_p_value": meta["mcnemar_p_value"],
            "significant": cmp_.significant,
            "claim_allowed": cmp_.claim_allowed,
            "bh_scope": "exp1_encoder_pairwise_interim",
            "n_resamples": meta["n_resamples"],
            "n_seeds": meta["n_seeds"],
            "n_paired_examples": meta["n_paired_examples"],
            "note": (
                "p_adjusted is BH within EXP-1 pairwise subfamily; "
                "EXP-G4 re-applies BH across the full confirmatory family"
            ),
        })
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# Finalize: pairwise stats + best_encoder
# ---------------------------------------------------------------------------

def finalize_exp1(
    *,
    results_dir: str | Path,
    artifacts_dir: str | Path,
    seeds: Sequence[int] | None = None,
    config_dir: str | Path | None = None,
    n_resamples: int | None = None,
    param_counts: Mapping[str, int] | None = None,
) -> dict[str, Path]:
    """Write pairwise stats + best_encoder.txt once all encoder×seed cells exist."""
    results = Path(results_dir)
    csv_path = results / "ds1_indist.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing {csv_path} — run EXP-1 cells first")

    seed_list = list(seeds) if seeds is not None else primary_seeds(config_dir)
    enc_ids = list(ENCODER_MODEL_IDS)

    df = pd.read_csv(csv_path)
    enc_needed = {(m, int(s)) for m in enc_ids for s in seed_list}
    have = {(str(r.model_id), int(r.seed)) for r in df.itertuples(index=False)}
    missing = sorted(enc_needed - have)
    if missing:
        raise RuntimeError(
            f"cannot finalize EXP-1: missing encoder cells "
            f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
        )

    pair_df = compute_encoder_pairwise_stats(
        ds1_indist=df, artifacts_dir=artifacts_dir,
        seeds=seed_list, encoder_ids=enc_ids,
        config_dir=config_dir, n_resamples=n_resamples,
    )
    pair_path = results / "ds1_encoder_pairwise_stats.csv"
    ensure_dir(pair_path.parent)
    pair_df.to_csv(pair_path, index=False)

    best = select_best_encoder(df, encoder_ids=enc_ids, param_counts=param_counts)
    best_path = write_best_encoder(results / "best_encoder.txt", best)
    logger.info("best_encoder=%s written to %s", best, best_path)
    return {
        "ds1_indist": csv_path,
        "ds1_encoder_pairwise_stats": pair_path,
        "best_encoder": best_path,
    }


# ---------------------------------------------------------------------------
# Top-level sweep (resumable)
# ---------------------------------------------------------------------------

def cells_complete(
    csv_path: Path,
    *,
    model_ids: Sequence[str] = EXP1_MODEL_IDS,
    seeds: Sequence[int],
) -> bool:
    have = _existing_result_keys(csv_path)
    needed = {(m, int(s)) for m in model_ids for s in seeds}
    return needed.issubset(have)


def run_exp1(
    *,
    models: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
    processed_dir: str | Path = "data/processed",
    splits_dir: str | Path = "data/splits",
    output_dir: str | Path = ".",
    config_dir: str | Path | None = None,
    force: bool = False,
    finalize: bool = True,
    device: str | None = None,
    n_resamples: int | None = None,
) -> dict[str, Any]:
    """Run requested EXP-1 cells and optionally finalize pairwise + best_encoder.

    Resumable: skips (model, seed) pairs already present in ``ds1_indist.csv``
    unless ``force=True``.
    """
    out = Path(output_dir)
    results_dir = out / "results"
    artifacts_dir = out / "artifacts"
    ensure_dir(results_dir)
    ensure_dir(artifacts_dir)

    seed_list = list(seeds) if seeds is not None else primary_seeds(config_dir)
    model_list = list(models) if models is not None else list(EXP1_MODEL_IDS)
    for m in model_list:
        if m not in EXP1_MODEL_IDS:
            raise ValueError(
                f"unknown EXP-1 model_id {m!r}; expected one of {EXP1_MODEL_IDS}"
            )

    csv_path = results_dir / "ds1_indist.csv"
    existing = set() if force else _existing_result_keys(csv_path)
    ran: list[tuple[str, int]] = []
    skipped: list[tuple[str, int]] = []

    for model_id, seed in itertools.product(model_list, seed_list):
        key = (model_id, int(seed))
        if key in existing:
            logger.info("skip existing cell %s seed=%s", model_id, seed)
            skipped.append(key)
            continue
        logger.info("running EXP-1 cell model=%s seed=%s", model_id, seed)
        if model_id in CLASSICAL_MODEL_IDS:
            run_classical_cell(
                model_id, int(seed),
                processed_dir=processed_dir, splits_dir=splits_dir,
                artifacts_dir=artifacts_dir, results_dir=results_dir,
                config_dir=config_dir, base_dir=out,
            )
        else:
            run_encoder_cell(
                model_id, int(seed),
                processed_dir=processed_dir, splits_dir=splits_dir,
                artifacts_dir=artifacts_dir, results_dir=results_dir,
                config_dir=config_dir, base_dir=out, device=device,
            )
        ran.append(key)
        existing.add(key)

    summary: dict[str, Any] = {
        "ran": ran,
        "skipped": skipped,
        "results_csv": str(csv_path),
        "finalized": False,
        "finalize_paths": {},
    }

    all_seeds = primary_seeds(config_dir)
    if finalize and cells_complete(csv_path, seeds=all_seeds):
        paths = finalize_exp1(
            results_dir=results_dir, artifacts_dir=artifacts_dir,
            seeds=all_seeds, config_dir=config_dir, n_resamples=n_resamples,
        )
        summary["finalized"] = True
        summary["finalize_paths"] = {k: str(v) for k, v in paths.items()}
        summary["best_encoder"] = (
            Path(paths["best_encoder"]).read_text(encoding="utf-8").strip()
        )
    elif finalize:
        have_n = len(_existing_result_keys(csv_path))
        need_n = len(EXP1_MODEL_IDS) * len(all_seeds)
        logger.info(
            "EXP-1 progress %d/%d cells; finalize deferred", have_n, need_n,
        )
        summary["progress"] = f"{have_n}/{need_n}"

    return summary


__all__ = [
    "DATASET_ID",
    "DATASET_VERSION_TAG",
    "ENCODER_MODEL_IDS",
    "EXP1_MODEL_IDS",
    "EXPERIMENT_ID",
    "SPLIT_TYPE",
    "SplitBundle",
    "cells_complete",
    "compute_encoder_pairwise_stats",
    "finalize_exp1",
    "load_ds1_split",
    "load_predictions",
    "pred_artifact_path",
    "primary_seeds",
    "run_classical_cell",
    "run_encoder_cell",
    "run_exp1",
    "save_predictions",
    "select_best_encoder",
    "write_best_encoder",
]
