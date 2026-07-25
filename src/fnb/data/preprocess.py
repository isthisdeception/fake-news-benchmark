"""EXP-P1b: dual-track text preprocessing (protocol §6).

Classical (``vCLEAN-C``): Unicode NFC → lowercase → strip URLs/HTML/mentions/
hashtags/symbols → NLTK stopwords → WordNet lemmatization.

Neural (``vCLEAN-N``): Unicode NFC → strip URLs/HTML only. No stopword removal,
no lemmatization; model-native subword tokenization happens later at train/infer
time (``model_native_subword_tokenization`` is recorded as deferred).

TF-IDF is **not** fit here — vectorizers are fit on TRAIN only in later stages
(``configs/preprocessing.yaml`` classical_track.tfidf.fit_on).
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from fnb.config import load_config, load_config_raw
from fnb.utils.io import ensure_dir, write_json, write_parquet

logger = logging.getLogger("fnb.data.preprocess")

DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_NLTK_REPORT = Path("results/nltk_resource_versions.json")

# --- Regexes (classical + shared URL/HTML) ------------------------------------
_URL_RE = re.compile(
    r"(?i)\b(?:https?://|www\.)\S+",
)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MENTION_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#\w+")
# Keep letters (any script), digits, and whitespace; drop other symbols.
_SYMBOL_RE = re.compile(r"[^\w\s]+", flags=re.UNICODE)
_WHITESPACE_RE = re.compile(r"\s+")

_NLTK_READY = False


def ensure_nltk_resources() -> None:
    """Download stopwords + WordNet if missing (idempotent)."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    import nltk

    for resource, path in (
        ("stopwords", "corpora/stopwords"),
        ("wordnet", "corpora/wordnet"),
        ("omw-1.4", "corpora/omw-1.4"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)
    _NLTK_READY = True


def get_nltk_resource_info() -> dict[str, Any]:
    """Return NLTK / stopwords / WordNet version metadata for ENVIRONMENT.md."""
    import nltk
    from nltk.corpus import stopwords, wordnet

    ensure_nltk_resources()
    info: dict[str, Any] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "download_date": date.today().isoformat(),
        "nltk_version": getattr(nltk, "__version__", "unknown"),
        "stopwords": {
            "lang": "english",
            "n_words": len(stopwords.words("english")),
        },
        "wordnet": {
            "available": True,
            # WordNet corpus has no single semver; record nltk package + presence.
            "note": "NLTK WordNet corpus (Princeton WordNet via nltk.corpus.wordnet)",
        },
    }
    try:
        # WordNet synset count is a stable fingerprint of the installed corpus.
        info["wordnet"]["n_synsets"] = len(list(wordnet.all_synsets()))
    except Exception as exc:  # pragma: no cover - defensive
        info["wordnet"]["error"] = str(exc)
    return info


def write_nltk_versions(
    path: str | Path = DEFAULT_NLTK_REPORT,
    *,
    environment_md: str | Path | None = "ENVIRONMENT.md",
) -> dict[str, Any]:
    """Write ``results/nltk_resource_versions.json`` and patch ENVIRONMENT.md markers."""
    info = get_nltk_resource_info()
    write_json(info, path)
    if environment_md is not None:
        _patch_environment_md(Path(environment_md), info)
    return info


def _patch_environment_md(path: Path, info: dict[str, Any]) -> None:
    """Fill ``<!-- NLTK_* -->`` placeholders in ENVIRONMENT.md if the file exists."""
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    replacements = {
        "<!-- NLTK_VERSION -->": str(info.get("nltk_version", "TBD")),
        "<!-- NLTK_DOWNLOAD_DATE -->": str(info.get("download_date", "TBD")),
        "<!-- NLTK_STOPWORDS_N -->": str(info.get("stopwords", {}).get("n_words", "TBD")),
        "<!-- NLTK_WORDNET_SYNSETS -->": str(info.get("wordnet", {}).get("n_synsets", "TBD")),
    }
    new = text
    for marker, value in replacements.items():
        if marker in new:
            # Replace TBD cells that still contain the marker.
            new = new.replace(marker, value)
    if new != text:
        path.write_text(new, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Atomic cleaning steps
# --------------------------------------------------------------------------- #
def step_unicode_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def step_lowercase(text: str) -> str:
    return text.lower()


def step_remove_urls(text: str) -> str:
    return _URL_RE.sub(" ", text)


def step_remove_html(text: str) -> str:
    text = _HTML_TAG_RE.sub(" ", text)
    return html.unescape(text)


def step_remove_mentions(text: str) -> str:
    return _MENTION_RE.sub(" ", text)


def step_remove_hashtags(text: str) -> str:
    return _HASHTAG_RE.sub(" ", text)


def step_remove_symbols(text: str) -> str:
    text = _SYMBOL_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def step_nltk_stopword_removal(text: str) -> str:
    ensure_nltk_resources()
    from nltk.corpus import stopwords

    stops = set(stopwords.words("english"))
    tokens = text.split()
    return " ".join(t for t in tokens if t not in stops)


def step_wordnet_lemmatization(text: str) -> str:
    ensure_nltk_resources()
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    # Default POS=noun; full POS tagging is out of scope for the frozen recipe.
    return " ".join(lemmatizer.lemmatize(t) for t in tokens)


def step_model_native_subword_tokenization(text: str) -> str:
    """No-op here — HF tokenizer runs at model time (protocol §6 neural track)."""
    return text


STEP_FUNCS: dict[str, Callable[[str], str]] = {
    "unicode_nfc": step_unicode_nfc,
    "lowercase": step_lowercase,
    "remove_urls": step_remove_urls,
    "remove_html": step_remove_html,
    "remove_mentions": step_remove_mentions,
    "remove_hashtags": step_remove_hashtags,
    "remove_symbols": step_remove_symbols,
    "nltk_stopword_removal": step_nltk_stopword_removal,
    "wordnet_lemmatization": step_wordnet_lemmatization,
    "model_native_subword_tokenization": step_model_native_subword_tokenization,
}


def apply_steps(text: str | None, steps: list[str]) -> str:
    """Apply named cleaning steps in order to one string (None/NaN → \"\")."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        text = ""
    else:
        text = str(text)
    for name in steps:
        fn = STEP_FUNCS.get(name)
        if fn is None:
            raise KeyError(f"Unknown preprocessing step {name!r}")
        text = fn(text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_text_classical(text: str | None, steps: list[str] | None = None) -> str:
    """Classical-track cleaner (defaults to frozen classical steps)."""
    if steps is None:
        steps = list(load_config("preprocessing").classical_track.steps)
    return apply_steps(text, steps)


def clean_text_neural(text: str | None, steps: list[str] | None = None) -> str:
    """Neural-track cleaner (defaults to frozen neural steps)."""
    if steps is None:
        steps = list(load_config("preprocessing").neural_track.steps)
    return apply_steps(text, steps)


# --------------------------------------------------------------------------- #
# DataFrame / pipeline
# --------------------------------------------------------------------------- #
@dataclass
class PreprocessResult:
    dataset_id: str
    track: str
    version_tag: str
    dataframe: pd.DataFrame
    output_path: Path | None = None
    n_rows: int = 0
    extras: dict[str, Any] = field(default_factory=dict)


def _clean_columns(
    df: pd.DataFrame,
    steps: list[str],
    *,
    text_cols: tuple[str, ...] = ("title", "text"),
) -> pd.DataFrame:
    out = df.copy()
    for col in text_cols:
        if col not in out.columns:
            continue
        out[col] = out[col].map(lambda v, s=steps: apply_steps(v, s))
    return out


def preprocess_dataframe(
    df: pd.DataFrame,
    *,
    track: str,
    steps: list[str],
    version_tag: str,
    dataset_id: str,
) -> PreprocessResult:
    """Apply one track's steps to title/text columns; preserve labels/ids."""
    # Guardrails from config/protocol
    if track == "neural":
        forbidden = {"nltk_stopword_removal", "wordnet_lemmatization", "lowercase"}
        bad = forbidden.intersection(steps)
        # lowercase is classical-only; neural must not include it
        if "nltk_stopword_removal" in steps or "wordnet_lemmatization" in steps:
            raise ValueError(
                f"Neural track must not include stopword/lemmatization steps; got {steps}"
            )
        _ = bad  # lowercase absence is enforced by yaml; still allow if explicitly passed in tests

    cleaned = _clean_columns(df, steps)
    return PreprocessResult(
        dataset_id=dataset_id,
        track=track,
        version_tag=version_tag,
        dataframe=cleaned,
        n_rows=len(cleaned),
    )


def preprocess_dataset(
    dataset_id: str,
    *,
    input_path: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    config_dir: str | Path | None = None,
) -> list[PreprocessResult]:
    """Read ``{DSx}_vBIN.parquet``, write classical + neural cleaned parquets."""
    raw_cfg = load_config_raw("preprocessing", config_dir)
    classical = raw_cfg["classical_track"]
    neural = raw_cfg["neural_track"]

    # Soft-validate via pydantic schema
    _ = load_config("preprocessing", config_dir)

    df = pd.read_parquet(input_path)
    results: list[PreprocessResult] = []

    for track_name, track_cfg in (
        ("classical", classical),
        ("neural", neural),
    ):
        steps = list(track_cfg["steps"])
        version_tag = str(track_cfg["version_tag"])
        if track_name == "neural":
            if track_cfg.get("stopword_removal") or track_cfg.get("lemmatization"):
                raise ValueError("neural_track forbids stopword_removal/lemmatization")
        result = preprocess_dataframe(
            df,
            track=track_name,
            steps=steps,
            version_tag=version_tag,
            dataset_id=dataset_id,
        )
        out = Path(output_dir) / f"{dataset_id}_{version_tag}.parquet"
        write_parquet(result.dataframe, out)
        result.output_path = out
        results.append(result)
        logger.info(
            "%s %s wrote %s (%d rows)",
            dataset_id,
            version_tag,
            out,
            result.n_rows,
        )
    return results


def preprocess_all(
    *,
    processed_dir: str | Path = DEFAULT_OUTPUT_DIR,
    output_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
    dataset_ids: list[str] | None = None,
    nltk_report_path: str | Path = DEFAULT_NLTK_REPORT,
    environment_md: str | Path | None = "ENVIRONMENT.md",
) -> list[PreprocessResult]:
    """Run EXP-P1b for all (or selected) datasets that have vBIN parquets."""
    processed_dir = Path(processed_dir)
    output_dir = Path(output_dir) if output_dir is not None else processed_dir
    ensure_dir(output_dir)

    # Record NLTK versions once per run (protocol S8 DoD).
    nltk_info = write_nltk_versions(nltk_report_path, environment_md=environment_md)
    logger.info("NLTK resources: %s", nltk_info)

    if dataset_ids is None:
        from fnb.config import load_config_raw as _raw

        dataset_ids = list((_raw("datasets", config_dir).get("datasets") or {}).keys())

    all_results: list[PreprocessResult] = []
    for ds_id in dataset_ids:
        vbin = processed_dir / f"{ds_id}_vBIN.parquet"
        if not vbin.is_file():
            raise FileNotFoundError(f"{ds_id}: missing {vbin} — run EXP-P1a (--stage P1a) first")
        all_results.extend(
            preprocess_dataset(
                ds_id,
                input_path=vbin,
                output_dir=output_dir,
                config_dir=config_dir,
            )
        )
    return all_results


__all__ = [
    "PreprocessResult",
    "STEP_FUNCS",
    "apply_steps",
    "clean_text_classical",
    "clean_text_neural",
    "ensure_nltk_resources",
    "get_nltk_resource_info",
    "preprocess_all",
    "preprocess_dataframe",
    "preprocess_dataset",
    "step_lowercase",
    "step_model_native_subword_tokenization",
    "step_nltk_stopword_removal",
    "step_remove_hashtags",
    "step_remove_html",
    "step_remove_mentions",
    "step_remove_symbols",
    "step_remove_urls",
    "step_unicode_nfc",
    "step_wordnet_lemmatization",
    "write_nltk_versions",
]
