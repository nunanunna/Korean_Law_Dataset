#!/usr/bin/env python3
"""
25_fine_grid_search_weighted_methods.py
=======================================
국회 법률발의안 75개 전체 75 x 75 similarity matrix를 다시 계산한 뒤,
hybrid_cleaned / weighted_field / final_ensemble의 2차 fine grid search를 수행한다.

실행 방법:
    python 25_fine_grid_search_weighted_methods.py

실행 후 확인할 파일:
    Sbert_output/fine_grid_search_report.md
    Sbert_output/fine_grid_search_results.csv
    Sbert_output/fine_grid_search_best_weights.json
    Sbert_output/topk_hybrid_cleaned_fine_best.json
    Sbert_output/topk_weighted_field_fine_best.json
    Sbert_output/topk_final_ensemble_fine_best.json

주의:
    categories / manual_categories는 similarity 계산에 사용하지 않는다.
    top-k 파일에서 전체 행렬을 역으로 채우지 않으며, SBERT/TF-IDF/조문 component를
    전체 75개 법안에 대해 재계산한다.
"""

from __future__ import annotations

import json
import math
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError as exc:  # pragma: no cover - 실행 환경 안내용
    raise ImportError(
        "[의존성 로드 단계] torch, scikit-learn, sentence-transformers가 필요합니다. "
        "해당 패키지가 설치된 Python 3.10+ 환경에서 실행하세요."
    ) from exc

from article_similarity_utils import compute_article_similarity
from bill_text_parser import split_summary_sections
from clean_text_utils import (
    extract_light_keywords,
    normalize_legal_text_for_keywords,
    normalize_legal_text_for_sbert,
)
from text_builders import (
    build_problem_proposal_text,
    build_raw_text,
    build_structured_text,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── 실행 옵션 ────────────────────────────────────────────────────────────────
STEP = 0.025
TOP_K = 10
RANDOM_SEED = 42
SEARCH_CHUNK_SIZE = 128
MODEL_NAME = "woong0322/ko-legal-sbert-finetuned"

NORMALIZE_COMPONENTS_FOR_HYBRID_CLEANED = True
NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD = True
NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE = True
NORMALIZATION_METHOD = "row_minmax"

RELEVANCE_THRESHOLD = 3.0
LEGAL_WEIGHTS = {
    "human_issue_match_0_to_2": 0.25,
    "human_target_match_0_to_2": 0.25,
    "human_effect_match_0_to_2": 0.25,
    "human_scope_match_0_to_2": 0.15,
    "human_article_match_0_to_2": 0.10,
}


# ── 경로 ─────────────────────────────────────────────────────────────────────
DATASET_JSON = Path("test_dataset/full_dataset.json")
LABEL_EXCEL = Path("Sbert_output/evaluation_full_score_pooled_llm_labeled.xlsx")
V1_HYBRID_BEST_JSON = Path("Sbert_output/hybrid_grid_search_best_weights.json")
V1_WEIGHTED_ENSEMBLE_BEST_JSON = Path("Sbert_output/weighted_and_ensemble_best_weights.json")
V1_TOPK_HYBRID_JSON = Path("Sbert_output/topk_hybrid_grid_best.json")
V1_TOPK_WEIGHTED_JSON = Path("Sbert_output/topk_weighted_field_grid_best.json")
V1_TOPK_ENSEMBLE_JSON = Path("Sbert_output/topk_final_ensemble_grid_best.json")

# label_source가 없을 때 기존 20개 gold source를 판별하는 보조 입력
GOLD_SOURCE_FALLBACK_EXCEL = Path("Sbert_output/evaluation_pooled_label_template_v2.xlsx")

OUTPUT_DIR = Path("Sbert_output")
OUTPUT_CSV = OUTPUT_DIR / "fine_grid_search_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "fine_grid_search_results.json"
OUTPUT_BEST_JSON = OUTPUT_DIR / "fine_grid_search_best_weights.json"
OUTPUT_TOPK_HYBRID = OUTPUT_DIR / "topk_hybrid_cleaned_fine_best.json"
OUTPUT_TOPK_WEIGHTED = OUTPUT_DIR / "topk_weighted_field_fine_best.json"
OUTPUT_TOPK_ENSEMBLE = OUTPUT_DIR / "topk_final_ensemble_fine_best.json"
OUTPUT_REPORT = OUTPUT_DIR / "fine_grid_search_report.md"


# ── 2차 수동 후보 ────────────────────────────────────────────────────────────
HYBRID_FINE_MANUAL_CANDIDATES = [
    {"name": "HC_v1_best", "w_cleaned": 0.90, "w_tfidf": 0.10, "w_article": 0.00},
    {"name": "HC_pure_sbert", "w_cleaned": 1.00, "w_tfidf": 0.00, "w_article": 0.00},
    {"name": "HC_sbert_975_tfidf_025", "w_cleaned": 0.975, "w_tfidf": 0.025, "w_article": 0.00},
    {"name": "HC_sbert_95_tfidf_05", "w_cleaned": 0.95, "w_tfidf": 0.05, "w_article": 0.00},
    {"name": "HC_sbert_925_tfidf_075", "w_cleaned": 0.925, "w_tfidf": 0.075, "w_article": 0.00},
    {"name": "HC_ranking_v1", "w_cleaned": 0.90, "w_tfidf": 0.05, "w_article": 0.05},
    {"name": "HC_small_article", "w_cleaned": 0.925, "w_tfidf": 0.05, "w_article": 0.025},
]

WEIGHTED_FIELD_FINE_MANUAL_CANDIDATES = [
    {
        "name": "WF_v1_best", "w_title": 0.20, "w_full": 0.20,
        "w_current": 0.10, "w_problem": 0.20, "w_proposal": 0.20, "w_article": 0.10,
    },
    {
        "name": "WF_more_title", "w_title": 0.25, "w_full": 0.20,
        "w_current": 0.10, "w_problem": 0.175, "w_proposal": 0.175, "w_article": 0.10,
    },
    {
        "name": "WF_more_article", "w_title": 0.20, "w_full": 0.175,
        "w_current": 0.10, "w_problem": 0.20, "w_proposal": 0.175, "w_article": 0.15,
    },
    {
        "name": "WF_context_heavy", "w_title": 0.25, "w_full": 0.25,
        "w_current": 0.10, "w_problem": 0.15, "w_proposal": 0.15, "w_article": 0.10,
    },
    {
        "name": "WF_balanced_fine", "w_title": 0.20, "w_full": 0.225,
        "w_current": 0.10, "w_problem": 0.20, "w_proposal": 0.175, "w_article": 0.10,
    },
]

FINAL_ENSEMBLE_FINE_MANUAL_CANDIDATES = [
    {
        "name": "FE_v1_best", "w_raw": 0.10, "w_structured": 0.10,
        "w_problem_proposal": 0.10, "w_weighted_field": 0.20,
        "w_cleaned": 0.05, "w_hybrid": 0.45,
    },
    {
        "name": "FE_more_hybrid_50", "w_raw": 0.075, "w_structured": 0.10,
        "w_problem_proposal": 0.075, "w_weighted_field": 0.20,
        "w_cleaned": 0.05, "w_hybrid": 0.50,
    },
    {
        "name": "FE_more_hybrid_55", "w_raw": 0.05, "w_structured": 0.10,
        "w_problem_proposal": 0.05, "w_weighted_field": 0.20,
        "w_cleaned": 0.05, "w_hybrid": 0.55,
    },
    {
        "name": "FE_hybrid_60_light_context", "w_raw": 0.025, "w_structured": 0.075,
        "w_problem_proposal": 0.05, "w_weighted_field": 0.20,
        "w_cleaned": 0.05, "w_hybrid": 0.60,
    },
    {
        "name": "FE_more_weighted_field", "w_raw": 0.075, "w_structured": 0.10,
        "w_problem_proposal": 0.075, "w_weighted_field": 0.275,
        "w_cleaned": 0.025, "w_hybrid": 0.45,
    },
    {
        "name": "FE_structured_stable", "w_raw": 0.075, "w_structured": 0.15,
        "w_problem_proposal": 0.075, "w_weighted_field": 0.20,
        "w_cleaned": 0.025, "w_hybrid": 0.475,
    },
]


HYBRID_WEIGHT_KEYS = ["w_cleaned", "w_tfidf", "w_article"]
WEIGHTED_WEIGHT_KEYS = [
    "w_title", "w_full", "w_current", "w_problem", "w_proposal", "w_article"
]
ENSEMBLE_WEIGHT_KEYS = [
    "w_raw", "w_structured", "w_problem_proposal",
    "w_weighted_field", "w_cleaned", "w_hybrid",
]

METRIC_NAMES = [
    "precision_at_5", "precision_at_10", "ndcg_at_10", "mrr",
    "average_relevance", "average_legal_meaning_score",
    "objective_score", "candidate_recall_objective", "ranking_objective",
    "num_evaluated_pairs", "num_unlabeled_pairs", "num_sources",
]


@dataclass
class LabelData:
    relevance_full: np.ndarray
    legal_full: np.ndarray
    relevance_gold: np.ndarray
    legal_gold: np.ndarray
    full_source_mask: np.ndarray
    gold_source_mask: np.ndarray
    full_source_ids: list[str]
    gold_source_ids: list[str]
    gold_detection: str
    num_label_pairs: int
    num_gold_pairs: int


@dataclass
class FamilySearchResult:
    method_family: str
    weight_keys: list[str]
    candidates: list[dict[str, Any]]
    full_metrics: dict[str, np.ndarray]
    gold_metrics: dict[str, np.ndarray]
    split_metrics: dict[str, dict[str, np.ndarray]]
    best_indices: dict[str, int]
    train_validation_result: dict[str, Any]
    cv_average_validation_result: dict[str, Any]


def require_file(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"[{stage}] 필수 파일이 없습니다: {path}")


def json_load(path: Path) -> Any:
    require_file(path, "JSON 입력")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def json_dump(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def safe_float(value: Any) -> float | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_bill_name(name: str) -> str:
    cleaned = str(name).strip()
    cleaned = re.sub(
        r"(?:일부개정법률안|전부개정법률안|개정법률안|법률안|일부개정안|개정안|안)$",
        "",
        cleaned,
    )
    return cleaned.strip() or str(name).strip()


def row_minmax_normalize(
    matrix: np.ndarray,
    ignore_diagonal: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """대각을 min/max 계산에서 제외하는 row-wise min-max normalization."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"[정규화 단계] 정방행렬이 아닙니다: shape={matrix.shape}")

    n = matrix.shape[0]
    active = np.ones_like(matrix, dtype=bool)
    if ignore_diagonal:
        np.fill_diagonal(active, False)
    before = matrix[active]

    normalized = np.zeros_like(matrix, dtype=np.float64)
    for row_idx in range(n):
        row_mask = active[row_idx]
        row_values = matrix[row_idx, row_mask]
        if row_values.size == 0:
            continue
        row_min = float(np.min(row_values))
        row_max = float(np.max(row_values))
        if math.isclose(row_min, row_max, rel_tol=0.0, abs_tol=1e-12):
            normalized[row_idx] = 0.0
        else:
            normalized[row_idx] = (matrix[row_idx] - row_min) / (row_max - row_min)
            normalized[row_idx] = np.clip(normalized[row_idx], 0.0, 1.0)
        if ignore_diagonal:
            normalized[row_idx, row_idx] = 0.0

    after = normalized[active]
    stats = {
        "before_min": float(np.min(before)) if before.size else 0.0,
        "before_max": float(np.max(before)) if before.size else 0.0,
        "before_mean": float(np.mean(before)) if before.size else 0.0,
        "after_min": float(np.min(after)) if after.size else 0.0,
        "after_max": float(np.max(after)) if after.size else 0.0,
        "after_mean": float(np.mean(after)) if after.size else 0.0,
    }
    return normalized, stats


def normalize_component(
    matrix: np.ndarray,
    name: str,
    enabled: bool,
    stats_store: dict[str, dict[str, Any]],
) -> np.ndarray:
    if enabled:
        normalized, stats = row_minmax_normalize(matrix, ignore_diagonal=True)
    else:
        normalized = np.asarray(matrix, dtype=np.float64).copy()
        active = ~np.eye(matrix.shape[0], dtype=bool)
        values = normalized[active]
        stats = {
            "before_min": float(np.min(values)), "before_max": float(np.max(values)),
            "before_mean": float(np.mean(values)), "after_min": float(np.min(values)),
            "after_max": float(np.max(values)), "after_mean": float(np.mean(values)),
        }
    stats_store[name] = {
        "normalization_enabled": enabled,
        "normalization_method": NORMALIZATION_METHOD if enabled else "none",
        **stats,
    }
    print(
        f"    {name}: raw({stats['before_min']:.4f}, {stats['before_max']:.4f}, "
        f"{stats['before_mean']:.4f}) -> normalized({stats['after_min']:.4f}, "
        f"{stats['after_max']:.4f}, {stats['after_mean']:.4f})"
    )
    return normalized


def validate_weights(candidate: dict[str, Any], keys: list[str], stage: str) -> None:
    missing = [key for key in keys if key not in candidate]
    if missing:
        raise ValueError(f"[{stage}] 가중치 컬럼 누락: {missing}, candidate={candidate}")
    total = sum(float(candidate[key]) for key in keys)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"[{stage}] 가중치 합이 1.0이 아닙니다: sum={total}, {candidate}")


def int_range(start: float, end: float) -> range:
    unit = int(round(STEP * 1000))
    return range(int(round(start * 1000)), int(round(end * 1000)) + 1, unit)


def make_candidates(
    ranges: list[tuple[float, float]],
    keys: list[str],
    manual_candidates: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    candidates_by_tuple: dict[tuple[float, ...], dict[str, Any]] = {}
    grid_counter = 0

    def walk(depth: int, units: list[int]) -> None:
        nonlocal grid_counter
        if depth == len(ranges):
            if sum(units) != 1000:
                return
            weights = tuple(round(value / 1000.0, 3) for value in units)
            grid_counter += 1
            candidates_by_tuple[weights] = {
                "name": f"{prefix}_grid_{grid_counter:05d}",
                **dict(zip(keys, weights)),
                "is_manual_candidate": False,
            }
            return
        for value in int_range(*ranges[depth]):
            if sum(units) + value <= 1000:
                walk(depth + 1, [*units, value])

    walk(0, [])

    for manual in manual_candidates:
        validate_weights(manual, keys, f"{prefix} 수동 후보")
        weights = tuple(round(float(manual[key]), 3) for key in keys)
        record = candidates_by_tuple.get(weights, {**dict(zip(keys, weights))})
        record.update(
            name=manual["name"],
            is_manual_candidate=True,
        )
        candidates_by_tuple[weights] = record

    candidates = list(candidates_by_tuple.values())
    candidates.sort(key=lambda item: tuple(float(item[key]) for key in keys))
    for candidate in candidates:
        validate_weights(candidate, keys, f"{prefix} 후보 생성")
    return candidates


def load_v1_weights() -> dict[str, dict[str, float]]:
    """1차 JSON을 읽고 알려진 1차 best와 일치하는지 검증한다."""
    hybrid_json = json_load(V1_HYBRID_BEST_JSON)
    weighted_json = json_load(V1_WEIGHTED_ENSEMBLE_BEST_JSON)
    for path in (V1_TOPK_HYBRID_JSON, V1_TOPK_WEIGHTED_JSON, V1_TOPK_ENSEMBLE_JSON):
        require_file(path, "1차 top-k 비교 입력")

    loaded = {
        "hybrid": hybrid_json["best_by_objective_score"]["weights"],
        "weighted": weighted_json["weighted_field"]["best_by_objective_score"]["weights"],
        "ensemble": weighted_json["final_ensemble"]["best_by_objective_score"]["weights"],
    }
    expected = {
        "hybrid": {"w_cleaned": 0.90, "w_tfidf": 0.10, "w_article": 0.00},
        "weighted": {
            "w_title": 0.20, "w_full": 0.20, "w_current": 0.10,
            "w_problem": 0.20, "w_proposal": 0.20, "w_article": 0.10,
        },
        "ensemble": {
            "w_raw": 0.10, "w_structured": 0.10, "w_problem_proposal": 0.10,
            "w_weighted_field": 0.20, "w_cleaned": 0.05, "w_hybrid": 0.45,
        },
    }
    for family, weights in loaded.items():
        for key, value in expected[family].items():
            if not math.isclose(float(weights[key]), value, abs_tol=1e-9):
                raise ValueError(
                    f"[1차 best 검증] {family}.{key}: 파일={weights[key]}, 기대값={value}"
                )
    return {family: {k: float(v) for k, v in values.items()} for family, values in loaded.items()}


def build_component_matrices(
    bills: list[dict[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    n = len(bills)
    sections = [split_summary_sections(str(bill.get("summary", ""))) for bill in bills]

    raw_texts: list[str] = []
    structured_texts: list[str] = []
    problem_proposal_texts: list[str] = []
    cleaned_texts: list[str] = []
    tfidf_texts: list[str] = []
    title_texts: list[str] = []
    full_texts: list[str] = []
    current_texts: list[str] = []
    problem_texts: list[str] = []
    proposal_texts: list[str] = []

    for bill, parsed in zip(bills, sections):
        summary = str(bill.get("summary", ""))
        bill_name = str(bill.get("bill_name", ""))
        problem = str(parsed.get("problem", "")).strip()
        proposal = str(parsed.get("proposal", "")).strip()
        combined = f"{problem} {proposal}".strip()

        raw_texts.append(build_raw_text(bill))
        structured_texts.append(build_structured_text(bill, parsed))
        problem_proposal_texts.append(build_problem_proposal_text(bill, parsed))

        cleaned = normalize_legal_text_for_sbert(combined)
        if len(cleaned) < 5:
            cleaned = normalize_legal_text_for_sbert(f"{bill_name} {summary}".strip())
        cleaned_texts.append(cleaned)

        keyword_cleaned = normalize_legal_text_for_keywords(combined)
        keywords = extract_light_keywords(keyword_cleaned)
        if len(keywords.strip()) < 3:
            keyword_cleaned = normalize_legal_text_for_keywords(f"{bill_name} {summary}".strip())
            keywords = extract_light_keywords(keyword_cleaned)
        tfidf_texts.append(keywords)

        title_texts.append(normalize_bill_name(bill_name))
        full_texts.append(f"{bill_name} {summary}".strip() or "[내용 없음]")
        current_texts.append(str(parsed.get("current_law", "")).strip() or "[내용 없음]")
        problem_texts.append(problem or "[내용 없음]")
        proposal_texts.append(proposal or "[내용 없음]")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[3/9] SBERT 모델 로드: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)

    def encode(texts: list[str], description: str) -> np.ndarray:
        print(f"    {description} 전체 {n}개 인코딩")
        embeddings = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_tensor=True,
        )
        return (embeddings @ embeddings.T).detach().cpu().numpy().astype(np.float64)

    matrices = {
        "raw_score": encode(raw_texts, "raw_score"),
        "structured_score": encode(structured_texts, "structured_score"),
        "problem_proposal_score": encode(problem_proposal_texts, "problem_proposal_score"),
        "cleaned_problem_proposal_score": encode(cleaned_texts, "cleaned_problem_proposal_score"),
        "title_law_name_similarity": encode(title_texts, "title_law_name_similarity"),
        "full_text_similarity": encode(full_texts, "full_text_similarity"),
        "current_law_similarity": encode(current_texts, "current_law_similarity"),
        "problem_similarity": encode(problem_texts, "problem_similarity"),
        "proposal_similarity": encode(proposal_texts, "proposal_similarity"),
    }

    print("    keyword_tfidf_score 전체 행렬 계산")
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(tfidf_texts)
    matrices["keyword_tfidf_score"] = cosine_similarity(tfidf_matrix).astype(np.float64)

    print("    article_similarity 전체 행렬 계산")
    article = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        article[i, i] = 1.0
        for j in range(i + 1, n):
            value = compute_article_similarity(
                sections[i].get("article_numbers", []),
                sections[j].get("article_numbers", []),
            )
            article[i, j] = article[j, i] = float(value)
    matrices["article_similarity"] = article
    return matrices, sections


def validate_matrices(
    matrices: dict[str, np.ndarray],
    bill_ids: list[str],
    matrix_bill_ids: dict[str, list[str]],
) -> dict[str, Any]:
    expected_shape = (len(bill_ids), len(bill_ids))
    for name, matrix in matrices.items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"[행렬 shape 검증] {name}: {matrix.shape}, 기대값={expected_shape}"
            )
        if not np.all(np.isfinite(matrix)):
            raise ValueError(f"[행렬 값 검증] {name}에 NaN/Inf가 있습니다.")
        if matrix_bill_ids.get(name) != bill_ids:
            raise ValueError(f"[법안 순서 검증] {name}의 source/target bill_id 순서가 다릅니다.")
    return {
        "all_matrix_shapes_identical": True,
        "all_bill_id_orders_identical": True,
        "shape": list(expected_shape),
        "bill_id_count": len(bill_ids),
        "matrix_names": sorted(matrices),
    }


def detect_gold_rows(df: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    candidates = [
        "label_source", "relevance_label_source", "human_label_source",
        "annotation_source", "label_type",
    ]
    for column in candidates:
        if column in df.columns:
            values = df[column].fillna("").astype(str).str.strip().str.lower()
            mask = values.str.contains(r"gold|human|manual|reference|copied|사람|휴먼", regex=True)
            if mask.any():
                return mask, column
    return None, None


def load_labels(bill_ids: list[str]) -> LabelData:
    require_file(LABEL_EXCEL, "평가 라벨 로드")
    try:
        df = pd.read_excel(LABEL_EXCEL)
    except Exception as exc:
        raise RuntimeError(f"[평가 라벨 로드] Excel 읽기 실패: {LABEL_EXCEL}: {exc}") from exc

    required = ["source_bill_id", "target_bill_id", "human_relevance_0_to_4", *LEGAL_WEIGHTS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"[평가 컬럼 검증] {LABEL_EXCEL}에 컬럼이 없습니다: {missing}")

    id_to_index = {bill_id: idx for idx, bill_id in enumerate(bill_ids)}
    n = len(bill_ids)
    relevance = np.full((n, n), np.nan, dtype=np.float64)
    legal = np.full((n, n), np.nan, dtype=np.float64)

    gold_row_mask, gold_column = detect_gold_rows(df)
    gold_pair_keys: set[tuple[str, str]] = set()
    gold_source_ids: set[str] = set()

    for row_number, row in df.iterrows():
        excel_row = row_number + 2
        source_id = str(row["source_bill_id"]).strip()
        target_id = str(row["target_bill_id"]).strip()
        if source_id not in id_to_index or target_id not in id_to_index:
            raise ValueError(
                f"[평가 bill_id 검증] {LABEL_EXCEL} row {excel_row}: "
                f"데이터셋에 없는 pair=({source_id}, {target_id})"
            )
        rel = safe_float(row["human_relevance_0_to_4"])
        if rel is None:
            continue
        if not 0.0 <= rel <= 4.0:
            raise ValueError(
                f"[relevance 범위 검증] {LABEL_EXCEL} row {excel_row}, "
                f"human_relevance_0_to_4={rel}; 허용 범위는 0~4입니다."
            )

        legal_values: dict[str, float] = {}
        for column in LEGAL_WEIGHTS:
            value = safe_float(row[column])
            if value is not None and not 0.0 <= value <= 2.0:
                raise ValueError(
                    f"[세부 점수 범위 검증] {LABEL_EXCEL} row {excel_row}, "
                    f"{column}={value}; 허용 범위는 0~2입니다."
                )
            if value is not None:
                legal_values[column] = value

        i, j = id_to_index[source_id], id_to_index[target_id]
        if np.isfinite(relevance[i, j]) and not math.isclose(relevance[i, j], rel):
            raise ValueError(
                f"[중복 라벨 검증] pair=({source_id}, {target_id})에 상충하는 relevance가 있습니다."
            )
        relevance[i, j] = rel
        if len(legal_values) == len(LEGAL_WEIGHTS):
            weighted = sum(LEGAL_WEIGHTS[key] * legal_values[key] for key in LEGAL_WEIGHTS)
            legal[i, j] = weighted / 2.0 * 100.0

        if gold_row_mask is not None and bool(gold_row_mask.iloc[row_number]):
            gold_pair_keys.add((source_id, target_id))
            gold_source_ids.add(source_id)

    if gold_row_mask is not None and gold_pair_keys:
        gold_detection = f"'{gold_column}' 컬럼의 gold/human/manual 계열 값"
        relevance_gold = np.full_like(relevance, np.nan)
        legal_gold = np.full_like(legal, np.nan)
        for source_id, target_id in gold_pair_keys:
            i, j = id_to_index[source_id], id_to_index[target_id]
            relevance_gold[i, j] = relevance[i, j]
            legal_gold[i, j] = legal[i, j]
    else:
        if not GOLD_SOURCE_FALLBACK_EXCEL.exists():
            raise FileNotFoundError(
                "[gold-only 판별] label_source 컬럼이 없고 기존 20개 source를 판별할 "
                f"fallback 파일도 없습니다: {GOLD_SOURCE_FALLBACK_EXCEL}"
            )
        fallback = pd.read_excel(GOLD_SOURCE_FALLBACK_EXCEL, usecols=["source_bill_id"])
        gold_source_ids = set(fallback["source_bill_id"].dropna().astype(str).str.strip())
        unknown = gold_source_ids - set(bill_ids)
        if unknown:
            raise ValueError(f"[gold-only source 검증] 데이터셋에 없는 source ID: {sorted(unknown)}")
        relevance_gold = relevance.copy()
        legal_gold = legal.copy()
        gold_detection = (
            "label_source 컬럼 없음; 기존 evaluation_pooled_label_template_v2.xlsx의 "
            f"{len(gold_source_ids)}개 source subset"
        )

    full_source_ids = [bill_id for bill_id in bill_ids if np.isfinite(relevance[id_to_index[bill_id]]).any()]
    gold_source_ids_sorted = sorted(gold_source_ids)
    full_mask = np.array([bill_id in set(full_source_ids) for bill_id in bill_ids], dtype=bool)
    gold_mask = np.array([bill_id in gold_source_ids for bill_id in bill_ids], dtype=bool)
    if len(full_source_ids) != len(bill_ids):
        raise ValueError(
            f"[full-label source 검증] {len(bill_ids)}개 중 {len(full_source_ids)}개만 라벨이 있습니다."
        )
    if not gold_source_ids_sorted:
        raise ValueError("[gold-only source 검증] gold source를 한 개도 찾지 못했습니다.")

    return LabelData(
        relevance_full=relevance,
        legal_full=legal,
        relevance_gold=relevance_gold,
        legal_gold=legal_gold,
        full_source_mask=full_mask,
        gold_source_mask=gold_mask,
        full_source_ids=sorted(full_source_ids),
        gold_source_ids=gold_source_ids_sorted,
        gold_detection=gold_detection,
        num_label_pairs=int(np.isfinite(relevance).sum()),
        num_gold_pairs=int(np.isfinite(relevance_gold[gold_mask]).sum()),
    )


def deterministic_topk_indices(
    matrices: np.ndarray,
    bill_ids: list[str],
    k: int = TOP_K,
) -> np.ndarray:
    """similarity desc, tie 시 target_bill_id asc로 top-k index를 반환한다."""
    if matrices.ndim == 2:
        matrices = matrices[None, :, :]
    candidate_count, n, n2 = matrices.shape
    if n != n2 or n != len(bill_ids):
        raise ValueError(f"[top-k 단계] matrix shape/order 불일치: {matrices.shape}")
    if k > n - 1:
        raise ValueError(f"[top-k 단계] k={k}가 가능한 target 수 {n - 1}보다 큽니다.")

    scores = matrices.copy()
    diag = np.arange(n)
    scores[:, diag, diag] = -np.inf
    target_order = np.argsort(np.asarray(bill_ids, dtype=str), kind="stable")
    ordered_scores = scores[:, :, target_order]
    ranked_positions = np.argsort(-ordered_scores, axis=2, kind="stable")[:, :, :k]
    top_indices = target_order[ranked_positions]

    source_indices = np.arange(n)[None, :, None]
    if np.any(top_indices == source_indices):
        raise RuntimeError("[top-k 자기 자신 검증] diagonal이 top-k에 포함되었습니다.")
    if top_indices.shape != (candidate_count, n, k):
        raise RuntimeError(f"[top-k 개수 검증] 잘못된 shape: {top_indices.shape}")
    return top_indices


def metrics_for_topk(
    top_indices: np.ndarray,
    relevance_matrix: np.ndarray,
    legal_matrix: np.ndarray,
    source_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    """후보 chunk 전체의 지표를 source 단위 macro 평균으로 벡터 계산한다."""
    selected_sources = np.flatnonzero(source_mask)
    if selected_sources.size == 0:
        raise ValueError("[평가 단계] 선택된 source가 없습니다.")
    selected_top = top_indices[:, selected_sources, :]
    source_grid = selected_sources[None, :, None]
    rel = relevance_matrix[source_grid, selected_top]
    legal = legal_matrix[source_grid, selected_top]
    judged = np.isfinite(rel)
    relevant = judged & (rel >= RELEVANCE_THRESHOLD)
    candidate_count = top_indices.shape[0]

    def macro_precision(limit: int) -> np.ndarray:
        denom = judged[:, :, :limit].sum(axis=2)
        numer = relevant[:, :, :limit].sum(axis=2)
        values = np.divide(
            numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom > 0
        )
        valid = denom > 0
        return np.divide(
            (values * valid).sum(axis=1), valid.sum(axis=1),
            out=np.zeros(candidate_count, dtype=np.float64), where=valid.sum(axis=1) > 0,
        )

    p5 = macro_precision(5)
    p10 = macro_precision(10)

    safe_rel = np.where(judged, rel, 0.0)
    discounts = 1.0 / np.log2(np.arange(2, TOP_K + 2, dtype=np.float64))
    dcg = ((np.power(2.0, safe_rel) - 1.0) * discounts).sum(axis=2)
    ideal_rel = np.sort(safe_rel, axis=2)[:, :, ::-1]
    idcg = ((np.power(2.0, ideal_rel) - 1.0) * discounts).sum(axis=2)
    ndcg_source = np.divide(dcg, idcg, out=np.zeros_like(dcg), where=idcg > 0)
    source_valid = judged.any(axis=2)
    ndcg = np.divide(
        (ndcg_source * source_valid).sum(axis=1), source_valid.sum(axis=1),
        out=np.zeros(candidate_count, dtype=np.float64), where=source_valid.sum(axis=1) > 0,
    )

    positions = np.arange(1, TOP_K + 1, dtype=np.float64)[None, None, :]
    reciprocal = np.where(relevant, 1.0 / positions, 0.0)
    mrr_source = reciprocal.max(axis=2)
    mrr = np.divide(
        (mrr_source * source_valid).sum(axis=1), source_valid.sum(axis=1),
        out=np.zeros(candidate_count, dtype=np.float64), where=source_valid.sum(axis=1) > 0,
    )

    evaluated_pairs = judged.sum(axis=(1, 2)).astype(np.float64)
    average_relevance = np.divide(
        np.where(judged, rel, 0.0).sum(axis=(1, 2)), evaluated_pairs,
        out=np.zeros(candidate_count, dtype=np.float64), where=evaluated_pairs > 0,
    )
    legal_judged = np.isfinite(legal) & judged
    legal_count = legal_judged.sum(axis=(1, 2))
    average_legal = np.divide(
        np.where(legal_judged, legal, 0.0).sum(axis=(1, 2)), legal_count,
        out=np.full(candidate_count, np.nan, dtype=np.float64), where=legal_count > 0,
    )

    objective = 0.50 * p5 + 0.30 * ndcg + 0.20 * mrr
    candidate_recall = 0.60 * p10 + 0.25 * p5 + 0.15 * average_relevance
    ranking = 0.45 * ndcg + 0.35 * mrr + 0.20 * p5

    total_slots = float(selected_sources.size * TOP_K)
    return {
        "precision_at_5": p5,
        "precision_at_10": p10,
        "ndcg_at_10": ndcg,
        "mrr": mrr,
        "average_relevance": average_relevance,
        "average_legal_meaning_score": average_legal,
        "objective_score": objective,
        "candidate_recall_objective": candidate_recall,
        "ranking_objective": ranking,
        "num_evaluated_pairs": evaluated_pairs,
        "num_unlabeled_pairs": np.full(candidate_count, total_slots) - evaluated_pairs,
        "num_sources": source_valid.sum(axis=1).astype(np.float64),
    }


def metric_record(metrics: dict[str, np.ndarray], index: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    count_names = {"num_evaluated_pairs", "num_unlabeled_pairs", "num_sources"}
    for name in METRIC_NAMES:
        value = float(metrics[name][index])
        if name in count_names:
            result[name] = int(round(value))
        elif math.isnan(value):
            result[name] = None
        elif name == "average_legal_meaning_score":
            result[name] = round(value, 2)
        else:
            result[name] = round(value, 4)
    return result


def average_metric_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    averaged: dict[str, Any] = {}
    for name in METRIC_NAMES:
        values = [record[name] for record in records if record.get(name) is not None]
        if not values:
            averaged[name] = None
        elif name == "average_legal_meaning_score":
            averaged[name] = round(float(np.mean(values)), 2)
        else:
            averaged[name] = round(float(np.mean(values)), 4)
    return averaged


def build_source_splits(bill_ids: list[str], full_source_ids: list[str]) -> dict[str, np.ndarray]:
    sorted_sources = sorted(full_source_ids)
    shuffled = sorted_sources.copy()
    random.Random(RANDOM_SEED).shuffle(shuffled)
    split_at = int(len(shuffled) * 0.8)
    train_ids = set(shuffled[:split_at])
    validation_ids = set(shuffled[split_at:])

    splits: dict[str, np.ndarray] = {
        "train": np.array([bill_id in train_ids for bill_id in bill_ids], dtype=bool),
        "validation": np.array([bill_id in validation_ids for bill_id in bill_ids], dtype=bool),
    }
    folds = np.array_split(np.asarray(shuffled, dtype=object), 5)
    all_ids = set(shuffled)
    for fold_index, fold_values in enumerate(folds):
        fold_validation = set(str(value) for value in fold_values.tolist())
        fold_train = all_ids - fold_validation
        splits[f"fold_{fold_index}_train"] = np.array(
            [bill_id in fold_train for bill_id in bill_ids], dtype=bool
        )
        splits[f"fold_{fold_index}_validation"] = np.array(
            [bill_id in fold_validation for bill_id in bill_ids], dtype=bool
        )
    return splits


def search_family(
    method_family: str,
    candidates: list[dict[str, Any]],
    weight_keys: list[str],
    component_matrices: list[np.ndarray],
    bill_ids: list[str],
    labels: LabelData,
    source_splits: dict[str, np.ndarray],
) -> FamilySearchResult:
    print(f"[탐색] {method_family}: {len(candidates):,}개 후보")
    component_stack = np.stack(component_matrices, axis=0).astype(np.float64)
    weight_matrix = np.asarray(
        [[float(candidate[key]) for key in weight_keys] for candidate in candidates],
        dtype=np.float64,
    )
    candidate_count = len(candidates)

    def allocate() -> dict[str, np.ndarray]:
        return {name: np.empty(candidate_count, dtype=np.float64) for name in METRIC_NAMES}

    full_metrics = allocate()
    gold_metrics = allocate()
    split_metrics = {name: allocate() for name in source_splits}

    for start in range(0, candidate_count, SEARCH_CHUNK_SIZE):
        end = min(start + SEARCH_CHUNK_SIZE, candidate_count)
        combined = np.einsum(
            "ck,kij->cij", weight_matrix[start:end], component_stack, optimize=True
        )
        top_indices = deterministic_topk_indices(combined, bill_ids, TOP_K)
        evaluations = {
            "full": metrics_for_topk(
                top_indices, labels.relevance_full, labels.legal_full, labels.full_source_mask
            ),
            "gold": metrics_for_topk(
                top_indices, labels.relevance_gold, labels.legal_gold, labels.gold_source_mask
            ),
        }
        for split_name, split_mask in source_splits.items():
            evaluations[split_name] = metrics_for_topk(
                top_indices, labels.relevance_full, labels.legal_full, split_mask
            )

        for metric in METRIC_NAMES:
            full_metrics[metric][start:end] = evaluations["full"][metric]
            gold_metrics[metric][start:end] = evaluations["gold"][metric]
            for split_name in source_splits:
                split_metrics[split_name][metric][start:end] = evaluations[split_name][metric]
        if end == candidate_count or end % 1024 == 0:
            print(f"    진행: {end:,}/{candidate_count:,}")

    best_indices = {
        "best_by_objective_score": int(np.argmax(full_metrics["objective_score"])),
        "best_by_candidate_recall_objective": int(
            np.argmax(full_metrics["candidate_recall_objective"])
        ),
        "best_by_ranking_objective": int(np.argmax(full_metrics["ranking_objective"])),
    }

    train_best_index = int(np.argmax(split_metrics["train"]["objective_score"]))
    train_validation_result = {
        "selection_scope": "full_label_train_sources",
        "random_seed": RANDOM_SEED,
        "train_source_count": int(source_splits["train"].sum()),
        "validation_source_count": int(source_splits["validation"].sum()),
        "train_best_weights": {
            key: float(candidates[train_best_index][key]) for key in weight_keys
        },
        "train_metrics": metric_record(split_metrics["train"], train_best_index),
        "validation_metrics": metric_record(split_metrics["validation"], train_best_index),
    }

    fold_records = []
    validation_records = []
    for fold_index in range(5):
        train_name = f"fold_{fold_index}_train"
        validation_name = f"fold_{fold_index}_validation"
        best_index = int(np.argmax(split_metrics[train_name]["objective_score"]))
        validation_metrics = metric_record(split_metrics[validation_name], best_index)
        validation_records.append(validation_metrics)
        fold_records.append({
            "fold": fold_index + 1,
            "train_source_count": int(source_splits[train_name].sum()),
            "validation_source_count": int(source_splits[validation_name].sum()),
            "train_best_weights": {
                key: float(candidates[best_index][key]) for key in weight_keys
            },
            "train_metrics": metric_record(split_metrics[train_name], best_index),
            "validation_metrics": validation_metrics,
        })
    cv_result = {
        "selection_scope": "full_label_train_sources",
        "random_seed": RANDOM_SEED,
        "num_folds": 5,
        "folds": fold_records,
        "average_validation_metrics": average_metric_records(validation_records),
    }

    return FamilySearchResult(
        method_family=method_family,
        weight_keys=weight_keys,
        candidates=candidates,
        full_metrics=full_metrics,
        gold_metrics=gold_metrics,
        split_metrics=split_metrics,
        best_indices=best_indices,
        train_validation_result=train_validation_result,
        cv_average_validation_result=cv_result,
    )


def weighted_matrix(
    matrices: Iterable[np.ndarray],
    weights: dict[str, float],
    keys: list[str],
) -> np.ndarray:
    matrix_list = list(matrices)
    return np.einsum(
        "k,kij->ij",
        np.asarray([weights[key] for key in keys], dtype=np.float64),
        np.stack(matrix_list, axis=0),
        optimize=True,
    )


def family_best_payload(result: FamilySearchResult) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for best_name, index in result.best_indices.items():
        candidate = result.candidates[index]
        payload[best_name] = {
            "candidate_name": candidate["name"],
            "selection_scope": "full_label",
            "weights": {key: float(candidate[key]) for key in result.weight_keys},
            "metrics_full_label": metric_record(result.full_metrics, index),
            "metrics_gold_only": metric_record(result.gold_metrics, index),
        }
    payload["train_validation_result"] = result.train_validation_result
    payload["cv_average_validation_result"] = result.cv_average_validation_result
    return payload


def family_rows(
    result: FamilySearchResult,
    v1_weights: dict[str, float],
    normalization_enabled: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(result.candidates):
        weights = {key: float(candidate[key]) for key in result.weight_keys}
        is_v1 = all(
            math.isclose(weights[key], float(v1_weights[key]), abs_tol=1e-9)
            for key in result.weight_keys
        )
        for scope, metrics in (
            ("full_label", result.full_metrics),
            ("gold_only", result.gold_metrics),
        ):
            record = metric_record(metrics, index)
            rows.append({
                "method_family": result.method_family,
                "candidate_name": candidate["name"],
                "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "label_scope": scope,
                **{name: record[name] for name in METRIC_NAMES if name != "num_unlabeled_pairs"},
                "normalization_enabled": normalization_enabled,
                "is_manual_candidate": bool(candidate["is_manual_candidate"]),
                "is_v1_best_candidate": is_v1,
            })
    return rows


def evaluate_single_matrix(
    matrix: np.ndarray,
    bill_ids: list[str],
    labels: LabelData,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    top_indices = deterministic_topk_indices(matrix, bill_ids, TOP_K)
    full = metrics_for_topk(
        top_indices, labels.relevance_full, labels.legal_full, labels.full_source_mask
    )
    gold = metrics_for_topk(
        top_indices, labels.relevance_gold, labels.legal_gold, labels.gold_source_mask
    )
    return metric_record(full, 0), metric_record(gold, 0), top_indices[0]


def baseline_rows_and_payload(
    baseline_matrices: dict[str, tuple[np.ndarray, dict[str, float], bool]],
    bill_ids: list[str],
    labels: LabelData,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {}
    for name, (matrix, weights, normalized) in baseline_matrices.items():
        full, gold, _ = evaluate_single_matrix(matrix, bill_ids, labels)
        payload[name] = {
            "weights": weights,
            "metrics_full_label": full,
            "metrics_gold_only": gold,
            "normalization_enabled": normalized,
        }
        for scope, metrics in (("full_label", full), ("gold_only", gold)):
            rows.append({
                "method_family": "baseline",
                "candidate_name": name,
                "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "label_scope": scope,
                **{key: metrics[key] for key in METRIC_NAMES if key != "num_unlabeled_pairs"},
                "normalization_enabled": normalized,
                "is_manual_candidate": False,
                "is_v1_best_candidate": name.endswith("best_v1"),
            })
    return rows, payload


def topk_json_records(
    ranking_matrix: np.ndarray,
    raw_weighted_matrix: np.ndarray,
    bills: list[dict[str, Any]],
    method: str,
    weights: dict[str, float],
    normalized_components: dict[str, np.ndarray],
    raw_components: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    bill_ids = [str(bill["bill_id"]) for bill in bills]
    top_indices = deterministic_topk_indices(ranking_matrix, bill_ids, TOP_K)[0]
    records: list[dict[str, Any]] = []
    for source_index, targets in enumerate(top_indices):
        if len(targets) != TOP_K:
            raise RuntimeError(
                f"[top-k 개수 검증] source={bill_ids[source_index]}: {len(targets)}개"
            )
        for rank, target_index in enumerate(targets, start=1):
            if source_index == int(target_index):
                raise RuntimeError(
                    f"[top-k diagonal 검증] source={bill_ids[source_index]}, rank={rank}"
                )
            records.append({
                "source_bill_id": bill_ids[source_index],
                "source_bill_name": str(bills[source_index]["bill_name"]),
                "target_bill_id": bill_ids[int(target_index)],
                "target_bill_name": str(bills[int(target_index)]["bill_name"]),
                "rank": rank,
                "similarity": round(float(ranking_matrix[source_index, target_index]), 6),
                "similarity_normalized_weighted": round(
                    float(ranking_matrix[source_index, target_index]), 6
                ),
                "similarity_raw_weighted": round(
                    float(raw_weighted_matrix[source_index, target_index]), 6
                ),
                "method": method,
                "component_scores_normalized": {
                    name: round(float(matrix[source_index, target_index]), 6)
                    for name, matrix in normalized_components.items()
                },
                "component_scores_raw": {
                    name: round(float(matrix[source_index, target_index]), 6)
                    for name, matrix in raw_components.items()
                },
                "weights": weights,
            })
    expected = len(bills) * TOP_K
    if len(records) != expected:
        raise RuntimeError(f"[top-k 저장 검증] {len(records)}개 생성, 기대값={expected}")
    return records


def improvement_text(new_value: float, old_value: float) -> str:
    delta = new_value - old_value
    label = "개선" if delta > 1e-12 else "하락" if delta < -1e-12 else "동률"
    return f"{label} ({delta:+.4f})"


def weights_markdown(weights: dict[str, Any]) -> str:
    return ", ".join(f"{key}={float(value):.3f}" for key, value in weights.items())


def metrics_markdown(metrics: dict[str, Any]) -> str:
    return (
        f"P@5={metrics['precision_at_5']:.4f}, P@10={metrics['precision_at_10']:.4f}, "
        f"nDCG@10={metrics['ndcg_at_10']:.4f}, MRR={metrics['mrr']:.4f}, "
        f"AvgRel={metrics['average_relevance']:.4f}, "
        f"Legal={metrics['average_legal_meaning_score']}"
    )


def write_report(
    best_payload: dict[str, Any],
    normalization_stats: dict[str, Any],
    label_data: LabelData,
    candidate_counts: dict[str, int],
) -> None:
    hc = best_payload["hybrid_cleaned_fine"]["best_by_objective_score"]
    wf = best_payload["weighted_field_fine"]["best_by_objective_score"]
    fe = best_payload["final_ensemble_fine"]["best_by_objective_score"]
    baselines = best_payload["baselines"]

    lines = [
        "# 2차 Fine Grid Search 보고서",
        "",
        "## 1. 실험 목적",
        "",
        "1차 최적 가중치 주변을 0.025 간격으로 더 촘촘하게 탐색했다. 특히 최적 가중치가 "
        "탐색 범위 경계값에 붙어 있었기 때문에 범위를 확장한 2차 fine search를 수행했다.",
        "",
        "## 2. 1차 Grid Search 결과 요약",
        "",
        "- hybrid_cleaned: cleaned=0.90, TF-IDF=0.10, article=0.00",
        "- weighted_field: title=0.20, full=0.20, current=0.10, problem=0.20, proposal=0.20, article=0.10",
        "- final_ensemble: raw=0.10, structured=0.10, problem_proposal=0.10, weighted_field=0.20, cleaned=0.05, hybrid=0.45",
        "",
        "## 3. 2차 탐색 범위 확장 이유",
        "",
        "hybrid의 cleaned 상한을 1.00, final ensemble의 hybrid 상한을 0.60으로 확장했다. "
        "weighted field의 title/current/article도 1차 경계 바깥을 확인하도록 넓혔고 article=0.00은 유지했다.",
        "",
        "## 4. 평가 데이터 설명",
        "",
        f"- 전체 데이터: 75개 법안, full-label {label_data.num_label_pairs:,}개 라벨 쌍",
        f"- gold-only: {len(label_data.gold_source_ids)}개 source, {label_data.num_gold_pairs:,}개 라벨 쌍",
        f"- gold 판별 방식: {label_data.gold_detection}",
        "- full-label은 gold/LLM 혼합 라벨 기준이므로 보조 평가로 해석해야 하며, 최종 판단에서는 gold-only가 더 중요하다.",
        "- 평가되지 않은 추천 쌍은 similarity를 0으로 채우지 않았으며, 지표 집계에서 unjudged로 제외했다.",
        "",
        "## 5. Component Normalization 정책",
        "",
        f"모든 family에 `{NORMALIZATION_METHOD}`를 적용했다. 대각은 min/max 계산에서 제외했고 "
        "row 내 min=max인 경우 해당 row를 0으로 처리했다. 원본 및 정규화 통계는 결과 JSON에 저장했다.",
        "",
        "## 6. hybrid_cleaned 2차 탐색 결과",
        "",
        f"- 후보 수: {candidate_counts['hybrid_cleaned_fine']:,}",
        f"- 최종 가중치: {weights_markdown(hc['weights'])}",
        f"- full-label: {metrics_markdown(hc['metrics_full_label'])}",
        f"- gold-only: {metrics_markdown(hc['metrics_gold_only'])}",
        "",
        "## 7. weighted_field 2차 탐색 결과",
        "",
        f"- 후보 수: {candidate_counts['weighted_field_fine']:,}",
        f"- 최종 가중치: {weights_markdown(wf['weights'])}",
        f"- full-label: {metrics_markdown(wf['metrics_full_label'])}",
        f"- gold-only: {metrics_markdown(wf['metrics_gold_only'])}",
        "",
        "## 8. final_ensemble 2차 탐색 결과",
        "",
        f"- 후보 수: {candidate_counts['final_ensemble_fine']:,}",
        f"- 최종 가중치: {weights_markdown(fe['weights'])}",
        f"- full-label: {metrics_markdown(fe['metrics_full_label'])}",
        f"- gold-only: {metrics_markdown(fe['metrics_gold_only'])}",
        "",
        "## 9. 1차 best 대비 개선 여부",
        "",
        "아래 v1 수치는 과거 리포트 값을 그대로 복사한 것이 아니라, 공정한 비교를 위해 v1 가중치를 "
        "2차와 동일한 전체 75×75 행렬, row-minmax 및 평가 규칙으로 다시 계산한 값이다.",
        "",
    ]

    comparison = [
        ("hybrid_cleaned", hc, baselines["hybrid_cleaned_best_v1"]),
        ("weighted_field", wf, baselines["weighted_field_best_v1"]),
        ("final_ensemble", fe, baselines["final_ensemble_best_v1"]),
    ]
    lines.extend([
        "| Method | v1 objective | v2 objective | full-label 변화 | gold-only 변화 |",
        "|---|---:|---:|---|---|",
    ])
    for name, new, old in comparison:
        old_full = old["metrics_full_label"]["objective_score"]
        new_full = new["metrics_full_label"]["objective_score"]
        old_gold = old["metrics_gold_only"]["objective_score"]
        new_gold = new["metrics_gold_only"]["objective_score"]
        lines.append(
            f"| {name} | {old_full:.4f} | {new_full:.4f} | "
            f"{improvement_text(new_full, old_full)} | {improvement_text(new_gold, old_gold)} |"
        )

    fe_full = fe["metrics_full_label"]["objective_score"]
    best_single_full = max(
        hc["metrics_full_label"]["objective_score"],
        wf["metrics_full_label"]["objective_score"],
    )
    fe_gold = fe["metrics_gold_only"]["objective_score"]
    best_single_gold = max(
        hc["metrics_gold_only"]["objective_score"],
        wf["metrics_gold_only"]["objective_score"],
    )
    lines.extend([
        "",
        "final_ensemble이 단일 메소드보다 실제로 개선되는지 별도로 확인해야 한다. 이번 결과에서 "
        f"full-label 기준 변화는 {fe_full - best_single_full:+.4f}, gold-only 기준 변화는 "
        f"{fe_gold - best_single_gold:+.4f}이다.",
        "",
        "## 10. full-label 결과와 gold-only 결과 비교",
        "",
        "full-label은 75개 법안의 gold/LLM 혼합 라벨 전체를 사용한다. gold-only는 기존 20개 "
        "human 평가 source subset만 사용하며, 모델 선택의 최종 판단에서는 gold-only 결과를 더 중요하게 본다.",
        "",
        "## 11. train/validation 검증 결과",
        "",
    ])
    for family in ("hybrid_cleaned_fine", "weighted_field_fine", "final_ensemble_fine"):
        tv = best_payload[family]["train_validation_result"]
        lines.append(
            f"- {family}: train best `{weights_markdown(tv['train_best_weights'])}` / "
            f"validation {metrics_markdown(tv['validation_metrics'])}"
        )
    lines.extend(["", "## 12. 5-fold 검증 결과", ""])
    for family in ("hybrid_cleaned_fine", "weighted_field_fine", "final_ensemble_fine"):
        cv = best_payload[family]["cv_average_validation_result"]["average_validation_metrics"]
        lines.append(f"- {family}: {metrics_markdown(cv)}")
    lines.extend([
        "",
        "## 13. 최종 추천 가중치",
        "",
        f"- hybrid_cleaned: {weights_markdown(hc['weights'])}",
        f"- weighted_field: {weights_markdown(wf['weights'])}",
        f"- final_ensemble: {weights_markdown(fe['weights'])}",
        "",
        "full-label 최적 후보를 저장했지만 실제 채택 시에는 gold-only 성능과 CV 안정성을 함께 확인한다. "
        "hybrid_cleaned의 article_similarity가 계속 0 또는 매우 낮게 선택된다면 현재 article_similarity는 "
        "점수 계산보다 추천 설명 근거로 사용하는 편이 적절하다.",
        "",
        "## 14. row-minmax 점수 해석 주의",
        "",
        "row-minmax normalized score는 절대적 동일도를 뜻하지 않고 source별 상대 순위를 위한 점수다. "
        "사용자에게 추천 이유를 보여줄 때 normalized score를 그대로 ‘유사도 1.0’ 또는 ‘완전 일치’로 "
        "표현하지 말고, 원본 점수나 높음/중간/낮음 등급으로 변환해야 한다. top-k JSON에는 두 점수를 "
        "명확히 분리해 저장했다.",
        "",
        "## 15. 향후 개선 방향",
        "",
        "- gold source와 완전 판단 qrels를 확대해 pooled-label 편향을 줄인다.",
        "- article component는 조문 번호 외에 조문 역할·개정 효과를 반영하도록 재설계한다.",
        "- 후보 생성 성능과 최종 ranking 성능을 분리해 온라인 평가 또는 사용자 피드백으로 검증한다.",
        "- final_ensemble의 단일 메소드 대비 개선 폭이 작거나 gold-only에서 하락하면 더 단순한 모델을 우선한다.",
        "",
        "### Normalization 통계 요약",
        "",
        f"총 {len(normalization_stats)}개 component/파생 component의 정규화 전후 통계를 JSON에 저장했다.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(results_df: pd.DataFrame, best_payload: dict[str, Any]) -> None:
    expected_baselines = {
        "raw", "structured", "problem_proposal", "cleaned_problem_proposal",
        "hybrid_cleaned_original", "hybrid_cleaned_best_v1", "weighted_field_original",
        "weighted_field_best_v1", "final_ensemble_best_v1",
    }
    found_baselines = set(
        results_df.loc[results_df["method_family"] == "baseline", "candidate_name"]
    )
    if not expected_baselines.issubset(found_baselines):
        raise RuntimeError(
            f"[baseline 결과 검증] 누락={sorted(expected_baselines - found_baselines)}"
        )
    for family in ("hybrid_cleaned_fine", "weighted_field_fine", "final_ensemble_fine"):
        rows = results_df[results_df["method_family"] == family]
        if not rows["is_v1_best_candidate"].any():
            raise RuntimeError(f"[v1 후보 검증] {family} 결과에 v1 best가 없습니다.")
        if family not in best_payload:
            raise RuntimeError(f"[best JSON 검증] {family} 누락")
    required_outputs = [
        OUTPUT_CSV, OUTPUT_JSON, OUTPUT_BEST_JSON, OUTPUT_TOPK_HYBRID,
        OUTPUT_TOPK_WEIGHTED, OUTPUT_TOPK_ENSEMBLE, OUTPUT_REPORT,
    ]
    missing = [str(path) for path in required_outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"[출력 파일 검증] 생성 실패/빈 파일: {missing}")


def main() -> None:
    started = time.time()
    print("=" * 88)
    print("2차 Fine Grid Search: hybrid_cleaned / weighted_field / final_ensemble")
    print("실행 방법: python 25_fine_grid_search_weighted_methods.py")
    print("=" * 88)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/9] 입력 파일 및 1차 best 검증")
    for path in (DATASET_JSON, LABEL_EXCEL, V1_HYBRID_BEST_JSON, V1_WEIGHTED_ENSEMBLE_BEST_JSON):
        require_file(path, "입력 검증")
    v1_weights = load_v1_weights()

    print("[2/9] 75개 법안 및 평가 라벨 로드")
    dataset = json_load(DATASET_JSON)
    bills = dataset.get("bills", [])
    if len(bills) != 75:
        raise ValueError(f"[데이터셋 검증] 법안 수가 75개가 아닙니다: {len(bills)}")
    bill_ids = [str(bill.get("bill_id", "")).strip() for bill in bills]
    if any(not bill_id for bill_id in bill_ids) or len(set(bill_ids)) != len(bill_ids):
        raise ValueError("[데이터셋 bill_id 검증] 빈 ID 또는 중복 ID가 있습니다.")
    labels = load_labels(bill_ids)
    print(
        f"    full-label: {len(labels.full_source_ids)} sources / {labels.num_label_pairs:,} pairs; "
        f"gold-only: {len(labels.gold_source_ids)} sources / {labels.num_gold_pairs:,} pairs"
    )

    raw_matrices, _ = build_component_matrices(bills)
    matrix_bill_ids = {name: bill_ids.copy() for name in raw_matrices}
    matrix_validation = validate_matrices(raw_matrices, bill_ids, matrix_bill_ids)

    print("[4/9] row-minmax component normalization")
    normalization_stats: dict[str, dict[str, Any]] = {}
    hc_raw = {
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "keyword_tfidf_score": raw_matrices["keyword_tfidf_score"],
        "article_similarity": raw_matrices["article_similarity"],
    }
    hc_norm = {
        name: normalize_component(
            matrix, f"HC {name}", NORMALIZE_COMPONENTS_FOR_HYBRID_CLEANED, normalization_stats
        )
        for name, matrix in hc_raw.items()
    }
    wf_raw = {
        "title_law_name_similarity": raw_matrices["title_law_name_similarity"],
        "full_text_similarity": raw_matrices["full_text_similarity"],
        "current_law_similarity": raw_matrices["current_law_similarity"],
        "problem_similarity": raw_matrices["problem_similarity"],
        "proposal_similarity": raw_matrices["proposal_similarity"],
        "article_similarity": raw_matrices["article_similarity"],
    }
    wf_norm = {
        name: normalize_component(
            matrix, f"WF {name}", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD, normalization_stats
        )
        for name, matrix in wf_raw.items()
    }

    source_splits = build_source_splits(bill_ids, labels.full_source_ids)
    hybrid_candidates = make_candidates(
        [(0.85, 1.00), (0.00, 0.15), (0.00, 0.10)],
        HYBRID_WEIGHT_KEYS,
        HYBRID_FINE_MANUAL_CANDIDATES,
        "HC",
    )
    weighted_candidates = make_candidates(
        [(0.15, 0.30), (0.10, 0.30), (0.05, 0.15), (0.15, 0.30), (0.15, 0.30), (0.05, 0.15)],
        WEIGHTED_WEIGHT_KEYS,
        WEIGHTED_FIELD_FINE_MANUAL_CANDIDATES,
        "WF",
    )
    ensemble_candidates = make_candidates(
        [(0.00, 0.20), (0.05, 0.20), (0.05, 0.20), (0.10, 0.30), (0.00, 0.10), (0.35, 0.60)],
        ENSEMBLE_WEIGHT_KEYS,
        FINAL_ENSEMBLE_FINE_MANUAL_CANDIDATES,
        "FE",
    )
    print(
        f"    후보 수: HC={len(hybrid_candidates):,}, WF={len(weighted_candidates):,}, "
        f"FE={len(ensemble_candidates):,}"
    )

    print("[5/9] hybrid_cleaned fine grid search")
    hc_result = search_family(
        "hybrid_cleaned_fine", hybrid_candidates, HYBRID_WEIGHT_KEYS,
        list(hc_norm.values()), bill_ids, labels, source_splits,
    )
    hc_best_index = hc_result.best_indices["best_by_objective_score"]
    hc_best_weights = {
        key: float(hc_result.candidates[hc_best_index][key]) for key in HYBRID_WEIGHT_KEYS
    }
    hc_best_norm_matrix = weighted_matrix(hc_norm.values(), hc_best_weights, HYBRID_WEIGHT_KEYS)
    hc_best_raw_matrix = weighted_matrix(hc_raw.values(), hc_best_weights, HYBRID_WEIGHT_KEYS)

    print("[6/9] weighted_field fine grid search")
    wf_result = search_family(
        "weighted_field_fine", weighted_candidates, WEIGHTED_WEIGHT_KEYS,
        list(wf_norm.values()), bill_ids, labels, source_splits,
    )
    wf_best_index = wf_result.best_indices["best_by_objective_score"]
    wf_best_weights = {
        key: float(wf_result.candidates[wf_best_index][key]) for key in WEIGHTED_WEIGHT_KEYS
    }
    wf_best_norm_matrix = weighted_matrix(wf_norm.values(), wf_best_weights, WEIGHTED_WEIGHT_KEYS)
    wf_best_raw_matrix = weighted_matrix(wf_raw.values(), wf_best_weights, WEIGHTED_WEIGHT_KEYS)

    print("[7/9] final_ensemble fine grid search (HC/WF v2 best 사용)")
    fe_raw = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_score": wf_best_raw_matrix,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_score": hc_best_raw_matrix,
    }
    # FE가 실제로 받는 파생 component(WF/HC)는 각 family ranking score이며,
    # component_scores_raw에는 별도로 원본 component 가중합을 기록한다.
    fe_pre_normalization = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_score": wf_best_norm_matrix,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_score": hc_best_norm_matrix,
    }
    fe_norm = {
        name: normalize_component(
            matrix, f"FE {name}", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE, normalization_stats
        )
        for name, matrix in fe_pre_normalization.items()
    }
    fe_result = search_family(
        "final_ensemble_fine", ensemble_candidates, ENSEMBLE_WEIGHT_KEYS,
        list(fe_norm.values()), bill_ids, labels, source_splits,
    )
    fe_best_index = fe_result.best_indices["best_by_objective_score"]
    fe_best_weights = {
        key: float(fe_result.candidates[fe_best_index][key]) for key in ENSEMBLE_WEIGHT_KEYS
    }
    fe_best_norm_matrix = weighted_matrix(fe_norm.values(), fe_best_weights, ENSEMBLE_WEIGHT_KEYS)
    fe_best_raw_matrix = weighted_matrix(fe_raw.values(), fe_best_weights, ENSEMBLE_WEIGHT_KEYS)

    print("[8/9] baseline, v1 비교, CSV/JSON/top-k/Markdown 저장")
    hc_v1_norm = weighted_matrix(hc_norm.values(), v1_weights["hybrid"], HYBRID_WEIGHT_KEYS)
    hc_v1_raw = weighted_matrix(hc_raw.values(), v1_weights["hybrid"], HYBRID_WEIGHT_KEYS)
    hc_original_weights = {"w_cleaned": 0.70, "w_tfidf": 0.20, "w_article": 0.10}
    hc_original = weighted_matrix(hc_norm.values(), hc_original_weights, HYBRID_WEIGHT_KEYS)

    wf_v1_norm = weighted_matrix(wf_norm.values(), v1_weights["weighted"], WEIGHTED_WEIGHT_KEYS)
    wf_v1_raw = weighted_matrix(wf_raw.values(), v1_weights["weighted"], WEIGHTED_WEIGHT_KEYS)
    wf_original_weights = {
        "w_title": 0.00, "w_full": 0.20, "w_current": 0.05,
        "w_problem": 0.35, "w_proposal": 0.35, "w_article": 0.05,
    }
    wf_original = weighted_matrix(wf_norm.values(), wf_original_weights, WEIGHTED_WEIGHT_KEYS)

    # 비교용 v1 기반 final ensemble: v1 HC와 v1 WF를 component로 사용한다.
    fe_v1_pre = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_score": wf_v1_norm,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_score": hc_v1_norm,
    }
    fe_v1_norm = {
        name: normalize_component(
            matrix, f"FE-v1 {name}", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE, normalization_stats
        )
        for name, matrix in fe_v1_pre.items()
    }
    final_v1_matrix = weighted_matrix(
        fe_v1_norm.values(), v1_weights["ensemble"], ENSEMBLE_WEIGHT_KEYS
    )

    baseline_matrices = {
        "raw": (raw_matrices["raw_score"], {}, False),
        "structured": (raw_matrices["structured_score"], {}, False),
        "problem_proposal": (raw_matrices["problem_proposal_score"], {}, False),
        "cleaned_problem_proposal": (
            raw_matrices["cleaned_problem_proposal_score"], {}, False
        ),
        "hybrid_cleaned_original": (hc_original, hc_original_weights, True),
        "hybrid_cleaned_best_v1": (hc_v1_norm, v1_weights["hybrid"], True),
        "weighted_field_original": (wf_original, wf_original_weights, True),
        "weighted_field_best_v1": (wf_v1_norm, v1_weights["weighted"], True),
        "final_ensemble_best_v1": (final_v1_matrix, v1_weights["ensemble"], True),
    }
    baseline_rows, baseline_payload = baseline_rows_and_payload(
        baseline_matrices, bill_ids, labels
    )

    all_rows = [
        *family_rows(
            hc_result, v1_weights["hybrid"], NORMALIZE_COMPONENTS_FOR_HYBRID_CLEANED
        ),
        *family_rows(
            wf_result, v1_weights["weighted"], NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD
        ),
        *family_rows(
            fe_result, v1_weights["ensemble"], NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE
        ),
        *baseline_rows,
    ]
    csv_columns = [
        "method_family", "candidate_name", "weights_json", "label_scope",
        "precision_at_5", "precision_at_10", "ndcg_at_10", "mrr",
        "average_relevance", "average_legal_meaning_score", "objective_score",
        "candidate_recall_objective", "ranking_objective", "num_evaluated_pairs",
        "num_sources", "normalization_enabled", "is_manual_candidate",
        "is_v1_best_candidate",
    ]
    results_df = pd.DataFrame(all_rows)[csv_columns]
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

    best_payload = {
        "hybrid_cleaned_fine": family_best_payload(hc_result),
        "weighted_field_fine": family_best_payload(wf_result),
        "final_ensemble_fine": family_best_payload(fe_result),
        "baselines": baseline_payload,
        "metadata": {
            "step": STEP,
            "top_k": TOP_K,
            "random_seed": RANDOM_SEED,
            "selection_scope_for_best": "full_label",
            "gold_detection": labels.gold_detection,
            "normalization_method": NORMALIZATION_METHOD,
            "normalization_enabled": {
                "hybrid_cleaned": NORMALIZE_COMPONENTS_FOR_HYBRID_CLEANED,
                "weighted_field": NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD,
                "final_ensemble": NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE,
            },
            "normalization_stats": normalization_stats,
            "matrix_validation": matrix_validation,
        },
    }
    json_dump(OUTPUT_BEST_JSON, best_payload)

    results_json = {
        "metadata": best_payload["metadata"],
        "label_data": {
            "full_label_source_count": len(labels.full_source_ids),
            "full_label_pair_count": labels.num_label_pairs,
            "gold_only_source_count": len(labels.gold_source_ids),
            "gold_only_pair_count": labels.num_gold_pairs,
            "gold_detection": labels.gold_detection,
            "unjudged_topk_policy": "지표 집계에서 제외; similarity score를 0으로 대체하지 않음",
        },
        "candidate_counts": {
            "hybrid_cleaned_fine": len(hybrid_candidates),
            "weighted_field_fine": len(weighted_candidates),
            "final_ensemble_fine": len(ensemble_candidates),
            "baseline": len(baseline_matrices),
        },
        "best_weights": {
            key: value for key, value in best_payload.items() if key != "metadata"
        },
        "results": all_rows,
    }
    json_dump(OUTPUT_JSON, results_json)

    json_dump(
        OUTPUT_TOPK_HYBRID,
        topk_json_records(
            hc_best_norm_matrix, hc_best_raw_matrix, bills,
            "hybrid_cleaned_fine_best", hc_best_weights, hc_norm, hc_raw,
        ),
    )
    json_dump(
        OUTPUT_TOPK_WEIGHTED,
        topk_json_records(
            wf_best_norm_matrix, wf_best_raw_matrix, bills,
            "weighted_field_fine_best", wf_best_weights, wf_norm, wf_raw,
        ),
    )
    json_dump(
        OUTPUT_TOPK_ENSEMBLE,
        topk_json_records(
            fe_best_norm_matrix, fe_best_raw_matrix, bills,
            "final_ensemble_fine_best", fe_best_weights, fe_norm, fe_raw,
        ),
    )

    candidate_counts = {
        "hybrid_cleaned_fine": len(hybrid_candidates),
        "weighted_field_fine": len(weighted_candidates),
        "final_ensemble_fine": len(ensemble_candidates),
    }
    write_report(best_payload, normalization_stats, labels, candidate_counts)

    print("[9/9] 최종 산출물 검증")
    verify_outputs(results_df, best_payload)
    elapsed = time.time() - started
    print(f"완료: {elapsed:.1f}초")
    print("다음 파일을 확인하세요:")
    for path in (
        OUTPUT_REPORT, OUTPUT_CSV, OUTPUT_BEST_JSON,
        OUTPUT_TOPK_HYBRID, OUTPUT_TOPK_WEIGHTED, OUTPUT_TOPK_ENSEMBLE,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
