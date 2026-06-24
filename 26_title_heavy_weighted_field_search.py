#!/usr/bin/env python3
"""
26_title_heavy_weighted_field_search.py
=======================================
75개 국회 법률발의안 전체 75 x 75 matrix를 재계산하여 다음을 수행한다.

1. title SBERT / exact 두 weighted_field title-heavy 3차 grid search
2. title ablation 및 same-law 편향 진단
3. balanced 기준 weighted_field_title_best 선택
4. 선택된 weighted_field를 사용한 final_ensemble 재탐색
5. full-label / 20-source gold-only / train-validation / 5-fold 평가
6. top-k, 사례 검토 CSV, JSON, Markdown 보고서 저장

실행 방법:
    python 26_title_heavy_weighted_field_search.py

실행 후 확인할 파일:
    Sbert_output/title_heavy_grid_search_report.md
    Sbert_output/title_heavy_grid_search_results.csv
    Sbert_output/title_heavy_grid_search_best_weights.json
    Sbert_output/topk_weighted_field_title_sbert_best.json
    Sbert_output/topk_weighted_field_title_exact_best.json
    Sbert_output/topk_final_ensemble_title_heavy_best.json
    Sbert_output/title_heavy_case_review_sample.csv

categories / manual_categories는 similarity 계산에 사용하지 않는다. 기존 top-k에 없는
pair를 0으로 채우지 않으며, 전체 75 x 75 component matrix를 다시 계산한다.
"""

from __future__ import annotations

import importlib.util
import json
import math
import random
import re
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import torch
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "[의존성 로드 단계] torch와 sentence-transformers가 필요합니다. "
        "Python 3.10+ 및 pandas/numpy/scikit-learn 환경에서 실행하세요."
    ) from exc

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# ── 실행 옵션 ────────────────────────────────────────────────────────────────
STEP = 0.025
TOP_K = 10
RANDOM_SEED = 42
SEARCH_CHUNK_SIZE = 128
MODEL_NAME = "woong0322/ko-legal-sbert-finetuned"

NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD = True
NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE = True
NORMALIZATION_METHOD = "row_minmax"

WEIGHTED_KEYS = [
    "w_title", "w_full", "w_current", "w_problem", "w_proposal", "w_article"
]
ENSEMBLE_KEYS = [
    "w_raw", "w_structured", "w_problem_proposal",
    "w_weighted_field", "w_cleaned", "w_hybrid",
]


# ── 입력/출력 ────────────────────────────────────────────────────────────────
DATASET_JSON = Path("test_dataset/full_dataset.json")
LABEL_EXCEL = Path("Sbert_output/evaluation_full_score_pooled_llm_labeled.xlsx")
V2_BEST_JSON = Path("Sbert_output/fine_grid_search_best_weights.json")
V2_RESULTS_JSON = Path("Sbert_output/fine_grid_search_results.json")
V2_TOPK_HYBRID = Path("Sbert_output/topk_hybrid_cleaned_fine_best.json")
V2_TOPK_WEIGHTED = Path("Sbert_output/topk_weighted_field_fine_best.json")
V2_TOPK_ENSEMBLE = Path("Sbert_output/topk_final_ensemble_fine_best.json")
FINE_SCRIPT = Path("25_fine_grid_search_weighted_methods.py")

OUTPUT_DIR = Path("Sbert_output")
OUTPUT_CSV = OUTPUT_DIR / "title_heavy_grid_search_results.csv"
OUTPUT_JSON = OUTPUT_DIR / "title_heavy_grid_search_results.json"
OUTPUT_BEST_JSON = OUTPUT_DIR / "title_heavy_grid_search_best_weights.json"
OUTPUT_TOPK_SBERT = OUTPUT_DIR / "topk_weighted_field_title_sbert_best.json"
OUTPUT_TOPK_EXACT = OUTPUT_DIR / "topk_weighted_field_title_exact_best.json"
OUTPUT_TOPK_ENSEMBLE = OUTPUT_DIR / "topk_final_ensemble_title_heavy_best.json"
OUTPUT_CASES = OUTPUT_DIR / "title_heavy_case_review_sample.csv"
OUTPUT_REPORT = OUTPUT_DIR / "title_heavy_grid_search_report.md"


WEIGHTED_FIELD_TITLE_HEAVY_MANUAL_CANDIDATES = [
    {
        "name": "WF_v2_best", "w_title": 0.30, "w_full": 0.20,
        "w_current": 0.10, "w_problem": 0.15, "w_proposal": 0.15, "w_article": 0.10,
    },
    {
        "name": "WF_title_40", "w_title": 0.40, "w_full": 0.15,
        "w_current": 0.10, "w_problem": 0.125, "w_proposal": 0.125, "w_article": 0.10,
    },
    {
        "name": "WF_title_50", "w_title": 0.50, "w_full": 0.10,
        "w_current": 0.10, "w_problem": 0.10, "w_proposal": 0.10, "w_article": 0.10,
    },
    {
        "name": "WF_title_60", "w_title": 0.60, "w_full": 0.10,
        "w_current": 0.05, "w_problem": 0.10, "w_proposal": 0.10, "w_article": 0.05,
    },
    {
        "name": "WF_title_70", "w_title": 0.70, "w_full": 0.05,
        "w_current": 0.05, "w_problem": 0.075, "w_proposal": 0.075, "w_article": 0.05,
    },
    {
        "name": "WF_title_ablation_zero", "w_title": 0.00, "w_full": 0.25,
        "w_current": 0.10, "w_problem": 0.30, "w_proposal": 0.25, "w_article": 0.10,
    },
    {
        "name": "WF_title_only_like", "w_title": 0.85, "w_full": 0.05,
        "w_current": 0.00, "w_problem": 0.025, "w_proposal": 0.025, "w_article": 0.05,
    },
]

FINAL_ENSEMBLE_TITLE_HEAVY_MANUAL_CANDIDATES = [
    {
        "name": "FE_v2_best", "w_raw": 0.075, "w_structured": 0.100,
        "w_problem_proposal": 0.075, "w_weighted_field": 0.300,
        "w_cleaned": 0.000, "w_hybrid": 0.450,
    },
    {
        "name": "FE_more_weighted_35", "w_raw": 0.050, "w_structured": 0.100,
        "w_problem_proposal": 0.050, "w_weighted_field": 0.350,
        "w_cleaned": 0.000, "w_hybrid": 0.450,
    },
    {
        "name": "FE_more_weighted_40", "w_raw": 0.025, "w_structured": 0.100,
        "w_problem_proposal": 0.050, "w_weighted_field": 0.400,
        "w_cleaned": 0.000, "w_hybrid": 0.425,
    },
    {
        "name": "FE_hybrid_60_gold_candidate", "w_raw": 0.000, "w_structured": 0.100,
        "w_problem_proposal": 0.150, "w_weighted_field": 0.125,
        "w_cleaned": 0.025, "w_hybrid": 0.600,
    },
    {
        "name": "FE_weighted_hybrid_balanced", "w_raw": 0.025, "w_structured": 0.100,
        "w_problem_proposal": 0.050, "w_weighted_field": 0.350,
        "w_cleaned": 0.000, "w_hybrid": 0.475,
    },
]


def load_fine_module() -> Any:
    if not FINE_SCRIPT.exists():
        raise FileNotFoundError(f"[공통 로직 로드] 파일이 없습니다: {FINE_SCRIPT}")
    spec = importlib.util.spec_from_file_location("fine_grid_v2_support", FINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"[공통 로직 로드] import spec 생성 실패: {FINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


FINE = load_fine_module()
METRIC_NAMES = FINE.METRIC_NAMES


@dataclass
class SearchResult:
    method_family: str
    weight_keys: list[str]
    candidates: list[dict[str, Any]]
    full_metrics: dict[str, np.ndarray]
    gold_metrics: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    balanced_scores: np.ndarray
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


def row_minmax_normalize(
    matrix: np.ndarray, ignore_diagonal: bool = True
) -> tuple[np.ndarray, dict[str, float]]:
    return FINE.row_minmax_normalize(matrix, ignore_diagonal=ignore_diagonal)


def normalize_law_name(bill_name: str) -> str:
    """개정 형식 표현을 제거하고 실제 법률명 중심 문자열을 반환한다."""
    value = re.sub(r"\s+", " ", str(bill_name)).strip()
    suffix = (
        r"(?:일부개정법률안|전부개정법률안|제정법률안|개정법률안|"
        r"일부개정안|전부개정안|개정안|법률안)$"
    )
    previous = None
    while previous != value:
        previous = value
        value = re.sub(suffix, "", value).strip()
    return value or str(bill_name).strip()


def validate_v2_inputs() -> dict[str, Any]:
    for path in (
        DATASET_JSON, LABEL_EXCEL, V2_BEST_JSON, V2_RESULTS_JSON,
        V2_TOPK_HYBRID, V2_TOPK_WEIGHTED, V2_TOPK_ENSEMBLE,
    ):
        require_file(path, "2차 입력 검증")
    best = json_load(V2_BEST_JSON)
    results = json_load(V2_RESULTS_JSON)
    topk_validation: dict[str, Any] = {}
    for path in (V2_TOPK_HYBRID, V2_TOPK_WEIGHTED, V2_TOPK_ENSEMBLE):
        records = json_load(path)
        self_pairs = sum(
            item["source_bill_id"] == item["target_bill_id"] for item in records
        )
        if len(records) != 750 or self_pairs:
            raise ValueError(
                f"[2차 top-k 검증] {path}: rows={len(records)}, self_pairs={self_pairs}"
            )
        topk_validation[path.name] = {"rows": len(records), "self_pairs": self_pairs}
    return {
        "best": best,
        "v2_results_candidate_counts": results.get("candidate_counts", {}),
        "topk_validation": topk_validation,
    }


def validate_v2_weights(best: dict[str, Any]) -> dict[str, dict[str, float]]:
    weights = {
        "hybrid": best["hybrid_cleaned_fine"]["best_by_objective_score"]["weights"],
        "weighted": best["weighted_field_fine"]["best_by_objective_score"]["weights"],
        "ensemble": best["final_ensemble_fine"]["best_by_objective_score"]["weights"],
    }
    expected = {
        "hybrid": {"w_cleaned": 0.90, "w_tfidf": 0.10, "w_article": 0.00},
        "weighted": {
            "w_title": 0.30, "w_full": 0.20, "w_current": 0.10,
            "w_problem": 0.15, "w_proposal": 0.15, "w_article": 0.10,
        },
        "ensemble": {
            "w_raw": 0.075, "w_structured": 0.100, "w_problem_proposal": 0.075,
            "w_weighted_field": 0.300, "w_cleaned": 0.000, "w_hybrid": 0.450,
        },
    }
    for family, expected_values in expected.items():
        for key, expected_value in expected_values.items():
            actual = float(weights[family][key])
            if not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-9):
                raise ValueError(
                    f"[2차 best 검증] {family}.{key}: actual={actual}, expected={expected_value}"
                )
    return {
        family: {key: float(value) for key, value in values.items()}
        for family, values in weights.items()
    }


def normalize_component(
    matrix: np.ndarray,
    name: str,
    enabled: bool,
    stats: dict[str, Any],
) -> np.ndarray:
    return FINE.normalize_component(matrix, name, enabled, stats)


def weighted_matrix(
    matrices: Iterable[np.ndarray], weights: dict[str, float], keys: list[str]
) -> np.ndarray:
    return FINE.weighted_matrix(matrices, weights, keys)


def build_title_matrices(
    law_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    title 법률명 SBERT 재계산: {MODEL_NAME} (device={device})")
    model = SentenceTransformer(MODEL_NAME, device=device)
    embeddings = model.encode(
        law_names,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    title_sbert = (embeddings @ embeddings.T).detach().cpu().numpy().astype(np.float64)
    names = np.asarray(law_names, dtype=object)
    title_exact = (names[:, None] == names[None, :]).astype(np.float64)
    return title_sbert, title_exact


def make_candidates(
    ranges: list[tuple[float, float]],
    keys: list[str],
    manuals: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    """마지막 가중치를 1-sum(first weights)로 역산해 6차원 탐색을 빠르게 생성한다."""
    unit = int(round(STEP * 1000))

    def values(bounds: tuple[float, float]) -> list[int]:
        start, end = (int(round(value * 1000)) for value in bounds)
        return list(range(start, end + 1, unit))

    value_lists = [values(bounds) for bounds in ranges]
    allowed_last = set(value_lists[-1])
    by_weights: dict[tuple[float, ...], dict[str, Any]] = {}
    counter = 0
    for leading in product(*value_lists[:-1]):
        last = 1000 - sum(leading)
        if last not in allowed_last:
            continue
        counter += 1
        weight_tuple = tuple(round(value / 1000.0, 3) for value in (*leading, last))
        by_weights[weight_tuple] = {
            "name": f"{prefix}_grid_{counter:05d}",
            **dict(zip(keys, weight_tuple)),
            "is_manual_candidate": False,
        }
    for manual in manuals:
        FINE.validate_weights(manual, keys, f"{prefix} 수동 후보")
        weight_tuple = tuple(round(float(manual[key]), 3) for key in keys)
        record = by_weights.get(weight_tuple, {**dict(zip(keys, weight_tuple))})
        record.update(name=manual["name"], is_manual_candidate=True)
        by_weights[weight_tuple] = record
    candidates = sorted(
        by_weights.values(), key=lambda item: tuple(float(item[key]) for key in keys)
    )
    for candidate in candidates:
        FINE.validate_weights(candidate, keys, f"{prefix} 후보 생성")
    return candidates


def diagnostic_arrays(
    top_indices: np.ndarray,
    same_law_matrix: np.ndarray,
    law_group_ids: np.ndarray,
    title_similarity_matrix: np.ndarray,
    non_title_average_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate_count, source_count, _ = top_indices.shape
    source_grid = np.arange(source_count)[None, :, None]
    same = same_law_matrix[source_grid, top_indices]
    title_values = title_similarity_matrix[source_grid, top_indices]
    non_title_values = non_title_average_matrix[source_grid, top_indices]
    target_groups = law_group_ids[top_indices]
    sorted_groups = np.sort(target_groups, axis=2)
    unique_counts = 1 + np.sum(sorted_groups[:, :, 1:] != sorted_groups[:, :, :-1], axis=2)
    diagnostics = {
        "same_law_ratio_at_5": same[:, :, :5].mean(axis=(1, 2)),
        "same_law_ratio_at_10": same.mean(axis=(1, 2)),
        "unique_target_law_count_avg_at_10": unique_counts.mean(axis=1),
        "avg_title_similarity_at_10": title_values.mean(axis=(1, 2)),
        "avg_non_title_component_score_at_10": non_title_values.mean(axis=(1, 2)),
    }
    for name, values in diagnostics.items():
        if values.shape != (candidate_count,) or not np.all(np.isfinite(values)):
            raise ValueError(f"[same-law 진단] {name} 계산 이상: shape={values.shape}")
    if np.any((diagnostics["same_law_ratio_at_10"] < 0) | (diagnostics["same_law_ratio_at_10"] > 1)):
        raise ValueError("[same-law 진단] same_law_ratio_at_10이 0~1 범위를 벗어났습니다.")
    return diagnostics


def balance_scores(
    full: dict[str, np.ndarray],
    gold: dict[str, np.ndarray],
    diagnostics: dict[str, np.ndarray],
    reference_full: dict[str, Any],
    reference_gold: dict[str, Any],
    balance_kind: str,
) -> np.ndarray:
    full_legal = np.nan_to_num(full["average_legal_meaning_score"], nan=0.0)
    gold_legal = np.nan_to_num(gold["average_legal_meaning_score"], nan=0.0)
    scores = (
        0.35 * full["objective_score"]
        + 0.45 * gold["objective_score"]
        + 0.10 * full_legal / 100.0
        + 0.10 * gold_legal / 100.0
    )
    if balance_kind == "weighted":
        excessive_same_law = diagnostics["same_law_ratio_at_10"] > 0.70
        gold_not_improved = gold["objective_score"] <= float(reference_gold["objective_score"])
        scores -= np.where(excessive_same_law & gold_not_improved, 0.03, 0.0)
        # 범위가 명시되지 않은 Legal Meaning 페널티는 최종 판단 우선순위에 따라 gold-only를 사용한다.
        legal_drop = gold_legal < float(reference_gold["average_legal_meaning_score"]) - 2.0
        scores -= np.where(legal_drop, 0.02, 0.0)
    elif balance_kind == "ensemble":
        scores -= np.where(
            full["objective_score"] < float(reference_full["objective_score"]) - 0.01,
            0.02,
            0.0,
        )
        scores -= np.where(
            gold["objective_score"] < float(reference_gold["objective_score"]) - 0.01,
            0.02,
            0.0,
        )
    else:
        raise ValueError(f"[balanced score] 알 수 없는 family 종류: {balance_kind}")
    return scores


def search_family(
    method_family: str,
    candidates: list[dict[str, Any]],
    weight_keys: list[str],
    component_matrices: list[np.ndarray],
    bill_ids: list[str],
    labels: Any,
    source_splits: dict[str, np.ndarray],
    same_law_matrix: np.ndarray,
    law_group_ids: np.ndarray,
    title_similarity_matrix: np.ndarray,
    non_title_average_matrix: np.ndarray,
    reference_full: dict[str, Any],
    reference_gold: dict[str, Any],
    balance_kind: str,
) -> SearchResult:
    print(f"[탐색] {method_family}: {len(candidates):,}개 후보")
    component_stack = np.stack(component_matrices, axis=0).astype(np.float64)
    weights = np.asarray(
        [[float(candidate[key]) for key in weight_keys] for candidate in candidates],
        dtype=np.float64,
    )
    count = len(candidates)

    def allocate_metrics() -> dict[str, np.ndarray]:
        return {name: np.empty(count, dtype=np.float64) for name in METRIC_NAMES}

    full = allocate_metrics()
    gold = allocate_metrics()
    split_metrics = {name: allocate_metrics() for name in source_splits}
    diagnostic_names = [
        "same_law_ratio_at_5", "same_law_ratio_at_10",
        "unique_target_law_count_avg_at_10", "avg_title_similarity_at_10",
        "avg_non_title_component_score_at_10",
    ]
    diagnostics = {name: np.empty(count, dtype=np.float64) for name in diagnostic_names}

    for start in range(0, count, SEARCH_CHUNK_SIZE):
        end = min(start + SEARCH_CHUNK_SIZE, count)
        combined = np.einsum(
            "ck,kij->cij", weights[start:end], component_stack, optimize=True
        )
        top_indices = FINE.deterministic_topk_indices(combined, bill_ids, TOP_K)
        evaluations = {
            "full": FINE.metrics_for_topk(
                top_indices, labels.relevance_full, labels.legal_full, labels.full_source_mask
            ),
            "gold": FINE.metrics_for_topk(
                top_indices, labels.relevance_gold, labels.legal_gold, labels.gold_source_mask
            ),
        }
        for split_name, split_mask in source_splits.items():
            evaluations[split_name] = FINE.metrics_for_topk(
                top_indices, labels.relevance_full, labels.legal_full, split_mask
            )
        chunk_diagnostics = diagnostic_arrays(
            top_indices, same_law_matrix, law_group_ids,
            title_similarity_matrix, non_title_average_matrix,
        )
        for metric in METRIC_NAMES:
            full[metric][start:end] = evaluations["full"][metric]
            gold[metric][start:end] = evaluations["gold"][metric]
            for split_name in source_splits:
                split_metrics[split_name][metric][start:end] = evaluations[split_name][metric]
        for name in diagnostic_names:
            diagnostics[name][start:end] = chunk_diagnostics[name]
        if end == count or end % 2048 == 0:
            print(f"    진행: {end:,}/{count:,}")

    balanced = balance_scores(
        full, gold, diagnostics, reference_full, reference_gold, balance_kind
    )
    best_indices = {
        "best_by_full_label_objective": int(np.argmax(full["objective_score"])),
        "best_by_gold_only_objective": int(np.argmax(gold["objective_score"])),
        "best_by_full_label_candidate_recall": int(
            np.argmax(full["candidate_recall_objective"])
        ),
        "best_by_gold_only_candidate_recall": int(
            np.argmax(gold["candidate_recall_objective"])
        ),
        "best_by_balanced_score": int(np.argmax(balanced)),
        "best_by_ranking_objective": int(np.argmax(full["ranking_objective"])),
    }

    train_best = int(np.argmax(split_metrics["train"]["objective_score"]))
    train_validation = {
        "selection_scope": "full_label_train_sources",
        "random_seed": RANDOM_SEED,
        "train_source_count": int(source_splits["train"].sum()),
        "validation_source_count": int(source_splits["validation"].sum()),
        "train_best_weights": {
            key: float(candidates[train_best][key]) for key in weight_keys
        },
        "train_metrics": FINE.metric_record(split_metrics["train"], train_best),
        "validation_metrics": FINE.metric_record(split_metrics["validation"], train_best),
    }
    folds: list[dict[str, Any]] = []
    fold_validation_metrics: list[dict[str, Any]] = []
    for fold in range(5):
        train_name = f"fold_{fold}_train"
        validation_name = f"fold_{fold}_validation"
        fold_best = int(np.argmax(split_metrics[train_name]["objective_score"]))
        validation_metrics = FINE.metric_record(split_metrics[validation_name], fold_best)
        fold_validation_metrics.append(validation_metrics)
        folds.append({
            "fold": fold + 1,
            "train_source_count": int(source_splits[train_name].sum()),
            "validation_source_count": int(source_splits[validation_name].sum()),
            "train_best_weights": {
                key: float(candidates[fold_best][key]) for key in weight_keys
            },
            "train_metrics": FINE.metric_record(split_metrics[train_name], fold_best),
            "validation_metrics": validation_metrics,
        })
    cv = {
        "selection_scope": "full_label_train_sources",
        "random_seed": RANDOM_SEED,
        "num_folds": 5,
        "folds": folds,
        "average_validation_metrics": FINE.average_metric_records(fold_validation_metrics),
    }
    return SearchResult(
        method_family=method_family,
        weight_keys=weight_keys,
        candidates=candidates,
        full_metrics=full,
        gold_metrics=gold,
        diagnostics=diagnostics,
        balanced_scores=balanced,
        split_metrics=split_metrics,
        best_indices=best_indices,
        train_validation_result=train_validation,
        cv_average_validation_result=cv,
    )


def diagnostic_record(result: SearchResult, index: int) -> dict[str, float]:
    return {
        name: round(float(values[index]), 4)
        for name, values in result.diagnostics.items()
    }


def selection_record(result: SearchResult, index: int) -> dict[str, Any]:
    candidate = result.candidates[index]
    return {
        "candidate_name": candidate["name"],
        "weights": {key: float(candidate[key]) for key in result.weight_keys},
        "balanced_score": round(float(result.balanced_scores[index]), 4),
        "metrics_full_label": FINE.metric_record(result.full_metrics, index),
        "metrics_gold_only": FINE.metric_record(result.gold_metrics, index),
        "same_law_bias_diagnostics": diagnostic_record(result, index),
    }


def ablation_group(weight: float) -> str | None:
    if math.isclose(weight, 0.0, abs_tol=1e-9):
        return "no_title"
    if 0.30 <= weight <= 0.45:
        return "moderate_title"
    if 0.45 < weight <= 0.70:
        return "heavy_title"
    if weight > 0.70:
        return "title_dominant_manual"
    return None


def ablation_summary(result: SearchResult) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in ("no_title", "moderate_title", "heavy_title", "title_dominant_manual"):
        indices = np.asarray([
            idx for idx, candidate in enumerate(result.candidates)
            if ablation_group(float(candidate["w_title"])) == group
        ], dtype=int)
        if indices.size == 0:
            output[group] = {"candidate_count": 0}
            continue
        best_full = int(indices[np.argmax(result.full_metrics["objective_score"][indices])])
        best_gold = int(indices[np.argmax(result.gold_metrics["objective_score"][indices])])
        best_balanced = int(indices[np.argmax(result.balanced_scores[indices])])
        output[group] = {
            "candidate_count": int(indices.size),
            "best_by_full_label_objective": selection_record(result, best_full),
            "best_by_gold_only_objective": selection_record(result, best_gold),
            "best_by_balanced_score": selection_record(result, best_balanced),
        }
    return output


def bias_summary(result: SearchResult) -> dict[str, Any]:
    weights = np.asarray([float(candidate["w_title"]) for candidate in result.candidates])

    def correlation(values: np.ndarray) -> float | None:
        if np.std(weights) < 1e-12 or np.std(values) < 1e-12:
            return None
        return round(float(np.corrcoef(weights, values)[0, 1]), 4)

    representative_names = {
        "WF_v2_best", "WF_title_40", "WF_title_50", "WF_title_60",
        "WF_title_70", "WF_title_ablation_zero", "WF_title_only_like",
    }
    representatives: dict[str, Any] = {}
    for index, candidate in enumerate(result.candidates):
        if candidate["name"] in representative_names:
            representatives[candidate["name"]] = selection_record(result, index)
    for selection_name, index in result.best_indices.items():
        representatives[selection_name] = selection_record(result, index)
    return {
        "correlation_w_title_same_law_ratio_at_10": correlation(
            result.diagnostics["same_law_ratio_at_10"]
        ),
        "correlation_w_title_full_label_objective": correlation(
            result.full_metrics["objective_score"]
        ),
        "correlation_w_title_gold_only_objective": correlation(
            result.gold_metrics["objective_score"]
        ),
        "correlation_w_title_full_label_legal_meaning": correlation(
            result.full_metrics["average_legal_meaning_score"]
        ),
        "representative_candidates": representatives,
    }


def family_payload(result: SearchResult, include_ablation: bool) -> dict[str, Any]:
    payload = {
        name: selection_record(result, index)
        for name, index in result.best_indices.items()
    }
    if include_ablation:
        payload["title_ablation_summary"] = ablation_summary(result)
        payload["same_law_bias_diagnostics"] = bias_summary(result)
    payload["train_validation_result"] = result.train_validation_result
    payload["cv_average_validation_result"] = result.cv_average_validation_result
    return payload


def family_rows(result: SearchResult, v2_weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(result.candidates):
        weights = {key: float(candidate[key]) for key in result.weight_keys}
        is_v2 = all(
            key in v2_weights and math.isclose(weights[key], float(v2_weights[key]), abs_tol=1e-9)
            for key in result.weight_keys
        )
        diagnostic = diagnostic_record(result, index)
        for scope, metrics in (
            ("full_label", result.full_metrics),
            ("gold_only", result.gold_metrics),
        ):
            metric = FINE.metric_record(metrics, index)
            rows.append({
                "method_family": result.method_family,
                "candidate_name": candidate["name"],
                "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "label_scope": scope,
                **metric,
                "balanced_score": round(float(result.balanced_scores[index]), 4),
                **diagnostic,
                "normalization_enabled": True,
                "is_manual_candidate": bool(candidate["is_manual_candidate"]),
                "is_v2_best_candidate": is_v2,
            })
    return rows


def evaluate_matrix(
    matrix: np.ndarray,
    bill_ids: list[str],
    labels: Any,
    same_law_matrix: np.ndarray,
    law_group_ids: np.ndarray,
    title_matrix: np.ndarray,
    non_title_matrix: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, float], np.ndarray]:
    top_indices = FINE.deterministic_topk_indices(matrix, bill_ids, TOP_K)
    full_arrays = FINE.metrics_for_topk(
        top_indices, labels.relevance_full, labels.legal_full, labels.full_source_mask
    )
    gold_arrays = FINE.metrics_for_topk(
        top_indices, labels.relevance_gold, labels.legal_gold, labels.gold_source_mask
    )
    diag_arrays = diagnostic_arrays(
        top_indices, same_law_matrix, law_group_ids, title_matrix, non_title_matrix
    )
    full = FINE.metric_record(full_arrays, 0)
    gold = FINE.metric_record(gold_arrays, 0)
    diagnostics = {name: round(float(values[0]), 4) for name, values in diag_arrays.items()}
    return full, gold, diagnostics, top_indices[0]


def baseline_payload_and_rows(
    baselines: dict[str, tuple[np.ndarray, dict[str, float], bool]],
    bill_ids: list[str],
    labels: Any,
    same_law_matrix: np.ndarray,
    law_group_ids: np.ndarray,
    title_matrix: np.ndarray,
    non_title_matrix: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name, (matrix, weights, normalized) in baselines.items():
        full, gold, diagnostics, _ = evaluate_matrix(
            matrix, bill_ids, labels, same_law_matrix, law_group_ids,
            title_matrix, non_title_matrix,
        )
        balanced = (
            0.35 * full["objective_score"] + 0.45 * gold["objective_score"]
            + 0.10 * (full["average_legal_meaning_score"] or 0.0) / 100.0
            + 0.10 * (gold["average_legal_meaning_score"] or 0.0) / 100.0
        )
        payload[name] = {
            "weights": weights,
            "metrics_full_label": full,
            "metrics_gold_only": gold,
            "balanced_score": round(balanced, 4),
            "same_law_bias_diagnostics": diagnostics,
            "normalization_enabled": normalized,
        }
        for scope, metrics in (("full_label", full), ("gold_only", gold)):
            rows.append({
                "method_family": "baseline",
                "candidate_name": name,
                "weights_json": json.dumps(weights, ensure_ascii=False, sort_keys=True),
                "label_scope": scope,
                **metrics,
                "balanced_score": round(balanced, 4),
                **diagnostics,
                "normalization_enabled": normalized,
                "is_manual_candidate": False,
                "is_v2_best_candidate": name.endswith("_v2"),
            })
    return payload, rows


def topk_records(
    ranking_matrix: np.ndarray,
    raw_weighted_matrix: np.ndarray,
    bills: list[dict[str, Any]],
    law_names: list[str],
    method: str,
    weights: dict[str, float],
    normalized_components: dict[str, np.ndarray],
    raw_components: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    bill_ids = [str(bill["bill_id"]) for bill in bills]
    indices = FINE.deterministic_topk_indices(ranking_matrix, bill_ids, TOP_K)[0]
    records: list[dict[str, Any]] = []
    for source, targets in enumerate(indices):
        if len(targets) != TOP_K:
            raise RuntimeError(f"[top-k 저장] {bill_ids[source]}에 top-10이 없습니다.")
        for rank, target in enumerate(targets, start=1):
            target = int(target)
            if source == target:
                raise RuntimeError(f"[top-k 저장] diagonal 포함: {bill_ids[source]}")
            records.append({
                "source_bill_id": bill_ids[source],
                "source_bill_name": str(bills[source]["bill_name"]),
                "source_law_name_normalized": law_names[source],
                "target_bill_id": bill_ids[target],
                "target_bill_name": str(bills[target]["bill_name"]),
                "target_law_name_normalized": law_names[target],
                "same_law": law_names[source] == law_names[target],
                "rank": rank,
                "similarity": round(float(ranking_matrix[source, target]), 6),
                "similarity_normalized_weighted": round(
                    float(ranking_matrix[source, target]), 6
                ),
                "similarity_raw_weighted": round(float(raw_weighted_matrix[source, target]), 6),
                "method": method,
                "component_scores_normalized": {
                    name: round(float(matrix[source, target]), 6)
                    for name, matrix in normalized_components.items()
                },
                "component_scores_raw": {
                    name: round(float(matrix[source, target]), 6)
                    for name, matrix in raw_components.items()
                },
                "weights": weights,
            })
    if len(records) != 750:
        raise RuntimeError(f"[top-k 저장] rows={len(records)}, expected=750")
    return records


def full_rank_matrix(matrix: np.ndarray, bill_ids: list[str]) -> np.ndarray:
    indices = FINE.deterministic_topk_indices(matrix, bill_ids, len(bill_ids) - 1)[0]
    ranks = np.zeros_like(matrix, dtype=int)
    for source in range(len(bill_ids)):
        for rank, target in enumerate(indices[source], start=1):
            ranks[source, int(target)] = rank
    return ranks


def write_case_review_sample(
    v2_matrix: np.ndarray,
    title_matrix: np.ndarray,
    bill_ids: list[str],
    bills: list[dict[str, Any]],
    law_names: list[str],
    labels: Any,
    title_component_raw: np.ndarray,
    non_title_average_norm: np.ndarray,
) -> dict[str, int]:
    rank_v2 = full_rank_matrix(v2_matrix, bill_ids)
    rank_title = full_rank_matrix(title_matrix, bill_ids)
    id_to_index = {bill_id: idx for idx, bill_id in enumerate(bill_ids)}
    label_df = pd.read_excel(LABEL_EXCEL)
    required = [
        "source_bill_id", "target_bill_id", "human_relevance_0_to_4",
        "human_issue_match_0_to_2", "human_target_match_0_to_2",
        "human_effect_match_0_to_2", "human_scope_match_0_to_2",
        "human_article_match_0_to_2", "notes",
    ]
    missing = [column for column in required if column not in label_df.columns]
    if missing:
        raise KeyError(f"[사례 샘플] 라벨 파일 컬럼 누락: {missing}")

    rows: list[dict[str, Any]] = []
    gold_sources = set(labels.gold_source_ids)
    for _, row in label_df.iterrows():
        source_id = str(row["source_bill_id"]).strip()
        target_id = str(row["target_bill_id"]).strip()
        relevance = FINE.safe_float(row["human_relevance_0_to_4"])
        if source_id not in id_to_index or target_id not in id_to_index or relevance is None:
            continue
        source, target = id_to_index[source_id], id_to_index[target_id]
        rv2, rtitle = int(rank_v2[source, target]), int(rank_title[source, target])
        rows.append({
            "source_bill_id": source_id,
            "source_bill_name": str(bills[source]["bill_name"]),
            "source_law_name_normalized": law_names[source],
            "target_bill_id": target_id,
            "target_bill_name": str(bills[target]["bill_name"]),
            "target_law_name_normalized": law_names[target],
            "same_law": law_names[source] == law_names[target],
            "rank_v2": rv2,
            "rank_title_heavy": rtitle,
            "rank_change": rv2 - rtitle,
            "relevance": relevance,
            "issue_match": FINE.safe_float(row["human_issue_match_0_to_2"]),
            "target_match": FINE.safe_float(row["human_target_match_0_to_2"]),
            "effect_match": FINE.safe_float(row["human_effect_match_0_to_2"]),
            "scope_match": FINE.safe_float(row["human_scope_match_0_to_2"]),
            "article_match": FINE.safe_float(row["human_article_match_0_to_2"]),
            "title_similarity": round(float(title_component_raw[source, target]), 6),
            "non_title_average_score": round(
                float(non_title_average_norm[source, target]), 6
            ),
            "notes": "" if pd.isna(row["notes"]) else str(row["notes"]),
            "_gold_priority": source_id in gold_sources,
        })
    data = pd.DataFrame(rows)
    if data.empty:
        raise ValueError("[사례 샘플] 생성 가능한 라벨 쌍이 없습니다.")

    rising = data.sort_values(
        ["_gold_priority", "rank_change", "relevance"], ascending=[False, False, False]
    ).head(30)
    falling = data.sort_values(
        ["_gold_priority", "rank_change", "relevance"], ascending=[False, True, False]
    ).head(30)
    same_low = data[(data["same_law"]) & (data["relevance"] < 3)].sort_values(
        ["_gold_priority", "relevance", "rank_title_heavy"],
        ascending=[False, True, True],
    ).head(30)
    different_high = data[(~data["same_law"]) & (data["relevance"] >= 3)].sort_values(
        ["_gold_priority", "relevance", "rank_title_heavy"],
        ascending=[False, False, True],
    ).head(30)
    sample = pd.concat([rising, falling, same_low, different_high], ignore_index=True)
    sample = sample.drop_duplicates(["source_bill_id", "target_bill_id"], keep="first")
    columns = [
        "source_bill_id", "source_bill_name", "source_law_name_normalized",
        "target_bill_id", "target_bill_name", "target_law_name_normalized",
        "same_law", "rank_v2", "rank_title_heavy", "rank_change", "relevance",
        "issue_match", "target_match", "effect_match", "scope_match", "article_match",
        "title_similarity", "non_title_average_score", "notes",
    ]
    sample[columns].to_csv(OUTPUT_CASES, index=False, encoding="utf-8-sig")
    return {
        "title_heavy_rank_rise_selected": len(rising),
        "title_heavy_rank_fall_selected": len(falling),
        "same_law_low_relevance_selected": len(same_low),
        "different_law_high_relevance_selected": len(different_high),
        "unique_output_rows_after_deduplication": len(sample),
        "gold_source_rows": int(sample["_gold_priority"].sum()),
    }


def metric_text(metrics: dict[str, Any]) -> str:
    return (
        f"P@5={metrics['precision_at_5']:.4f}, P@10={metrics['precision_at_10']:.4f}, "
        f"nDCG@10={metrics['ndcg_at_10']:.4f}, MRR={metrics['mrr']:.4f}, "
        f"AvgRel={metrics['average_relevance']:.4f}, "
        f"Legal={metrics['average_legal_meaning_score']}"
    )


def weight_text(weights: dict[str, Any]) -> str:
    return ", ".join(f"{key}={float(value):.3f}" for key, value in weights.items())


def report_family_selection(lines: list[str], title: str, payload: dict[str, Any]) -> None:
    lines.extend([f"### {title}", ""])
    for key in (
        "best_by_full_label_objective", "best_by_gold_only_objective",
        "best_by_full_label_candidate_recall", "best_by_gold_only_candidate_recall",
        "best_by_ranking_objective", "best_by_balanced_score",
    ):
        if key not in payload:
            continue
        value = payload[key]
        lines.append(
            f"- {key}: `{weight_text(value['weights'])}` / balanced={value['balanced_score']:.4f} / "
            f"full objective={value['metrics_full_label']['objective_score']:.4f} / "
            f"gold objective={value['metrics_gold_only']['objective_score']:.4f}"
        )
    lines.append("")


def write_report(
    best_payload: dict[str, Any],
    candidate_counts: dict[str, int],
    label_data: Any,
    selected_wf_family: str,
    case_summary: dict[str, int],
) -> None:
    sbert = best_payload["weighted_field_title_sbert"]
    exact = best_payload["weighted_field_title_exact"]
    ensemble = best_payload["final_ensemble_title_heavy"]
    sbert_bal = sbert["best_by_balanced_score"]
    exact_bal = exact["best_by_balanced_score"]
    final_bal = ensemble["best_by_balanced_score"]
    selected_payload = best_payload[selected_wf_family]["best_by_balanced_score"]
    baselines = best_payload["baselines"]
    v2_wf = baselines["weighted_field_fine_best_v2"]
    v2_fe = baselines["final_ensemble_fine_best_v2"]

    sb_bias = sbert["same_law_bias_diagnostics"]
    ex_bias = exact["same_law_bias_diagnostics"]
    selected_bias = selected_payload["same_law_bias_diagnostics"]
    v2_bias = v2_wf["same_law_bias_diagnostics"]
    same_delta = selected_bias["same_law_ratio_at_10"] - v2_bias["same_law_ratio_at_10"]
    p5_delta = (
        selected_payload["metrics_gold_only"]["precision_at_5"]
        - v2_wf["metrics_gold_only"]["precision_at_5"]
    )
    ndcg_delta = (
        selected_payload["metrics_gold_only"]["ndcg_at_10"]
        - v2_wf["metrics_gold_only"]["ndcg_at_10"]
    )
    legal_delta = (
        selected_payload["metrics_gold_only"]["average_legal_meaning_score"]
        - v2_wf["metrics_gold_only"]["average_legal_meaning_score"]
    )

    lines = [
        "# 3차 Title-Heavy Weighted Field 탐색 보고서",
        "",
        "## 1. 실험 목적",
        "",
        "법률발의안 유사도에서 ‘어떤 법률을 개정하는가’가 어느 정도까지 중요한지 검증했다. "
        "전체 75×75 행렬을 재계산하고 title SBERT와 exact same-law 두 방식을 비교했다.",
        "",
        "## 2. 2차 Fine Search 결과 요약",
        "",
        "- hybrid_cleaned: cleaned=0.90, TF-IDF=0.10, article=0.00",
        "- weighted_field: title=0.30, full=0.20, current=0.10, problem=0.15, proposal=0.15, article=0.10",
        "- final_ensemble: raw=0.075, structured=0.10, problem_proposal=0.075, weighted_field=0.30, cleaned=0.00, hybrid=0.45",
        "",
        "## 3. 왜 title 비중을 확장했는가",
        "",
        "2차 최적 w_title=0.30이 탐색 상한에 걸렸기 때문에 0.70까지 확장했다. title 비중을 "
        "높이면 같은 법률명 또는 같은 법체계의 법안이 상위에 더 많이 노출될 수 있으므로 성능과 "
        "편향을 함께 측정했다.",
        "",
        "## 4. 평가 데이터 설명",
        "",
        f"- full-label: 75 sources, {label_data.num_label_pairs:,} judged pairs (gold/LLM 혼합)",
        f"- gold-only: {len(label_data.gold_source_ids)} sources, {label_data.num_gold_pairs:,} judged pairs",
        f"- gold 판별: {label_data.gold_detection}",
        "- full-label 결과는 혼합 라벨 기준이므로 보조 평가로 해석하며, gold-only와 사례 분석을 최종 판단에서 더 중요하게 본다.",
        "",
        "## 5. title_law_name_similarity 생성 방식",
        "",
        "bill_name에서 일부/전부개정법률안, 제정법률안, 개정안, 법률안 등의 suffix를 제거했다. "
        "정규화 법률명을 SBERT cosine으로 비교한 방식과 완전 일치 시 1인 exact 방식을 모두 계산했다.",
        "",
        "## 6. title-heavy weighted_field 탐색 범위",
        "",
        f"0.025 step으로 SBERT {candidate_counts['weighted_field_title_sbert']:,}개, exact "
        f"{candidate_counts['weighted_field_title_exact']:,}개를 평가했다. w_title 범위는 0.30~0.70이며 "
        "w_title=0.00과 0.85 수동 ablation도 포함했다.",
        "",
        "## 7. weighted_field_title_sbert 결과",
        "",
    ]
    report_family_selection(lines, "선택 기준별 결과", sbert)
    lines.extend([
        f"Balanced 추천 full: {metric_text(sbert_bal['metrics_full_label'])}",
        "",
        f"Balanced 추천 gold: {metric_text(sbert_bal['metrics_gold_only'])}",
        "",
        "## 8. weighted_field_title_exact 결과",
        "",
    ])
    report_family_selection(lines, "선택 기준별 결과", exact)
    lines.extend([
        f"Balanced 추천 full: {metric_text(exact_bal['metrics_full_label'])}",
        "",
        f"Balanced 추천 gold: {metric_text(exact_bal['metrics_gold_only'])}",
        "",
        "## 9. title ablation 분석",
        "",
        "| Family | Group | Candidates | w_title | Full P@5 | Full nDCG | Gold P@5 | Gold nDCG | Gold Legal | Balanced |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for family_name, family_payload in (
        ("title_sbert", sbert), ("title_exact", exact)
    ):
        for group, group_payload in family_payload["title_ablation_summary"].items():
            if group_payload.get("candidate_count", 0) == 0:
                continue
            best = group_payload["best_by_balanced_score"]
            lines.append(
                f"| {family_name} | {group} | {group_payload['candidate_count']:,} | "
                f"{best['weights']['w_title']:.3f} | "
                f"{best['metrics_full_label']['precision_at_5']:.4f} | "
                f"{best['metrics_full_label']['ndcg_at_10']:.4f} | "
                f"{best['metrics_gold_only']['precision_at_5']:.4f} | "
                f"{best['metrics_gold_only']['ndcg_at_10']:.4f} | "
                f"{best['metrics_gold_only']['average_legal_meaning_score']:.2f} | "
                f"{best['balanced_score']:.4f} |"
            )
    lines.extend([
        "",
        "## 10. same-law/title 편향 진단",
        "",
        f"- 선택 family: `{selected_wf_family}`",
        f"- v2 → title-heavy same_law_ratio@10 변화: {same_delta:+.4f}",
        f"- gold P@5 변화: {p5_delta:+.4f}",
        f"- gold nDCG@10 변화: {ndcg_delta:+.4f}",
        f"- gold Legal Meaning Score 변화: {legal_delta:+.2f}",
        f"- SBERT w_title와 same_law_ratio@10 상관: {sb_bias['correlation_w_title_same_law_ratio_at_10']}",
        f"- exact w_title와 same_law_ratio@10 상관: {ex_bias['correlation_w_title_same_law_ratio_at_10']}",
        "",
        "same_law_ratio 증가가 P@5 또는 nDCG 개선으로 이어지는지는 위 변화량과 ablation 표를 함께 "
        "봐야 한다. same_law 추천이 늘면서 gold-only 또는 Legal Meaning Score가 하락하는 구간은 "
        "title 과적합, 즉 실제 개정 취지 다양성을 놓치는 신호로 해석한다.",
        "",
        "## 11. final_ensemble 재탐색 결과",
        "",
    ])
    report_family_selection(lines, "선택 기준별 결과", ensemble)
    lines.extend([
        f"최종 balanced 가중치: `{weight_text(final_bal['weights'])}`",
        "",
        f"v2 대비 full objective 변화: {final_bal['metrics_full_label']['objective_score'] - v2_fe['metrics_full_label']['objective_score']:+.4f}",
        "",
        f"v2 대비 gold objective 변화: {final_bal['metrics_gold_only']['objective_score'] - v2_fe['metrics_gold_only']['objective_score']:+.4f}",
        "",
        "## 12. full-label vs gold-only 결과 비교",
        "",
        "full-label objective best와 gold-only objective best가 다를 수 있으므로 두 결과를 모두 저장했다. "
        "최종 추천은 gold 비중이 더 큰 balanced score를 사용한다.",
        "",
        "## 13. train/validation 및 5-fold 검증",
        "",
    ])
    for family in (
        "weighted_field_title_sbert", "weighted_field_title_exact", "final_ensemble_title_heavy"
    ):
        tv = best_payload[family]["train_validation_result"]
        cv = best_payload[family]["cv_average_validation_result"]["average_validation_metrics"]
        lines.append(
            f"- {family}: train best `{weight_text(tv['train_best_weights'])}` / "
            f"validation {metric_text(tv['validation_metrics'])} / 5-fold avg {metric_text(cv)}"
        )
    lines.extend([
        "",
        "## 14. 최종 추천 가중치",
        "",
        f"- weighted_field family: `{selected_wf_family}`",
        f"- weighted_field balanced: `{weight_text(selected_payload['weights'])}`",
        f"- final_ensemble balanced: `{weight_text(final_bal['weights'])}`",
        "",
        "full-label best와 gold-only best는 각각 별도 항목으로 best JSON에 기록했다. 실제 채택은 balanced "
        "추천, gold-only 지표, CV 안정성, 사례 검토를 함께 판단해야 한다.",
        "",
        "## 15. 사례 분석 샘플 요약",
        "",
        f"- 순위 상승 후보: {case_summary['title_heavy_rank_rise_selected']}개",
        f"- 순위 하락 후보: {case_summary['title_heavy_rank_fall_selected']}개",
        f"- same-law 저관련 후보: {case_summary['same_law_low_relevance_selected']}개",
        f"- different-law 고관련 후보: {case_summary['different_law_high_relevance_selected']}개",
        f"- 중복 제거 후 {case_summary['unique_output_rows_after_deduplication']}개, gold source "
        f"{case_summary['gold_source_rows']}개 row를 저장했다.",
        "",
        "## 16. 한계와 다음 단계",
        "",
        "- pooled qrels에 없는 추천은 0점 similarity로 만들지 않고 unjudged로 제외했다. 완전 판단 qrels 확대가 필요하다.",
        "- exact same-law는 법률명이 다른 연관 법체계를 잡지 못하고, SBERT title은 유사 법률명을 과대평가할 수 있다.",
        "- title-only SBERT와 exact baseline도 결과 CSV/JSON에 포함했으며, 문맥 component가 없는 순수 title 성능과 비교해야 한다.",
        "- title 비중을 높이면 같은 법률명 또는 같은 법체계 추천이 증가할 수 있으므로 실제 사용자 사례 검증이 필요하다.",
        "- row-minmax normalized score는 절대적 동일도가 아니라 source별 상대 점수다.",
        "- 추천 이유에서 normalized score를 ‘유사도 1.0’이라고 표현하지 말고 원본 점수 또는 높음/중간/낮음 등급을 사용해야 한다.",
    ])
    OUTPUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_outputs(results: pd.DataFrame) -> None:
    expected_baselines = {
        "raw", "structured", "problem_proposal", "cleaned_problem_proposal",
        "hybrid_cleaned_fine_best", "weighted_field_fine_best_v2",
        "final_ensemble_fine_best_v2",
    }
    baseline_names = set(
        results.loc[results["method_family"] == "baseline", "candidate_name"]
    )
    if not expected_baselines.issubset(baseline_names):
        raise RuntimeError(
            f"[baseline 검증] 누락={sorted(expected_baselines - baseline_names)}"
        )
    for family in (
        "weighted_field_title_sbert", "weighted_field_title_exact",
        "final_ensemble_title_heavy",
    ):
        family_rows = results[results["method_family"] == family]
        if not family_rows["is_v2_best_candidate"].any():
            raise RuntimeError(f"[v2 후보 검증] {family}에 v2 best 후보가 없습니다.")
    for family in ("weighted_field_title_sbert", "weighted_field_title_exact"):
        names = set(results.loc[results["method_family"] == family, "candidate_name"])
        for required in ("WF_title_ablation_zero", "WF_title_only_like"):
            if required not in names:
                raise RuntimeError(f"[title ablation 검증] {family}에 {required} 누락")
    if not results["same_law_ratio_at_10"].between(0, 1).all():
        raise RuntimeError("[same-law ratio 검증] 0~1 범위를 벗어난 결과가 있습니다.")
    outputs = [
        OUTPUT_CSV, OUTPUT_JSON, OUTPUT_BEST_JSON, OUTPUT_TOPK_SBERT,
        OUTPUT_TOPK_EXACT, OUTPUT_TOPK_ENSEMBLE, OUTPUT_CASES, OUTPUT_REPORT,
    ]
    missing = [str(path) for path in outputs if not path.exists() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"[출력 검증] 누락/빈 파일: {missing}")


def main() -> None:
    started = time.time()
    print("=" * 92)
    print("3차 Title-Heavy Weighted Field Search")
    print("실행 방법: python 26_title_heavy_weighted_field_search.py")
    print("=" * 92)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/10] 2차 입력 및 top-k 무결성 검증")
    v2_inputs = validate_v2_inputs()
    v2_weights = validate_v2_weights(v2_inputs["best"])

    print("[2/10] 데이터셋, full-label, gold-only 로드")
    dataset = json_load(DATASET_JSON)
    bills = dataset.get("bills", [])
    if len(bills) != 75:
        raise ValueError(f"[데이터셋 검증] 법안 수={len(bills)}, expected=75")
    bill_ids = [str(bill.get("bill_id", "")).strip() for bill in bills]
    if any(not value for value in bill_ids) or len(set(bill_ids)) != 75:
        raise ValueError("[데이터셋 검증] bill_id가 비었거나 중복되었습니다.")
    law_names = [normalize_law_name(str(bill.get("bill_name", ""))) for bill in bills]
    labels = FINE.load_labels(bill_ids)
    print(
        f"    full={len(labels.full_source_ids)} sources/{labels.num_label_pairs:,} pairs, "
        f"gold={len(labels.gold_source_ids)} sources/{labels.num_gold_pairs:,} pairs"
    )

    print("[3/10] 전체 75×75 공통 component matrix 재계산")
    raw_matrices, _ = FINE.build_component_matrices(bills)
    title_sbert_raw, title_exact_raw = build_title_matrices(law_names)
    raw_matrices["title_law_name_similarity"] = title_sbert_raw
    raw_matrices["title_exact_or_same_law_score"] = title_exact_raw
    matrix_bill_ids = {name: bill_ids.copy() for name in raw_matrices}
    matrix_validation = FINE.validate_matrices(raw_matrices, bill_ids, matrix_bill_ids)

    print("[4/10] row-minmax normalization 및 2차 baseline 재구성")
    normalization_stats: dict[str, Any] = {}
    non_title_raw = {
        "full_text_similarity": raw_matrices["full_text_similarity"],
        "current_law_similarity": raw_matrices["current_law_similarity"],
        "problem_similarity": raw_matrices["problem_similarity"],
        "proposal_similarity": raw_matrices["proposal_similarity"],
        "article_similarity": raw_matrices["article_similarity"],
    }
    non_title_norm = {
        name: normalize_component(
            matrix, f"WF {name}", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD,
            normalization_stats,
        )
        for name, matrix in non_title_raw.items()
    }
    title_sbert_norm = normalize_component(
        title_sbert_raw, "WF title_law_name_similarity",
        NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD, normalization_stats,
    )
    title_exact_norm = normalize_component(
        title_exact_raw, "WF title_exact_or_same_law_score",
        NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD, normalization_stats,
    )
    wf_sbert_norm = {"title_law_name_similarity": title_sbert_norm, **non_title_norm}
    wf_sbert_raw = {"title_law_name_similarity": title_sbert_raw, **non_title_raw}
    wf_exact_norm = {"title_exact_or_same_law_score": title_exact_norm, **non_title_norm}
    wf_exact_raw = {"title_exact_or_same_law_score": title_exact_raw, **non_title_raw}
    non_title_average_norm = np.mean(np.stack(list(non_title_norm.values())), axis=0)

    unique_laws = {name: index for index, name in enumerate(sorted(set(law_names)))}
    law_group_ids = np.asarray([unique_laws[name] for name in law_names], dtype=int)
    same_law_matrix = title_exact_raw.astype(bool)
    np.fill_diagonal(same_law_matrix, False)

    hybrid_raw_components = {
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "keyword_tfidf_score": raw_matrices["keyword_tfidf_score"],
        "article_similarity": raw_matrices["article_similarity"],
    }
    hybrid_norm_components = {
        name: normalize_component(
            matrix, f"HC-v2 {name}", True, normalization_stats
        )
        for name, matrix in hybrid_raw_components.items()
    }
    hybrid_v2_norm = weighted_matrix(
        hybrid_norm_components.values(), v2_weights["hybrid"],
        ["w_cleaned", "w_tfidf", "w_article"],
    )
    hybrid_v2_raw = weighted_matrix(
        hybrid_raw_components.values(), v2_weights["hybrid"],
        ["w_cleaned", "w_tfidf", "w_article"],
    )
    wf_v2_norm = weighted_matrix(wf_sbert_norm.values(), v2_weights["weighted"], WEIGHTED_KEYS)
    wf_v2_raw = weighted_matrix(wf_sbert_raw.values(), v2_weights["weighted"], WEIGHTED_KEYS)
    wf_v2_full, wf_v2_gold, _, _ = evaluate_matrix(
        wf_v2_norm, bill_ids, labels, same_law_matrix, law_group_ids,
        title_sbert_norm, non_title_average_norm,
    )

    fe_v2_pre = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_score": wf_v2_norm,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_score": hybrid_v2_norm,
    }
    fe_v2_norm_components = {
        name: normalize_component(
            matrix, f"FE-v2 {name}", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE,
            normalization_stats,
        )
        for name, matrix in fe_v2_pre.items()
    }
    final_v2_norm = weighted_matrix(
        fe_v2_norm_components.values(), v2_weights["ensemble"], ENSEMBLE_KEYS
    )
    fe_v2_full, fe_v2_gold, _, _ = evaluate_matrix(
        final_v2_norm, bill_ids, labels, same_law_matrix, law_group_ids,
        title_sbert_norm, non_title_average_norm,
    )

    source_splits = FINE.build_source_splits(bill_ids, labels.full_source_ids)
    weighted_candidates = make_candidates(
        [(0.30, 0.70), (0.05, 0.25), (0.00, 0.15), (0.05, 0.25), (0.05, 0.25), (0.00, 0.15)],
        WEIGHTED_KEYS,
        WEIGHTED_FIELD_TITLE_HEAVY_MANUAL_CANDIDATES,
        "WF_TITLE",
    )
    ensemble_candidates = make_candidates(
        [(0.00, 0.15), (0.05, 0.20), (0.00, 0.20), (0.20, 0.45), (0.00, 0.05), (0.35, 0.60)],
        ENSEMBLE_KEYS,
        FINAL_ENSEMBLE_TITLE_HEAVY_MANUAL_CANDIDATES,
        "FE_TITLE",
    )
    print(
        f"    candidates: WF family당 {len(weighted_candidates):,}, "
        f"FE={len(ensemble_candidates):,}"
    )

    print("[5/10] weighted_field_title_sbert 탐색")
    sbert_result = search_family(
        "weighted_field_title_sbert", weighted_candidates, WEIGHTED_KEYS,
        list(wf_sbert_norm.values()), bill_ids, labels, source_splits,
        same_law_matrix, law_group_ids, title_sbert_norm, non_title_average_norm,
        wf_v2_full, wf_v2_gold, "weighted",
    )
    print("[6/10] weighted_field_title_exact 탐색")
    exact_result = search_family(
        "weighted_field_title_exact", weighted_candidates, WEIGHTED_KEYS,
        list(wf_exact_norm.values()), bill_ids, labels, source_splits,
        same_law_matrix, law_group_ids, title_exact_norm, non_title_average_norm,
        wf_v2_full, wf_v2_gold, "weighted",
    )

    sbert_balanced_index = sbert_result.best_indices["best_by_balanced_score"]
    exact_balanced_index = exact_result.best_indices["best_by_balanced_score"]
    if sbert_result.balanced_scores[sbert_balanced_index] >= exact_result.balanced_scores[exact_balanced_index]:
        selected_wf_family = "weighted_field_title_sbert"
        selected_wf_result = sbert_result
        selected_wf_index = sbert_balanced_index
        selected_norm_components = wf_sbert_norm
        selected_raw_components = wf_sbert_raw
        selected_title_norm = title_sbert_norm
        selected_title_raw = title_sbert_raw
    else:
        selected_wf_family = "weighted_field_title_exact"
        selected_wf_result = exact_result
        selected_wf_index = exact_balanced_index
        selected_norm_components = wf_exact_norm
        selected_raw_components = wf_exact_raw
        selected_title_norm = title_exact_norm
        selected_title_raw = title_exact_raw
    selected_wf_weights = {
        key: float(selected_wf_result.candidates[selected_wf_index][key])
        for key in WEIGHTED_KEYS
    }
    selected_wf_norm = weighted_matrix(
        selected_norm_components.values(), selected_wf_weights, WEIGHTED_KEYS
    )
    selected_wf_raw = weighted_matrix(
        selected_raw_components.values(), selected_wf_weights, WEIGHTED_KEYS
    )
    print(f"    FE 입력 weighted_field 선택: {selected_wf_family} {selected_wf_weights}")

    print("[7/10] selected weighted_field를 사용한 final_ensemble 재탐색")
    final_raw_components = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_title_best_score": selected_wf_raw,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_fine_best_score": hybrid_v2_raw,
    }
    final_pre_normalization = {
        "raw_score": raw_matrices["raw_score"],
        "structured_score": raw_matrices["structured_score"],
        "problem_proposal_score": raw_matrices["problem_proposal_score"],
        "weighted_field_title_best_score": selected_wf_norm,
        "cleaned_problem_proposal_score": raw_matrices["cleaned_problem_proposal_score"],
        "hybrid_cleaned_fine_best_score": hybrid_v2_norm,
    }
    final_norm_components = {
        name: normalize_component(
            matrix, f"FE-title {name}", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE,
            normalization_stats,
        )
        for name, matrix in final_pre_normalization.items()
    }
    ensemble_result = search_family(
        "final_ensemble_title_heavy", ensemble_candidates, ENSEMBLE_KEYS,
        list(final_norm_components.values()), bill_ids, labels, source_splits,
        same_law_matrix, law_group_ids, selected_title_norm, non_title_average_norm,
        fe_v2_full, fe_v2_gold, "ensemble",
    )

    print("[8/10] baseline, best JSON, 전체 결과 CSV/JSON 저장")
    baseline_matrices = {
        "raw": (raw_matrices["raw_score"], {}, False),
        "structured": (raw_matrices["structured_score"], {}, False),
        "problem_proposal": (raw_matrices["problem_proposal_score"], {}, False),
        "cleaned_problem_proposal": (
            raw_matrices["cleaned_problem_proposal_score"], {}, False
        ),
        "title_law_name_sbert_only": (
            title_sbert_norm,
            {"w_title": 1.0, "title_component": "title_law_name_similarity"},
            True,
        ),
        "title_exact_same_law_only": (
            title_exact_norm,
            {"w_title": 1.0, "title_component": "title_exact_or_same_law_score"},
            True,
        ),
        "hybrid_cleaned_fine_best": (hybrid_v2_norm, v2_weights["hybrid"], True),
        "weighted_field_fine_best_v2": (wf_v2_norm, v2_weights["weighted"], True),
        "final_ensemble_fine_best_v2": (final_v2_norm, v2_weights["ensemble"], True),
    }
    baselines_payload, baseline_rows = baseline_payload_and_rows(
        baseline_matrices, bill_ids, labels, same_law_matrix, law_group_ids,
        title_sbert_norm, non_title_average_norm,
    )
    best_payload = {
        "weighted_field_title_sbert": family_payload(sbert_result, include_ablation=True),
        "weighted_field_title_exact": family_payload(exact_result, include_ablation=True),
        "final_ensemble_title_heavy": family_payload(ensemble_result, include_ablation=False),
        "baselines": baselines_payload,
        "metadata": {
            "step": STEP,
            "top_k": TOP_K,
            "random_seed": RANDOM_SEED,
            "selected_weighted_field_family_for_final_ensemble": selected_wf_family,
            "selected_weighted_field_weights": selected_wf_weights,
            "normalization_method": NORMALIZATION_METHOD,
            "normalization_enabled": {
                "weighted_field": NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD,
                "final_ensemble": NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE,
            },
            "weighted_legal_penalty_scope": "gold_only",
            "normalization_stats": normalization_stats,
            "matrix_validation": matrix_validation,
            "v2_input_validation": v2_inputs,
        },
    }
    json_dump(OUTPUT_BEST_JSON, best_payload)

    rows = [
        *family_rows(sbert_result, v2_weights["weighted"]),
        *family_rows(exact_result, v2_weights["weighted"]),
        *family_rows(ensemble_result, v2_weights["ensemble"]),
        *baseline_rows,
    ]
    csv_columns = [
        "method_family", "candidate_name", "weights_json", "label_scope",
        "precision_at_5", "precision_at_10", "ndcg_at_10", "mrr",
        "average_relevance", "average_legal_meaning_score", "objective_score",
        "candidate_recall_objective", "ranking_objective", "balanced_score",
        "same_law_ratio_at_5", "same_law_ratio_at_10",
        "unique_target_law_count_avg_at_10", "avg_title_similarity_at_10",
        "avg_non_title_component_score_at_10", "num_evaluated_pairs",
        "num_unlabeled_pairs", "num_sources", "normalization_enabled",
        "is_manual_candidate", "is_v2_best_candidate",
    ]
    results_df = pd.DataFrame(rows)[csv_columns]
    results_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    candidate_counts = {
        "weighted_field_title_sbert": len(weighted_candidates),
        "weighted_field_title_exact": len(weighted_candidates),
        "final_ensemble_title_heavy": len(ensemble_candidates),
        "baseline": len(baseline_matrices),
    }
    json_dump(OUTPUT_JSON, {
        "metadata": best_payload["metadata"],
        "label_data": {
            "full_label_source_count": len(labels.full_source_ids),
            "full_label_pair_count": labels.num_label_pairs,
            "gold_only_source_count": len(labels.gold_source_ids),
            "gold_only_pair_count": labels.num_gold_pairs,
            "gold_detection": labels.gold_detection,
            "unjudged_topk_policy": "지표 집계에서 제외; similarity를 0으로 대체하지 않음",
        },
        "candidate_counts": candidate_counts,
        "best_weights": {
            key: value for key, value in best_payload.items() if key != "metadata"
        },
        "results": rows,
    })

    print("[9/10] balanced top-k 및 사례 검토 샘플 저장")
    sbert_weights = {
        key: float(sbert_result.candidates[sbert_balanced_index][key]) for key in WEIGHTED_KEYS
    }
    sbert_best_norm = weighted_matrix(wf_sbert_norm.values(), sbert_weights, WEIGHTED_KEYS)
    sbert_best_raw = weighted_matrix(wf_sbert_raw.values(), sbert_weights, WEIGHTED_KEYS)
    exact_weights = {
        key: float(exact_result.candidates[exact_balanced_index][key]) for key in WEIGHTED_KEYS
    }
    exact_best_norm = weighted_matrix(wf_exact_norm.values(), exact_weights, WEIGHTED_KEYS)
    exact_best_raw = weighted_matrix(wf_exact_raw.values(), exact_weights, WEIGHTED_KEYS)
    ensemble_best_index = ensemble_result.best_indices["best_by_balanced_score"]
    ensemble_weights = {
        key: float(ensemble_result.candidates[ensemble_best_index][key])
        for key in ENSEMBLE_KEYS
    }
    ensemble_best_norm = weighted_matrix(
        final_norm_components.values(), ensemble_weights, ENSEMBLE_KEYS
    )
    ensemble_best_raw = weighted_matrix(
        final_raw_components.values(), ensemble_weights, ENSEMBLE_KEYS
    )
    json_dump(OUTPUT_TOPK_SBERT, topk_records(
        sbert_best_norm, sbert_best_raw, bills, law_names,
        "weighted_field_title_sbert_best", sbert_weights,
        wf_sbert_norm, wf_sbert_raw,
    ))
    json_dump(OUTPUT_TOPK_EXACT, topk_records(
        exact_best_norm, exact_best_raw, bills, law_names,
        "weighted_field_title_exact_best", exact_weights,
        wf_exact_norm, wf_exact_raw,
    ))
    json_dump(OUTPUT_TOPK_ENSEMBLE, topk_records(
        ensemble_best_norm, ensemble_best_raw, bills, law_names,
        "final_ensemble_title_heavy_best", ensemble_weights,
        final_norm_components, final_raw_components,
    ))
    case_summary = write_case_review_sample(
        wf_v2_norm, selected_wf_norm, bill_ids, bills, law_names, labels,
        selected_title_raw, non_title_average_norm,
    )
    best_payload["metadata"]["case_review_summary"] = case_summary
    json_dump(OUTPUT_BEST_JSON, best_payload)
    write_report(
        best_payload, candidate_counts, labels, selected_wf_family, case_summary
    )

    print("[10/10] 신규 산출물 검증")
    verify_outputs(results_df)
    elapsed = time.time() - started
    print(f"완료: {elapsed:.1f}초")
    print("다음 파일을 확인하세요:")
    for path in (
        OUTPUT_REPORT, OUTPUT_CSV, OUTPUT_BEST_JSON, OUTPUT_TOPK_SBERT,
        OUTPUT_TOPK_EXACT, OUTPUT_TOPK_ENSEMBLE, OUTPUT_CASES,
    ):
        print(f"  - {path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
