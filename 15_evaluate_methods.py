#!/usr/bin/env python3
"""
15_evaluate_methods.py
=======================
사람이 채운 evaluation_pooled_label_template.csv를 읽고
메소드별 성과 지표(Precision@5, Precision@10, nDCG@10, MRR, Average Relevance,
Legal Meaning Score)를 계산한다.

실행 방법:
    python 15_evaluate_methods.py

입력 파일:
    Sbert_output/evaluation_pooled_label_template.csv

출력 파일:
    Sbert_output/method_evaluation_metrics.csv
    Sbert_output/method_evaluation_metrics.json
"""

import json
import math
import sys
import os
from collections import defaultdict

import pandas as pd

# ── Windows 한글 출력 설정 ───────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 정의 ──────────────────────────────────────────────────────
INPUT_CSV = "Sbert_output/evaluation_pooled_label_template.csv"
INPUT_EXCEL = "Sbert_output/evaluation_pooled_label_template.xlsx"
OUTPUT_CSV = "Sbert_output/method_evaluation_metrics.csv"
OUTPUT_JSON = "Sbert_output/method_evaluation_metrics.json"

METHODS = ["raw", "structured", "problem_proposal", "weighted_field"]

RELEVANCE_THRESHOLD = 3  # human_relevance >= 3 이면 관련 있음
MIN_PAIRS_WARNING = 5    # 메소드별 평가 대상이 이 수 미만이면 경고

LEGAL_COLUMNS = [
    "human_issue_match_0_to_2",
    "human_target_match_0_to_2",
    "human_effect_match_0_to_2",
    "human_scope_match_0_to_2",
    "human_article_match_0_to_2",
]

LEGAL_WEIGHTS = {
    "human_issue_match_0_to_2": 0.25,
    "human_target_match_0_to_2": 0.25,
    "human_effect_match_0_to_2": 0.25,
    "human_scope_match_0_to_2": 0.15,
    "human_article_match_0_to_2": 0.10,
}


def safe_float(val, default=None):
    """값을 float로 변환, 실패 시 default 반환."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def validate_data(df: pd.DataFrame):
    """데이터 유효성 검사 및 경고 출력."""
    warnings = []

    # 1. human_relevance_0_to_4가 비어 있는 행
    empty_relevance = df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is None
    ).sum()
    if empty_relevance > 0:
        warnings.append(
            f"[WARNING] human_relevance_0_to_4가 비어 있는 행: {empty_relevance}개 "
            f"(전체 {len(df)}개 중). 이 행들은 평가에서 제외됩니다."
        )

    # 2. relevance 값 범위 검사 (0~4)
    for idx, row in df.iterrows():
        val = safe_float(row["human_relevance_0_to_4"])
        if val is not None and (val < 0 or val > 4):
            warnings.append(
                f"[WARNING] 행 {idx}: human_relevance_0_to_4 값이 범위 밖입니다: {val} (0~4 범위 필요)"
            )

    # 3. 세부 항목 범위 검사 (0~2)
    for col in LEGAL_COLUMNS:
        if col in df.columns:
            for idx, row in df.iterrows():
                val = safe_float(row[col])
                if val is not None and (val < 0 or val > 2):
                    warnings.append(
                        f"[WARNING] 행 {idx}: {col} 값이 범위 밖입니다: {val} (0~2 범위 필요)"
                    )

    for w in warnings:
        print(w)
    print()

    return warnings


def compute_dcg(relevance_scores: list[float], k: int) -> float:
    """DCG@k를 계산한다."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)  # i+2 because i is 0-indexed, rank starts at 1
    return dcg


def evaluate_method(df: pd.DataFrame, method: str) -> dict:
    """특정 메소드의 성과 지표를 계산한다."""
    rank_col = f"{method}_rank"
    score_col = f"{method}_score"

    # 해당 메소드의 rank가 존재하는 행만 필터링
    method_df = df[df[rank_col].apply(lambda x: safe_float(x) is not None)].copy()

    # relevance 라벨이 있는 행만 평가
    method_df = method_df[method_df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is not None
    )].copy()

    if len(method_df) == 0:
        return {
            "method": method,
            "num_evaluated_pairs": 0,
            "num_sources": 0,
            "precision_at_5": None,
            "precision_at_10": None,
            "ndcg_at_10": None,
            "mrr": None,
            "average_relevance": None,
            "average_legal_meaning_score": None,
        }

    # 4. 라벨이 없는 행 제외 후 경고
    if len(method_df) < MIN_PAIRS_WARNING:
        print(f"[WARNING] {method}: 평가 대상 행이 {len(method_df)}개로 매우 적습니다.")

    # 각 행에 float 변환된 값 추가
    method_df["_rank"] = method_df[rank_col].apply(lambda x: safe_float(x))
    method_df["_relevance"] = method_df["human_relevance_0_to_4"].apply(lambda x: safe_float(x, 0.0))

    # ── source별 그룹 ───────────────────────────────────────────────
    source_groups = defaultdict(list)
    for _, row in method_df.iterrows():
        source_groups[row["source_bill_id"]].append({
            "rank": row["_rank"],
            "relevance": row["_relevance"],
        })

    # source별 rank 순 정렬
    for sid in source_groups:
        source_groups[sid].sort(key=lambda x: x["rank"])

    num_sources = len(source_groups)

    # ── A. Precision@5 ──────────────────────────────────────────────
    precision_at_5_list = []
    for sid, items in source_groups.items():
        top5 = items[:5]
        if len(top5) == 0:
            continue
        relevant_count = sum(1 for item in top5 if item["relevance"] >= RELEVANCE_THRESHOLD)
        denom = min(len(top5), 5)
        if denom < 5:
            print(f"  [INFO] {method} - source {sid}: top5 결과가 {denom}개뿐입니다.")
        precision_at_5_list.append(relevant_count / denom)

    precision_at_5 = sum(precision_at_5_list) / len(precision_at_5_list) if precision_at_5_list else 0.0

    # ── B. Precision@10 ─────────────────────────────────────────────
    precision_at_10_list = []
    for sid, items in source_groups.items():
        top10 = items[:10]
        if len(top10) == 0:
            continue
        relevant_count = sum(1 for item in top10 if item["relevance"] >= RELEVANCE_THRESHOLD)
        denom = min(len(top10), 10)
        if denom < 10:
            print(f"  [INFO] {method} - source {sid}: top10 결과가 {denom}개뿐입니다.")
        precision_at_10_list.append(relevant_count / denom)

    precision_at_10 = sum(precision_at_10_list) / len(precision_at_10_list) if precision_at_10_list else 0.0

    # ── C. nDCG@10 ──────────────────────────────────────────────────
    ndcg_at_10_list = []
    for sid, items in source_groups.items():
        top10 = items[:10]
        if len(top10) == 0:
            continue

        actual_rels = [item["relevance"] for item in top10]
        dcg = compute_dcg(actual_rels, 10)

        # IDCG: 해당 메소드 결과들의 relevance를 내림차순 정렬한 DCG
        ideal_rels = sorted(actual_rels, reverse=True)
        idcg = compute_dcg(ideal_rels, 10)

        if idcg == 0:
            ndcg_at_10_list.append(0.0)
        else:
            ndcg_at_10_list.append(dcg / idcg)

    ndcg_at_10 = sum(ndcg_at_10_list) / len(ndcg_at_10_list) if ndcg_at_10_list else 0.0

    # ── D. MRR ──────────────────────────────────────────────────────
    rr_list = []
    for sid, items in source_groups.items():
        found = False
        for i, item in enumerate(items):
            if item["relevance"] >= RELEVANCE_THRESHOLD:
                rr_list.append(1.0 / (i + 1))
                found = True
                break
        if not found:
            rr_list.append(0.0)

    mrr = sum(rr_list) / len(rr_list) if rr_list else 0.0

    # ── E. Average Relevance ────────────────────────────────────────
    all_relevances = method_df["_relevance"].tolist()
    avg_relevance = sum(all_relevances) / len(all_relevances) if all_relevances else 0.0

    # ── F. Legal Meaning Score ──────────────────────────────────────
    legal_scores = []
    for _, row in method_df.iterrows():
        vals = {}
        all_filled = True
        for col in LEGAL_COLUMNS:
            v = safe_float(row.get(col))
            if v is None:
                all_filled = False
                break
            vals[col] = v

        if all_filled:
            lms = sum(LEGAL_WEIGHTS[col] * vals[col] for col in LEGAL_COLUMNS)
            lms_100 = lms / 2 * 100
            legal_scores.append(lms_100)

    avg_legal_meaning = (sum(legal_scores) / len(legal_scores)) if legal_scores else None

    return {
        "method": method,
        "num_evaluated_pairs": len(method_df),
        "num_sources": num_sources,
        "precision_at_5": round(precision_at_5, 4),
        "precision_at_10": round(precision_at_10, 4),
        "ndcg_at_10": round(ndcg_at_10, 4),
        "mrr": round(mrr, 4),
        "average_relevance": round(avg_relevance, 4),
        "average_legal_meaning_score": round(avg_legal_meaning, 2) if avg_legal_meaning is not None else None,
    }


def main():
    print("=" * 70)
    print("  법률 유사도 메소드별 평가 지표 계산")
    print("=" * 70)
    print()

    # ── 입력 파일 로드 ──────────────────────────────────────────────
    if os.path.exists(INPUT_EXCEL):
        print(f"[1] Excel 입력 파일 발견: {INPUT_EXCEL} 로드 중...")
        df = pd.read_excel(INPUT_EXCEL, dtype=str)
    elif os.path.exists(INPUT_CSV):
        print(f"[1] CSV 입력 파일 발견: {INPUT_CSV} 로드 중...")
        df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig", dtype=str)
    else:
        print(f"[ERROR] 입력 파일을 찾을 수 없습니다.")
        print(f"  - {INPUT_EXCEL} (권장)")
        print(f"  - {INPUT_CSV}")
        print("먼저 14_build_pooled_evaluation_set.py를 실행하고 라벨을 입력하세요.")
        sys.exit(1)

    df = df.fillna("")
    print(f"    데이터 로드 완료: {len(df)}행")
    print()

    # ── 데이터 유효성 검사 ──────────────────────────────────────────
    print("[2] 데이터 유효성 검사")
    validate_data(df)

    # ── 라벨이 없는 행 확인 ────────────────────────────────────────
    labeled_count = df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is not None
    ).sum()
    unlabeled_count = len(df) - labeled_count
    print(f"  라벨 입력된 행: {labeled_count}개 / 전체 {len(df)}개")
    if unlabeled_count > 0:
        print(f"  라벨 미입력 행: {unlabeled_count}개 (평가에서 제외)")
    print()

    if labeled_count == 0:
        print("[ERROR] 라벨이 입력된 행이 하나도 없습니다.")
        print("CSV 파일에서 human_relevance_0_to_4 컬럼에 0~4점을 입력하세요.")
        sys.exit(1)

    # ── 메소드별 평가 ──────────────────────────────────────────────
    print("[3] 메소드별 평가 지표 계산")
    print()
    metrics_list = []
    for method in METHODS:
        print(f"  ▶ {method} 평가 중...")
        metrics = evaluate_method(df, method)
        metrics_list.append(metrics)
        print(f"    평가 대상: {metrics['num_evaluated_pairs']}쌍, "
              f"source {metrics['num_sources']}개")
        print()

    # ── 결과 저장 ──────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metrics_list)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    metrics_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[출력] CSV 저장 완료: {OUTPUT_CSV}")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "description": "법률 유사도 알고리즘 메소드별 평가 결과",
            "relevance_threshold": RELEVANCE_THRESHOLD,
            "metrics": metrics_list,
        }, f, ensure_ascii=False, indent=2)
    print(f"[출력] JSON 저장 완료: {OUTPUT_JSON}")
    print()

    # ── 콘솔 출력: 성과표 ──────────────────────────────────────────
    print("=" * 90)
    print("  메소드별 성과 비교표")
    print("=" * 90)
    header = f"{'method':<22s} {'P@5':>6s} {'P@10':>6s} {'nDCG@10':>8s} {'MRR':>6s} {'AvgRel':>7s} {'LegalMeaning':>13s}"
    print(header)
    print("-" * 90)
    for m in metrics_list:
        lm = f"{m['average_legal_meaning_score']:.1f}" if m['average_legal_meaning_score'] is not None else "-"
        p5 = f"{m['precision_at_5']:.4f}" if m['precision_at_5'] is not None else "-"
        p10 = f"{m['precision_at_10']:.4f}" if m['precision_at_10'] is not None else "-"
        ndcg = f"{m['ndcg_at_10']:.4f}" if m['ndcg_at_10'] is not None else "-"
        mrr_v = f"{m['mrr']:.4f}" if m['mrr'] is not None else "-"
        avg_r = f"{m['average_relevance']:.4f}" if m['average_relevance'] is not None else "-"
        print(f"{m['method']:<22s} {p5:>6s} {p10:>6s} {ndcg:>8s} {mrr_v:>6s} {avg_r:>7s} {lm:>13s}")
    print("=" * 90)
    print()
    print("다음 단계: python 16_generate_evaluation_report.py")


if __name__ == "__main__":
    main()
