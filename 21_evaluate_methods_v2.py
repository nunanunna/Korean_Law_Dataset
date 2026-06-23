#!/usr/bin/env python3
"""
21_evaluate_methods_v2.py
=========================
v2 풀 데이터셋에서 라벨링이 완료된 쌍(기존 359쌍)을 기준으로
7가지 유사도 알고리즘의 성과 지표(P@5, P@10, nDCG@10, MRR 등)를 평가합니다.

실행 방법:
    python 21_evaluate_methods_v2.py
"""

import os
import sys
import math
import json
from collections import defaultdict
import pandas as pd

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 정의 ──────────────────────────────────────────────────────
INPUT_EXCEL_V2 = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"
INPUT_CSV_V2 = "Sbert_output/evaluation_pooled_label_template_v2.csv"
INPUT_EXCEL_V1 = "Sbert_output/evaluation_pooled_label_template.xlsx"
INPUT_CSV_V1 = "Sbert_output/evaluation_pooled_label_template.csv"

OUTPUT_CSV = "Sbert_output/method_evaluation_metrics_v2.csv"
OUTPUT_JSON = "Sbert_output/method_evaluation_metrics_v2.json"

METHODS = [
    "raw",
    "structured",
    "problem_proposal",
    "weighted_field",
    "cleaned_problem_proposal",
    "keyword_tfidf",
    "hybrid_cleaned"
]

RELEVANCE_THRESHOLD = 3.0
MIN_PAIRS_WARNING = 5

LEGAL_COLUMNS = [
    "human_issue_match_0_to_2",
    "human_target_match_0_to_2",
    "human_effect_match_0_to_2",
    "human_scope_match_0_to_2",
    "human_article_match_0_to_2"
]

LEGAL_WEIGHTS = {
    "human_issue_match_0_to_2": 0.3,
    "human_target_match_0_to_2": 0.2,
    "human_effect_match_0_to_2": 0.2,
    "human_scope_match_0_to_2": 0.1,
    "human_article_match_0_to_2": 0.2
}


def safe_float(val, default=None):
    """값을 float로 변환, 실패 시 default 반환"""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def compute_dcg(relevance_scores: list[float], k: int) -> float:
    """DCG@k를 계산합니다."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def evaluate_method(df: pd.DataFrame, method: str) -> dict:
    """특정 메소드의 성과 지표를 계산합니다."""
    rank_col = f"{method}_rank"
    score_col = f"{method}_score"

    # 해당 메소드가 추천(rank 정보 존재)한 행 필터링
    method_all_df = df[df[rank_col].apply(lambda x: safe_float(x) is not None)].copy()
    
    # 그 중 relevance 라벨이 없는(비어있는) 행 개수 파악
    unlabeled_df = method_all_df[method_all_df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is None
    )]
    num_unlabeled = len(unlabeled_df)
    
    # relevance 라벨이 있는(기존 평가된) 행만 평가 대상
    method_df = method_all_df[method_all_df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is not None
    )].copy()

    if len(method_df) == 0:
        return {
            "method": method,
            "num_evaluated_pairs": 0,
            "num_unlabeled_pairs": num_unlabeled,
            "num_sources": 0,
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
            "average_relevance": 0.0,
            "average_legal_meaning_score": None,
        }

    # 각 행에 float 변환된 값 추가
    method_df["_rank"] = method_df[rank_col].apply(lambda x: safe_float(x))
    method_df["_relevance"] = method_df["human_relevance_0_to_4"].apply(lambda x: safe_float(x, 0.0))

    # source별 그룹화
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

        # IDCG
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
        "num_unlabeled_pairs": num_unlabeled,
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
    print("  법률 유사도 메소드별 평가 지표 계산 (v2)")
    print("=" * 70)
    print()

    # ── 입력 파일 로드 ──────────────────────────────────────────────
    if os.path.exists(INPUT_EXCEL_V2):
        print(f"[1] Excel v2 입력 파일 발견: {INPUT_EXCEL_V2} 로드 중...")
        df = pd.read_excel(INPUT_EXCEL_V2, dtype=str)
    elif os.path.exists(INPUT_CSV_V2):
        print(f"[1] CSV v2 입력 파일 발견: {INPUT_CSV_V2} 로드 중...")
        df = pd.read_csv(INPUT_CSV_V2, encoding="utf-8-sig", dtype=str)
    elif os.path.exists(INPUT_EXCEL_V1):
        print(f"[1] [백업] Excel v1 입력 파일 발견: {INPUT_EXCEL_V1} 로드 중...")
        df = pd.read_excel(INPUT_EXCEL_V1, dtype=str)
    elif os.path.exists(INPUT_CSV_V1):
        print(f"[1] [백업] CSV v1 입력 파일 발견: {INPUT_CSV_V1} 로드 중...")
        df = pd.read_csv(INPUT_CSV_V1, encoding="utf-8-sig", dtype=str)
    else:
        print("[ERROR] 입력 파일을 찾을 수 없습니다.")
        print("먼저 20_update_pooled_evaluation_with_new_methods.py를 실행하세요.")
        sys.exit(1)

    df = df.fillna("")
    print(f"    데이터 로드 완료: {len(df)}행")
    print()

    # ── 평가 계산 ──────────────────────────────────────────────────
    results = []
    print("[2] 메소드별 평가 계산 중...")
    
    any_unlabeled_warning = False
    
    for method in METHODS:
        res = evaluate_method(df, method)
        results.append(res)
        
        print(f"  ▶ {method} 완료 (평가쌍 수: {res['num_evaluated_pairs']}개, 미라벨링쌍 수: {res['num_unlabeled_pairs']}개)")
        if res["num_unlabeled_pairs"] > 0:
            any_unlabeled_warning = True
            
    print()
    if any_unlabeled_warning:
        print("[WARNING] 일부 신규 메소드에 라벨링되지 않은 추천 쌍이 존재합니다.")
        print("          이 신규 쌍들은 이번 정확성 평가지표 연산에서 제외되었습니다.")
        print("          보다 엄격하고 공정한 알고리즘 비교를 위해서는")
        print("          v2 엑셀/CSV 파일에서 빈 라벨 칸에 추가 라벨링이 필요합니다.")
        print()

    # ── 결과 저장 ──────────────────────────────────────────────────
    df_metrics = pd.DataFrame(results)
    
    # CSV 저장
    df_metrics.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[출력] 메트릭 CSV 저장 완료: {OUTPUT_CSV}")
    
    # JSON 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[출력] 메트릭 JSON 저장 완료: {OUTPUT_JSON}")
    print()

    # ── 콘솔 요약 비교표 출력 ───────────────────────────────────────
    print("=" * 100)
    print("  7대 메소드별 성과 비교표 (v2)")
    print("=" * 100)
    print(f"{'method':<25} {'P@5':<8} {'P@10':<8} {'nDCG@10':<8} {'MRR':<8} {'AvgRel':<8} {'LegalMeaning':<12}")
    print("-" * 100)
    for res in results:
        p5 = f"{res['precision_at_5']:.4f}" if res['precision_at_5'] is not None else "-"
        p10 = f"{res['precision_at_10']:.4f}" if res['precision_at_10'] is not None else "-"
        ndcg = f"{res['ndcg_at_10']:.4f}" if res['ndcg_at_10'] is not None else "-"
        mrr = f"{res['mrr']:.4f}" if res['mrr'] is not None else "-"
        avg_r = f"{res['average_relevance']:.2f}" if res['average_relevance'] is not None else "-"
        lm = f"{res['average_legal_meaning_score']:.1f}" if res['average_legal_meaning_score'] is not None else "-"
        
        print(f"{res['method']:<25} {p5:<8} {p10:<8} {ndcg:<8} {mrr:<8} {avg_r:<8} {lm:<12}")
    print("=" * 100)
    print("다음 단계: python 22_generate_evaluation_report_v2.py")
    print()

if __name__ == "__main__":
    main()
