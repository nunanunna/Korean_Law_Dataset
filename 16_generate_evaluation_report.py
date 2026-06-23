#!/usr/bin/env python3
"""
16_generate_evaluation_report.py
=================================
평가 결과를 Markdown 리포트로 생성한다.

실행 방법:
    python 16_generate_evaluation_report.py

입력 파일:
    Sbert_output/method_evaluation_metrics.csv
    Sbert_output/evaluation_pooled_label_template.csv

출력 파일:
    Sbert_output/evaluation_report.md
"""

import json
import sys
import os
from collections import defaultdict

import pandas as pd

# ── Windows 한글 출력 설정 ───────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 정의 ──────────────────────────────────────────────────────
METRICS_CSV = "Sbert_output/method_evaluation_metrics.csv"
LABELS_CSV = "Sbert_output/evaluation_pooled_label_template.csv"
LABELS_EXCEL = "Sbert_output/evaluation_pooled_label_template.xlsx"
OUTPUT_MD = "Sbert_output/evaluation_report.md"

METHODS = ["raw", "structured", "problem_proposal", "weighted_field"]

METHOD_DESCRIPTIONS = {
    "raw": "bill_name + summary 전체를 그대로 SBERT 임베딩한 baseline",
    "structured": "summary를 현행법, 문제점, 개정내용, 개정조문으로 분리하고 라벨을 붙여 임베딩한 방식",
    "problem_proposal": "현행법 설명을 제외하고 문제점과 개정내용 중심으로 임베딩한 방식",
    "weighted_field": "full_text, current_law, problem, proposal, article_text를 각각 임베딩하고 가중합한 방식",
}

RELEVANCE_LABELS = """
| 점수 | 의미 |
|:---:|------|
| 4 | **매우 관련 높음.** 적용 대상, 법적 효과, 법률 문제, 조문 체계가 대부분 유사 |
| 3 | **관련 높음.** 적용 대상과 법적 효과가 유사하지만 법률·조문은 다를 수 있음 |
| 2 | **어느 정도 관련.** 같은 정책 이슈를 다루지만 적용 대상 또는 효과가 다름 |
| 1 | **약한 관련.** 넓은 분야만 비슷함 |
| 0 | **무관.** 키워드만 겹치거나 실질적으로 다름 |
"""

MAX_FAILURE_CASES = 10


def safe_float(val, default=None):
    """값을 float로 변환, 실패 시 default 반환."""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def find_best_method(metrics_df: pd.DataFrame) -> str:
    """종합 성능이 가장 좋은 메소드를 찾는다."""
    best_method = None
    best_score = -1
    for _, row in metrics_df.iterrows():
        # 종합 점수: nDCG@10에 가장 높은 가중치
        p5 = safe_float(row.get("precision_at_5"), 0)
        p10 = safe_float(row.get("precision_at_10"), 0)
        ndcg = safe_float(row.get("ndcg_at_10"), 0)
        mrr_val = safe_float(row.get("mrr"), 0)
        composite = 0.2 * p5 + 0.2 * p10 + 0.3 * ndcg + 0.3 * mrr_val
        if composite > best_score:
            best_score = composite
            best_method = row["method"]
    return best_method


def find_failure_cases(labels_df: pd.DataFrame) -> dict:
    """실패 사례 분석 후보를 추출한다."""
    cases = {
        "high_rank_low_relevance": [],   # 높은 순위인데 낮은 관련도
        "method_disagree": [],            # 메소드 간 점수 차이가 큰 경우
        "missed_relevant": [],            # 특정 메소드에서 누락된 관련 법안
    }

    for _, row in labels_df.iterrows():
        rel = safe_float(row.get("human_relevance_0_to_4"))
        if rel is None:
            continue

        s_name = row.get("source_bill_name", "")
        t_name = row.get("target_bill_name", "")

        method_info = {}
        for m in METHODS:
            rank = safe_float(row.get(f"{m}_rank"))
            score = safe_float(row.get(f"{m}_score"))
            method_info[m] = {"rank": rank, "score": score}

        # 1. 높은 순위인데 낮은 관련도 (rank <= 3 but relevance <= 1)
        for m in METHODS:
            r = method_info[m]["rank"]
            if r is not None and r <= 3 and rel <= 1:
                cases["high_rank_low_relevance"].append({
                    "source_bill_name": s_name,
                    "target_bill_name": t_name,
                    "relevance": rel,
                    "problem_method": m,
                    "problem_rank": r,
                    "method_info": {k: v for k, v in method_info.items()},
                })
                break  # 한 쌍에 대해 한 번만 기록

        # 2. 메소드 간 점수 차이가 큰 경우
        scores = [method_info[m]["score"] for m in METHODS if method_info[m]["score"] is not None]
        if len(scores) >= 2:
            score_diff = max(scores) - min(scores)
            if score_diff > 0.15:
                cases["method_disagree"].append({
                    "source_bill_name": s_name,
                    "target_bill_name": t_name,
                    "relevance": rel,
                    "score_diff": round(score_diff, 4),
                    "method_info": {k: v for k, v in method_info.items()},
                })

        # 3. 특정 메소드에서는 상위권이고 다른 메소드에서는 누락됨 + relevance 높음
        if rel >= 3:
            present_methods = [m for m in METHODS if method_info[m]["rank"] is not None]
            absent_methods = [m for m in METHODS if method_info[m]["rank"] is None]
            if present_methods and absent_methods:
                best_present_rank = min(method_info[m]["rank"] for m in present_methods)
                if best_present_rank <= 5:
                    cases["missed_relevant"].append({
                        "source_bill_name": s_name,
                        "target_bill_name": t_name,
                        "relevance": rel,
                        "present_methods": present_methods,
                        "absent_methods": absent_methods,
                        "best_rank": best_present_rank,
                        "method_info": {k: v for k, v in method_info.items()},
                    })

    # 각 카테고리에서 최대 MAX_FAILURE_CASES개만
    for key in cases:
        cases[key] = cases[key][:MAX_FAILURE_CASES]

    return cases


def format_method_info_table(method_info: dict) -> str:
    """메소드 정보를 간략한 텍스트로 포맷한다."""
    parts = []
    for m in METHODS:
        r = method_info[m]["rank"]
        s = method_info[m]["score"]
        r_str = f"#{int(r)}" if r is not None else "N/A"
        s_str = f"{s:.4f}" if s is not None else "N/A"
        parts.append(f"{m}({r_str}, {s_str})")
    return " | ".join(parts)


def generate_report(metrics_df: pd.DataFrame, labels_df: pd.DataFrame) -> str:
    """Markdown 리포트를 생성한다."""
    lines = []

    # ── 헤더 ────────────────────────────────────────────────────────
    lines.append("# SSK-Law 유사도 알고리즘 평가 리포트")
    lines.append("")

    # ── 1. 평가 목적 ────────────────────────────────────────────────
    lines.append("## 1. 평가 목적")
    lines.append("")
    lines.append("본 리포트는 국회 법률발의안 75개를 대상으로 여러 SBERT 기반 유사도 알고리즘의 성과를 비교·분석한 결과를 정리한 것이다.")
    lines.append("사람이 직접 0~4점 관련도 라벨을 부여한 뒤, 메소드별 Precision, nDCG, MRR 등 표준 IR 지표를 계산하여 어떤 방식이 법률 유사도 측정에 가장 효과적인지 판단한다.")
    lines.append("")

    # ── 2. 평가 대상 메소드 ─────────────────────────────────────────
    lines.append("## 2. 평가 대상 메소드")
    lines.append("")
    lines.append("| 메소드 | 설명 |")
    lines.append("|--------|------|")
    for m in METHODS:
        lines.append(f"| `{m}` | {METHOD_DESCRIPTIONS.get(m, '')} |")
    lines.append("")

    # ── 3. 평가 데이터 구성 ─────────────────────────────────────────
    lines.append("## 3. 평가 데이터 구성")
    lines.append("")

    labeled_df = labels_df[labels_df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is not None
    )]

    total_pairs = len(labels_df)
    labeled_pairs = len(labeled_df)
    source_count = labels_df["source_bill_id"].nunique()

    lines.append(f"- **전체 평가쌍 수**: {total_pairs}쌍")
    lines.append(f"- **라벨 입력 완료**: {labeled_pairs}쌍")
    lines.append(f"- **source 법안 수**: {source_count}개")
    lines.append(f"- **평가 방식**: 4개 메소드의 top-10 결과를 pooling하여 중복 제거 후 source당 최대 20개 후보 선정")
    lines.append("")

    # relevance 분포
    if labeled_pairs > 0:
        rel_values = labeled_df["human_relevance_0_to_4"].apply(lambda x: safe_float(x, 0))
        lines.append("### 관련도 점수 분포")
        lines.append("")
        lines.append("| 점수 | 건수 | 비율 |")
        lines.append("|:---:|-----:|-----:|")
        for score in range(5):
            count = (rel_values == score).sum()
            pct = count / labeled_pairs * 100 if labeled_pairs > 0 else 0
            lines.append(f"| {score} | {count} | {pct:.1f}% |")
        lines.append("")

    # ── 4. 라벨링 기준 ──────────────────────────────────────────────
    lines.append("## 4. 라벨링 기준")
    lines.append("")
    lines.append(RELEVANCE_LABELS)
    lines.append("")

    # ── 5. 메소드별 정량 평가 결과 ──────────────────────────────────
    lines.append("## 5. 메소드별 정량 평가 결과")
    lines.append("")
    lines.append("| 메소드 | P@5 | P@10 | nDCG@10 | MRR | AvgRel | Legal Meaning |")
    lines.append("|--------|----:|-----:|--------:|----:|-------:|--------------:|")

    for _, row in metrics_df.iterrows():
        m = row["method"]
        p5 = safe_float(row.get("precision_at_5"))
        p10 = safe_float(row.get("precision_at_10"))
        ndcg = safe_float(row.get("ndcg_at_10"))
        mrr_val = safe_float(row.get("mrr"))
        avg_r = safe_float(row.get("average_relevance"))
        lm = safe_float(row.get("average_legal_meaning_score"))

        p5_s = f"{p5:.4f}" if p5 is not None else "-"
        p10_s = f"{p10:.4f}" if p10 is not None else "-"
        ndcg_s = f"{ndcg:.4f}" if ndcg is not None else "-"
        mrr_s = f"{mrr_val:.4f}" if mrr_val is not None else "-"
        avg_r_s = f"{avg_r:.2f}" if avg_r is not None else "-"
        lm_s = f"{lm:.1f}" if lm is not None else "-"

        lines.append(f"| `{m}` | {p5_s} | {p10_s} | {ndcg_s} | {mrr_s} | {avg_r_s} | {lm_s} |")
    lines.append("")

    # ── 6. 가장 성능이 좋은 메소드 ──────────────────────────────────
    lines.append("## 6. 가장 성능이 좋은 메소드")
    lines.append("")
    best = find_best_method(metrics_df)
    if best:
        best_row = metrics_df[metrics_df["method"] == best].iloc[0]
        lines.append(f"종합 성능 기준 **`{best}`** 메소드가 가장 우수한 것으로 나타났다.")
        lines.append("")
        lines.append(f"- **설명**: {METHOD_DESCRIPTIONS.get(best, '')}")

        ndcg_val = safe_float(best_row.get("ndcg_at_10"))
        mrr_val = safe_float(best_row.get("mrr"))
        if ndcg_val is not None:
            lines.append(f"- **nDCG@10**: {ndcg_val:.4f}")
        if mrr_val is not None:
            lines.append(f"- **MRR**: {mrr_val:.4f}")
    else:
        lines.append("평가 데이터가 충분하지 않아 가장 성능이 좋은 메소드를 판단할 수 없다.")
    lines.append("")

    # ── 7. 지표별 해석 ──────────────────────────────────────────────
    lines.append("## 7. 지표별 해석")
    lines.append("")
    lines.append("### Precision@K")
    lines.append("상위 K개 추천 결과 중 관련도 3점 이상인 법안의 비율이다. 사용자가 상위 결과만 보는 시나리오에서의 실용적 성능을 나타낸다.")
    lines.append("")
    lines.append("### nDCG@10")
    lines.append("Normalized Discounted Cumulative Gain. 관련도 점수를 순위 가중치와 함께 반영하여 순위 품질을 종합적으로 평가한다. 가장 중요한 지표로 볼 수 있다.")
    lines.append("")
    lines.append("### MRR (Mean Reciprocal Rank)")
    lines.append("처음으로 관련 법안(관련도 ≥ 3)이 등장하는 순위의 역수 평균이다. 사용자가 첫 번째 관련 결과를 얼마나 빨리 찾을 수 있는지를 나타낸다.")
    lines.append("")
    lines.append("### Average Relevance")
    lines.append("추천 결과 전체의 평균 관련도 점수이다. 전체적인 추천 품질을 나타내는 보조 지표이다.")
    lines.append("")
    lines.append("### Legal Meaning Score")
    lines.append("법률적 의미 유사성을 세부 항목(이슈 일치, 대상 일치, 효과 일치, 범위 일치, 조문 일치)별로 평가한 복합 점수이다. 0~100점 범위이다.")
    lines.append("")

    # ── 8. 실패 사례 분석 후보 ──────────────────────────────────────
    lines.append("## 8. 실패 사례 분석 후보")
    lines.append("")

    cases = find_failure_cases(labels_df)

    # 8-1. 높은 순위인데 낮은 관련도
    lines.append("### 8.1 높은 순위(rank ≤ 3)인데 관련도가 낮은 경우(≤ 1)")
    lines.append("")
    if cases["high_rank_low_relevance"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 문제 메소드 | rank | 메소드별 정보 |")
        lines.append("|---|-----------|-----------|:-----:|-----------|:----:|------------|")
        for i, c in enumerate(cases["high_rank_low_relevance"], 1):
            info_str = format_method_info_table(c["method_info"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:20]} | {c['target_bill_name'][:20]} | "
                f"{int(c['relevance'])} | {c['problem_method']} | {int(c['problem_rank'])} | {info_str} |"
            )
        lines.append("")
        lines.append("> 분석 메모: (수동 기입 필요)")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # 8-2. 특정 메소드에서 누락된 관련 법안
    lines.append("### 8.2 특정 메소드에서 누락되었지만 관련도 높은 경우(≥ 3)")
    lines.append("")
    if cases["missed_relevant"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 포함된 메소드 | 누락된 메소드 | 최고 순위 |")
        lines.append("|---|-----------|-----------|:-----:|-------------|-------------|:-------:|")
        for i, c in enumerate(cases["missed_relevant"], 1):
            present = ", ".join(c["present_methods"])
            absent = ", ".join(c["absent_methods"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:20]} | {c['target_bill_name'][:20]} | "
                f"{int(c['relevance'])} | {present} | {absent} | {int(c['best_rank'])} |"
            )
        lines.append("")
        lines.append("> 분석 메모: (수동 기입 필요)")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # 8-3. 메소드 간 점수 차이가 큰 경우
    lines.append("### 8.3 메소드 간 점수 차이가 큰 경우(diff > 0.15)")
    lines.append("")
    if cases["method_disagree"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 점수 차이 | 메소드별 정보 |")
        lines.append("|---|-----------|-----------|:-----:|:-------:|------------|")
        for i, c in enumerate(cases["method_disagree"], 1):
            info_str = format_method_info_table(c["method_info"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:20]} | {c['target_bill_name'][:20]} | "
                f"{int(c['relevance'])} | {c['score_diff']:.4f} | {info_str} |"
            )
        lines.append("")
        lines.append("> 분석 메모: (수동 기입 필요)")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # ── 9. 향후 개선 방향 ──────────────────────────────────────────
    lines.append("## 9. 향후 개선 방향")
    lines.append("")
    lines.append("1. **라벨 데이터 확장**: 평가 source 법안을 20개에서 전체 75개로 확대하여 통계적 신뢰도를 높인다.")
    lines.append("2. **Legal Meaning Score 활용**: 세부 항목(이슈/대상/효과/범위/조문)별 가중치를 튜닝하여 법률 도메인 특화 지표를 정교하게 개선한다.")
    lines.append("3. **하이브리드 방식 탐구**: 실패 사례 분석 결과를 토대로 메소드 간 앙상블 전략을 설계한다.")
    lines.append("4. **임베딩 모델 실험**: ko-sbert 외에 legal-BERT 등 법률 도메인 특화 모델을 실험한다.")
    lines.append("5. **요약문 구조 파싱 개선**: 현행법/문제점/개정내용 구분의 정확도를 높여 structured/problem_proposal 방식의 성능을 개선한다.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*이 리포트는 `16_generate_evaluation_report.py`에 의해 자동 생성되었습니다.*")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  법률 유사도 알고리즘 평가 리포트 생성")
    print("=" * 70)
    print()

    # ── 입력 파일 로드 ──────────────────────────────────────────────
    if not os.path.exists(METRICS_CSV):
        print(f"[ERROR] 메트릭 파일을 찾을 수 없습니다: {METRICS_CSV}")
        print("먼저 15_evaluate_methods.py를 실행하세요.")
        sys.exit(1)

    metrics_df = pd.read_csv(METRICS_CSV, encoding="utf-8-sig", dtype=str)

    if os.path.exists(LABELS_EXCEL):
        print(f"[2] Excel 라벨 파일 발견: {LABELS_EXCEL} 로드 중...")
        labels_df = pd.read_excel(LABELS_EXCEL, dtype=str)
    elif os.path.exists(LABELS_CSV):
        print(f"[2] CSV 라벨 파일 발견: {LABELS_CSV} 로드 중...")
        labels_df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig", dtype=str)
    else:
        print(f"[ERROR] 라벨 파일을 찾을 수 없습니다.")
        print(f"  - {LABELS_EXCEL}")
        print(f"  - {LABELS_CSV}")
        sys.exit(1)

    labels_df = labels_df.fillna("")

    print(f"[1] 메트릭 데이터 로드 완료: {len(metrics_df)}행")
    print(f"    라벨 데이터 로드 완료: {len(labels_df)}행")
    print()

    # ── 리포트 생성 ────────────────────────────────────────────────
    report = generate_report(metrics_df, labels_df)

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[출력] 리포트 저장 완료: {OUTPUT_MD}")
    print()
    print("리포트 내용 미리보기 (첫 50줄):")
    print("-" * 70)
    for line in report.split("\n")[:50]:
        print(line)
    print("...")
    print("-" * 70)


if __name__ == "__main__":
    main()
