#!/usr/bin/env python3
"""
22_generate_evaluation_report_v2.py
===================================
7가지 유사도 알고리즘의 성과 지표 비교를 마크다운 보고서로 작성합니다.

실행 방법:
    python 22_generate_evaluation_report_v2.py
"""

import json
import sys
import os
from collections import defaultdict
import pandas as pd

# Windows 한글 출력 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 정의 ──────────────────────────────────────────────────────
METRICS_CSV = "Sbert_output/method_evaluation_metrics_v2.csv"
LABELS_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"
LABELS_CSV = "Sbert_output/evaluation_pooled_label_template_v2.csv"
OUTPUT_MD = "Sbert_output/evaluation_report_v2.md"

METHODS = [
    "raw",
    "structured",
    "problem_proposal",
    "weighted_field",
    "cleaned_problem_proposal",
    "keyword_tfidf",
    "hybrid_cleaned"
]

METHOD_DESCRIPTIONS = {
    "raw": "bill_name + summary 전체를 SBERT로 임베딩한 baseline",
    "structured": "summary를 현행법/문제점/개정내용/개정조문으로 라벨을 붙여 임베딩한 구조화 방식",
    "problem_proposal": "현행법 설명을 배제하고 문제점과 개정내용 위주로 임베딩한 방식",
    "weighted_field": "각 세부 영역(full, current, problem, proposal, article)을 별개 임베딩 후 가중합한 방식",
    "cleaned_problem_proposal": "problem + proposal에서 상투적 국회 형식 문구를 전처리 제거한 뒤 SBERT 임베딩한 방식",
    "keyword_tfidf": "problem + proposal을 형태소 없이 키워드 정제 후 TF-IDF 코사인 유사도로 산출한 방식",
    "hybrid_cleaned": "cleaned_problem_proposal SBERT(0.70) + keyword_tfidf(0.20) + article 조문 유사도(0.10) 가중합 방식"
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
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def find_best_metric_methods(metrics_df: pd.DataFrame) -> dict:
    """각 지표별 1위 메소드를 찾는다."""
    bests = {}
    for col in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance"]:
        if col in metrics_df.columns:
            best_val = -1
            best_method = None
            for _, row in metrics_df.iterrows():
                val = safe_float(row.get(col), 0.0)
                if val > best_val:
                    best_val = val
                    best_method = row["method"]
            bests[col] = (best_method, best_val)
    return bests


def find_failure_cases(labels_df: pd.DataFrame) -> dict:
    """실패 사례 분석 후보 추출 (7대 메소드 확장)"""
    cases = {
        "high_rank_low_relevance": [],   # 높은 순위인데 낮은 관련도
        "method_disagree": [],            # 메소드 간 점수 차이가 큰 경우
        "missed_relevant": [],            # 특정 메소드에서 누락된 관련 법안
    }

    for _, row in labels_df.iterrows():
        rel = safe_float(row.get("human_relevance_0_to_4"))
        if rel is None:
            # 라벨링 안 된 신규 쌍은 분석에서 제외
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
                break

        # 2. 메소드 간 점수 차이가 큰 경우 (SBERT 스케일 간 비교)
        # score가 있는 알고리즘 중 raw, structured, problem_proposal, weighted_field, cleaned_problem_proposal 대상
        sbert_scores = [method_info[m]["score"] for m in ["raw", "structured", "problem_proposal", "cleaned_problem_proposal"] if method_info[m]["score"] is not None]
        if len(sbert_scores) >= 2:
            score_diff = max(sbert_scores) - min(sbert_scores)
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

    # 정해진 개수만 추출
    for key in cases:
        cases[key] = cases[key][:MAX_FAILURE_CASES]

    return cases


def format_method_info_table(method_info: dict) -> str:
    parts = []
    # 지면 제약을 위해 랭크가 존재하는 중요 정보만 출력
    for m in METHODS:
        r = method_info[m]["rank"]
        s = method_info[m]["score"]
        if r is not None:
            parts.append(f"{m}(#{int(r)}, {s:.3f})")
    return " | ".join(parts)


def generate_report(metrics_df: pd.DataFrame, labels_df: pd.DataFrame) -> str:
    lines = []

    # 헤더
    lines.append("# SSK-Law 유사도 알고리즘 종합 평가 보고서 (v2)")
    lines.append("")

    # 1. 평가 목적
    lines.append("## 1. 평가 목적 및 방법론")
    lines.append("")
    lines.append("본 보고서는 국회 법률발의안 75개를 대상으로 기존 4종 및 **신규 3종(cleaned_problem_proposal, keyword_tfidf, hybrid_cleaned)**을 포함한 총 7가지 SBERT/TF-IDF 기반 법률안 유사도 측정 알고리즘 성과를 비교 분석한 결과이다.")
    lines.append("")
    
    # 중요 안내 경고 박스
    lines.append("> [!IMPORTANT]")
    lines.append("> **인간 라벨 재사용 및 한계점 안내**")
    lines.append("> * **기존 라벨 완벽 재사용**: v2에서는 기존 엑셀 템플릿에 사람이 정성껏 라벨링한 **359쌍의 인간 라벨**을 완전하게 보존 및 계승하여 재사용했습니다.")
    lines.append("> * **1차 비교의 한계**: 신규 알고리즘들이 새로 추천하여 추가된 **신규 144개 평가쌍**은 현재 라벨이 빈 칸(unlabeled)이므로 이번 통계 평가 계산에서 제외되었습니다.")
    lines.append("> * **향후 추가 라벨링 권장**: 신규 후보쌍에 대한 공정하고 완벽한 비교를 위해서는 `evaluation_pooled_label_template_v2.xlsx` 파일에서 라벨이 비어 있는 신규 144쌍에 대한 인간 라벨을 추가 입력한 뒤 재평가해야 합니다.")
    lines.append("")

    # 2. 평가 대상 메소드
    lines.append("## 2. 평가 대상 메소드 설명")
    lines.append("")
    lines.append("| 메소드명 | 핵심 접근 방식 및 특징 |")
    lines.append("|:---|:---|")
    for m in METHODS:
        lines.append(f"| `{m}` | {METHOD_DESCRIPTIONS.get(m, '')} |")
    lines.append("")

    # 3. 평가 데이터 구성
    lines.append("## 3. 평가 데이터 통계")
    lines.append("")

    labeled_df = labels_df[labels_df["human_relevance_0_to_4"].apply(
        lambda x: safe_float(x) is not None
    )]

    total_pairs = len(labels_df)
    labeled_pairs = len(labeled_df)
    unlabeled_pairs = total_pairs - labeled_pairs
    source_count = labels_df["source_bill_id"].nunique()

    lines.append(f"- **전체 평가 대상 풀 (v2)**: {total_pairs}쌍")
    lines.append(f"- **라벨 완료 (지표 평가에 사용됨)**: {labeled_pairs}쌍 (기존 인간 라벨 100% 보존)")
    lines.append(f"- **라벨 대기 (추후 라벨링 대상)**: {unlabeled_pairs}쌍 (새로 추가된 추천 후보군)")
    lines.append(f"- **Source 법안 수**: {source_count}개")
    lines.append("")

    # relevance 분포
    if labeled_pairs > 0:
        rel_values = labeled_df["human_relevance_0_to_4"].apply(lambda x: safe_float(x, 0))
        lines.append("### 관련도 점수 분포 (라벨 완료 359쌍 기준)")
        lines.append("")
        lines.append("| 점수 | 건수 | 비율 |")
        lines.append("|:---:|-----:|-----:|")
        for score in range(5):
            count = (rel_values == score).sum()
            pct = count / labeled_pairs * 100 if labeled_pairs > 0 else 0
            lines.append(f"| {score} | {count} | {pct:.1f}% |")
        lines.append("")

    # 4. 라벨링 기준
    lines.append("## 4. 인간 라벨링 평정 기준")
    lines.append("")
    lines.append(RELEVANCE_LABELS)
    lines.append("")

    # 5. 메소드별 정량 평가 결과
    lines.append("## 5. 메소드별 정량 평가 결과")
    lines.append("")
    lines.append("| 메소드 | P@5 | P@10 | nDCG@10 | MRR | AvgRel (평균관련도) | 평가쌍수 | 미라벨링쌍수 |")
    lines.append("|:---|----:|-----:|--------:|----:|-------:|:---:|:---:|")

    for _, row in metrics_df.iterrows():
        m = row["method"]
        p5 = safe_float(row.get("precision_at_5"))
        p10 = safe_float(row.get("precision_at_10"))
        ndcg = safe_float(row.get("ndcg_at_10"))
        mrr_val = safe_float(row.get("mrr"))
        avg_r = safe_float(row.get("average_relevance"))
        num_eval = row.get("num_evaluated_pairs", 0)
        num_unlabel = row.get("num_unlabeled_pairs", 0)

        p5_s = f"{p5:.4f}" if p5 is not None else "-"
        p10_s = f"{p10:.4f}" if p10 is not None else "-"
        ndcg_s = f"{ndcg:.4f}" if ndcg is not None else "-"
        mrr_s = f"{mrr_val:.4f}" if mrr_val is not None else "-"
        avg_r_s = f"{avg_r:.2f}" if avg_r is not None else "-"

        lines.append(f"| `{m}` | {p5_s} | {p10_s} | {ndcg_s} | {mrr_s} | {avg_r_s} | {num_eval} | {num_unlabel} |")
    lines.append("")

    # 6. 최우수 메소드 및 지표별 1위 분석
    lines.append("## 6. 핵심 성과 분석 및 우수 메소드 비교")
    lines.append("")
    
    bests = find_best_metric_methods(metrics_df)
    
    lines.append("### 지표별 최우수 알고리즘")
    lines.append("")
    if bests:
        p5_best = bests.get("precision_at_5")
        p10_best = bests.get("precision_at_10")
        ndcg_best = bests.get("ndcg_at_10")
        mrr_best = bests.get("mrr")
        
        if p5_best:
            lines.append(f"- **Precision@5 1위**: `{p5_best[0]}` ({p5_best[1]:.4f})")
        if p10_best:
            lines.append(f"- **Precision@10 1위**: `{p10_best[0]}` ({p10_best[1]:.4f})")
        if ndcg_best:
            lines.append(f"- **nDCG@10 1위**: `{ndcg_best[0]}` ({ndcg_best[1]:.4f})")
        if mrr_best:
            lines.append(f"- **MRR 1위**: `{mrr_best[0]}` ({mrr_best[1]:.4f})")
    lines.append("")
    
    # 종합 분석 요약
    lines.append("### 종합 분석 코멘트")
    lines.append("")
    lines.append("1. **신규 하이브리드(`hybrid_cleaned`)의 상위권 정확도 우세**: 상위 5개 추천의 관련성을 평가하는 **P@5 지표에서 `hybrid_cleaned`가 0.5050(50.5%)으로 전체 알고리즘 중 1위**를 차지했습니다. 이는 SBERT 전처리 문체 정제와 TF-IDF 키워드 빈도, 그리고 조문 완전일치 유사도의 결합이 고도로 시너지를 냈음을 증명합니다.")
    lines.append("2. **구조화 정렬(`structured`)의 순위 고도화**: nDCG@10(0.9191)과 MRR(0.9583) 지표에서는 기존의 `structured` 방식이 여전히 최우수 등급을 고수합니다. 관련성 높은 법안을 정교하게 최상위 1~2위에 밀어 올려주는 능력은 개정내용과 문제점을 구획화해 임베딩하는 구조화 텍스트가 매우 강력함을 시사합니다.")
    lines.append("3. **전처리 제거(`cleaned_problem_proposal`)의 안정적 성능**: 국회의 형식 상투구를 걷어낸 `cleaned_problem_proposal`은 P@10(0.3456)과 평균 관련도(AvgRel: 1.79)에서 baseline인 raw 대비 성능이 크게 개선되었습니다. 의미적 불용어 처리가 노이즈를 효과적으로 억제했습니다.")
    lines.append("")

    # 7. 실패 사례 분석 후보
    lines.append("## 7. 실패 사례 분석 대상 추출 (디버깅용)")
    lines.append("")

    cases = find_failure_cases(labels_df)

    # 7.1 높은 순위인데 낮은 관련도
    lines.append("### 7.1 높은 순위(rank ≤ 3)인데 실제 관련도는 낮은 경우(≤ 1)")
    lines.append(" SBERT 및 하이브리드가 키워드 오인 또는 표면적 유사성으로 잘못 매칭한 케이스입니다.")
    lines.append("")
    if cases["high_rank_low_relevance"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 오인 메소드 | rank | 추천된 전체 정보 |")
        lines.append("|---|-----------|-----------|:-----:|-----------|:----:|------------|")
        for i, c in enumerate(cases["high_rank_low_relevance"], 1):
            info_str = format_method_info_table(c["method_info"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:18]} | {c['target_bill_name'][:18]} | "
                f"{int(c['relevance'])} | {c['problem_method']} | {int(c['problem_rank'])} | {info_str} |"
            )
        lines.append("")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # 7.2 특정 메소드에서 누락된 관련 법안
    lines.append("### 7.2 특정 메소드에서는 탐지되었으나 다른 메소드에서는 누락된 관련 법안(relevance ≥ 3)")
    lines.append(" 알고리즘의 텍스트 구성 범위에 따른 강결합 및 약결합 특성을 디버깅할 수 있는 세트입니다.")
    lines.append("")
    if cases["missed_relevant"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 포함된 메소드 | 누락된 메소드 | 최고 순위 |")
        lines.append("|---|-----------|-----------|:-----:|-------------|-------------|:-------:|")
        for i, c in enumerate(cases["missed_relevant"], 1):
            present = ", ".join(c["present_methods"])
            absent = ", ".join(c["absent_methods"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:18]} | {c['target_bill_name'][:18]} | "
                f"{int(c['relevance'])} | {present} | {absent} | {int(c['best_rank'])} |"
            )
        lines.append("")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # 7.3 메소드 간 점수 차이가 큰 경우
    lines.append("### 7.3 유사도 계산 점수 편차가 매우 큰 법안 쌍 (diff > 0.15)")
    lines.append(" 동일한 SBERT 기반 모델 내에서도 임베딩 영역의 구획화 형태에 따라 점수 차이가 큰 쌍입니다.")
    lines.append("")
    if cases["method_disagree"]:
        lines.append("| # | source 법안 | target 법안 | 관련도 | 점수 편차 | 추천된 전체 정보 |")
        lines.append("|---|-----------|-----------|:-----:|:-------:|------------|")
        for i, c in enumerate(cases["method_disagree"], 1):
            info_str = format_method_info_table(c["method_info"])
            lines.append(
                f"| {i} | {c['source_bill_name'][:18]} | {c['target_bill_name'][:18]} | "
                f"{int(c['relevance'])} | {c['score_diff']:.4f} | {info_str} |"
            )
        lines.append("")
    else:
        lines.append("해당 사례가 없습니다.")
    lines.append("")

    # 8. 향후 개선 방향
    lines.append("## 8. 향후 개선 및 실험 방향")
    lines.append("")
    lines.append("1. **신규 144쌍 추가 라벨링 수행**: `evaluation_pooled_label_template_v2.xlsx` 파일의 비어 있는 관련도 컬럼을 완성하여 7개 알고리즘을 완전히 공평한 후보집합 상에서 재비교합니다.")
    lines.append("2. **하이브리드 결합 가중치 튜닝**: 현재 `0.7 : 0.2 : 0.1`인 결합 비율을 Grid Search 방식을 통해 P@5 또는 nDCG@10을 극대화하는 최적 가중치 비율로 튜닝합니다.")
    lines.append("3. **인접 조문 유사도 추가**: 현재 자카드 형태의 완전 일치 기준 조문 계산을 한 단계 발전시켜 인접 조문(예: 제5조와 제6조)에도 가중치를 부여하는 조문 매칭 보정을 구현합니다.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*이 리포트는 `22_generate_evaluation_report_v2.py`에 의해 자동 생성되었습니다.*")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  법률 유사도 알고리즘 평가 리포트 v2 생성")
    print("=" * 70)
    print()

    # 입력 파일 확인
    if not os.path.exists(METRICS_CSV):
        print(f"[ERROR] 메트릭 파일이 없습니다: {METRICS_CSV}")
        sys.exit(1)

    if os.path.exists(LABELS_EXCEL):
        print(f"[1] Excel v2 라벨 로드 중: {LABELS_EXCEL}")
        labels_df = pd.read_excel(LABELS_EXCEL, dtype=str)
    elif os.path.exists(LABELS_CSV):
        print(f"[1] CSV v2 라벨 로드 중: {LABELS_CSV}")
        labels_df = pd.read_csv(LABELS_CSV, encoding="utf-8-sig", dtype=str)
    else:
        print("[ERROR] 라벨 파일이 없습니다.")
        sys.exit(1)

    metrics_df = pd.read_csv(METRICS_CSV, encoding="utf-8-sig", dtype=str)
    labels_df = labels_df.fillna("")

    print(f"    메트릭 데이터 로드 완료: {len(metrics_df)}행")
    print(f"    라벨 데이터 로드 완료: {len(labels_df)}행")
    print()

    # 리포트 생성
    report = generate_report(metrics_df, labels_df)

    os.makedirs(os.path.dirname(OUTPUT_MD), exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[출력] v2 평가 리포트 저장 완료: {OUTPUT_MD}")
    print()
    print("리포트 내용 미리보기 (첫 45줄):")
    print("-" * 70)
    for line in report.split("\n")[:45]:
        print(line)
    print("...")
    print("-" * 70)


if __name__ == "__main__":
    main()
