#!/usr/bin/env python3
"""
23_build_full_source_pooled_evaluation.py
=========================================
75개 전체 법안을 source로 사용하여 7개 메소드의 top-10 추천 결과를
풀링한 평가 템플릿 xlsx 파일을 생성합니다.

기존 v2 템플릿(20개 source)과 동일한 컬럼 구조를 사용하며,
기존에 라벨링된 쌍의 인간 라벨은 그대로 보존합니다.

실행 방법:
    python 23_build_full_source_pooled_evaluation.py
"""

import os
import json
import sys
import pandas as pd
from collections import defaultdict
from openpyxl.styles import Alignment, PatternFill, Font

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 경로 상수 ──────────────────────────────────────────────────────
DATASET_JSON = "test_dataset/full_dataset.json"

TOPK_FILES = {
    "raw": "Sbert_output/topk_raw.json",
    "structured": "Sbert_output/topk_structured.json",
    "problem_proposal": "Sbert_output/topk_problem_proposal.json",
    "weighted_field": "Sbert_output/topk_weighted_field_similarity.json",
    "cleaned_problem_proposal": "Sbert_output/topk_cleaned_problem_proposal.json",
    "keyword_tfidf": "Sbert_output/topk_keyword_tfidf.json",
    "hybrid_cleaned": "Sbert_output/topk_hybrid_cleaned.json"
}

# 기존 라벨 보존용
EXISTING_LABEL_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"
EXISTING_LABEL_CSV = "Sbert_output/evaluation_pooled_label_template_v2.csv"

OUTPUT_XLSX = "Sbert_output/evaluation_full_source_pooled.xlsx"
OUTPUT_CSV = "Sbert_output/evaluation_full_source_pooled.csv"
OUTPUT_JSON = "Sbert_output/evaluation_full_source_pooled.json"

ALL_METHODS = [
    "raw", "structured", "problem_proposal", "weighted_field",
    "cleaned_problem_proposal", "keyword_tfidf", "hybrid_cleaned"
]

PRESERVE_COLS = [
    "human_relevance_0_to_4",
    "human_issue_match_0_to_2",
    "human_target_match_0_to_2",
    "human_effect_match_0_to_2",
    "human_scope_match_0_to_2",
    "human_article_match_0_to_2",
    "notes"
]


def format_categories(cats):
    """카테고리 리스트를 문자열로 포맷"""
    if isinstance(cats, list):
        return ";".join([c.strip() for c in cats if c and c.strip()])
    return str(cats) if cats else ""


def main():
    print("=" * 80)
    print("  75개 전체 source 법안 기반 풀링 평가 템플릿 생성")
    print("=" * 80)

    # 1. full_dataset.json 로드
    if not os.path.exists(DATASET_JSON):
        print(f"[ERROR] 데이터셋 파일이 없습니다: {DATASET_JSON}")
        sys.exit(1)
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    bills_db = {b["bill_id"]: b for b in dataset.get("bills", [])}
    print(f"[1] 전체 법안 수: {len(bills_db)}개")

    # 2. 기존 라벨 데이터 로드 (보존용)
    existing_labels = {}
    if os.path.exists(EXISTING_LABEL_EXCEL):
        print(f"[2] 기존 라벨 Excel 로드 중: {EXISTING_LABEL_EXCEL}")
        df_existing = pd.read_excel(EXISTING_LABEL_EXCEL, dtype=str).fillna("")
    elif os.path.exists(EXISTING_LABEL_CSV):
        print(f"[2] 기존 라벨 CSV 로드 중: {EXISTING_LABEL_CSV}")
        df_existing = pd.read_csv(EXISTING_LABEL_CSV, encoding="utf-8-sig", dtype=str).fillna("")
    else:
        print("[2] 기존 라벨 파일 없음 - 새로 생성합니다.")
        df_existing = pd.DataFrame()

    if len(df_existing) > 0:
        for _, row in df_existing.iterrows():
            key = (row["source_bill_id"], row["target_bill_id"])
            label_dict = {col: row.get(col, "") for col in PRESERVE_COLS}
            existing_labels[key] = label_dict
        print(f"    기존 라벨 보존 대상: {len(existing_labels)}쌍")

    # 3. 7개 topk JSON 파일 로드
    # 구조: pair_data[(source_id, target_id)][method] = (rank, score)
    pair_data = defaultdict(lambda: defaultdict(dict))

    for method, filepath in TOPK_FILES.items():
        if not os.path.exists(filepath):
            print(f"[WARNING] {method} 결과 파일이 없습니다: {filepath}")
            continue

        with open(filepath, "r", encoding="utf-8") as f:
            pairs_list = json.load(f)

        for item in pairs_list:
            s_id = item["source_bill_id"]
            t_id = item["target_bill_id"]
            rank = item["rank"]
            score = item.get("similarity", item.get("final_similarity", 0.0))
            pair_data[(s_id, t_id)][method] = (rank, score)

    print(f"[3] 전체 고유 (source, target) 쌍 수: {len(pair_data)}개")

    # 4. 데이터프레임 생성
    rows = []
    for (s_id, t_id), methods_scores in pair_data.items():
        source_bill = bills_db.get(s_id, {})
        target_bill = bills_db.get(t_id, {})

        row = {
            "source_bill_id": s_id,
            "source_bill_name": source_bill.get("bill_name", ""),
            "target_bill_id": t_id,
            "target_bill_name": target_bill.get("bill_name", ""),
            "source_summary": source_bill.get("summary", ""),
            "target_summary": target_bill.get("summary", ""),
            "source_manual_categories": format_categories(source_bill.get("categories", [])),
            "target_manual_categories": format_categories(target_bill.get("categories", []))
        }

        # 메소드별 rank/score
        appeared = []
        ranks = []
        scores = []
        for m in ALL_METHODS:
            if m in methods_scores:
                r, s = methods_scores[m]
                row[f"{m}_rank"] = str(r)
                row[f"{m}_score"] = str(s)
                appeared.append(m)
                ranks.append(int(r))
                scores.append(float(s))
            else:
                row[f"{m}_rank"] = ""
                row[f"{m}_score"] = ""

        row["appeared_methods"] = ";".join(appeared)
        row["method_support_count"] = str(len(appeared))
        row["best_rank"] = str(min(ranks)) if ranks else ""
        row["best_score"] = str(max(scores)) if scores else ""

        # 기존 라벨 보존
        label = existing_labels.get((s_id, t_id), {})
        for col in PRESERVE_COLS:
            row[col] = label.get(col, "")

        rows.append(row)

    # 컬럼 순서 정의 (기존 v2와 동일)
    col_order = [
        "source_bill_id", "source_bill_name", "target_bill_id", "target_bill_name",
        "source_summary", "target_summary", "source_manual_categories", "target_manual_categories",
        "appeared_methods", "method_support_count", "best_rank", "best_score",
        "raw_rank", "raw_score", "structured_rank", "structured_score",
        "problem_proposal_rank", "problem_proposal_score", "weighted_field_rank", "weighted_field_score",
        "cleaned_problem_proposal_rank", "cleaned_problem_proposal_score",
        "keyword_tfidf_rank", "keyword_tfidf_score", "hybrid_cleaned_rank", "hybrid_cleaned_score",
        "human_relevance_0_to_4", "human_issue_match_0_to_2", "human_target_match_0_to_2",
        "human_effect_match_0_to_2", "human_scope_match_0_to_2", "human_article_match_0_to_2", "notes"
    ]

    df_final = pd.DataFrame(rows, columns=col_order)
    df_final = df_final.sort_values(by=["source_bill_id", "best_rank"])
    df_final = df_final.fillna("")

    # 5. CSV 저장
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[출력] CSV 저장 완료: {OUTPUT_CSV}")

    # 6. JSON 저장
    final_dict = {
        "description": "75개 전체 source 법안 기반 7개 메소드 풀링 평가 템플릿",
        "total_pairs": len(df_final),
        "source_count": df_final["source_bill_id"].nunique(),
        "methods": ALL_METHODS,
        "pairs": df_final.to_dict(orient="records")
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)
    print(f"[출력] JSON 저장 완료: {OUTPUT_JSON}")

    # 7. Excel 저장 (가독성 서식 적용)
    writer = pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl")
    df_final.to_excel(writer, index=False, sheet_name="FullSourceEvaluation")

    workbook = writer.book
    worksheet = writer.sheets["FullSourceEvaluation"]

    # 헤더 스타일
    header_fill = PatternFill(start_color="E6EDF5", end_color="E6EDF5", fill_type="solid")
    header_font = Font(bold=True, size=11)

    for col in worksheet.columns:
        col_letter = col[0].column_letter
        col_name = df_final.columns[col[0].column - 1]

        # 헤더 셀 스타일
        header_cell = col[0]
        header_cell.fill = header_fill
        header_cell.font = header_font
        header_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

        # 데이터 기본 정렬
        align = Alignment(vertical="top", horizontal="left", wrap_text=True)

        if col_name in ["source_summary", "target_summary"]:
            worksheet.column_dimensions[col_letter].width = 65
        elif "id" in col_name:
            worksheet.column_dimensions[col_letter].width = 15
            align = Alignment(vertical="top", horizontal="center")
        elif "name" in col_name:
            worksheet.column_dimensions[col_letter].width = 30
        elif "score" in col_name or "rank" in col_name:
            worksheet.column_dimensions[col_letter].width = 12
            align = Alignment(vertical="top", horizontal="right")
        elif "human_" in col_name:
            worksheet.column_dimensions[col_letter].width = 18
            align = Alignment(vertical="top", horizontal="center")
        else:
            worksheet.column_dimensions[col_letter].width = 15

        for cell in col[1:]:
            cell.alignment = align

    writer.close()
    print(f"[출력] Excel 저장 완료 (서식 적용됨): {OUTPUT_XLSX}")

    # 통계 출력
    labeled = df_final[df_final["human_relevance_0_to_4"] != ""]
    unlabeled = df_final[df_final["human_relevance_0_to_4"] == ""]

    print()
    print("=" * 70)
    print("  생성 요약 통계")
    print("=" * 70)
    print(f"  전체 source 법안 수    : {df_final['source_bill_id'].nunique()}개")
    print(f"  전체 고유 평가쌍 수     : {len(df_final)}개")
    print(f"  기존 라벨 보존 row 수   : {len(labeled)}개")
    print(f"  라벨 미지정 row 수      : {len(unlabeled)}개")
    print("=" * 70)


if __name__ == '__main__':
    main()
