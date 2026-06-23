#!/usr/bin/env python3
"""
20_update_pooled_evaluation_with_new_methods.py
================================================
기존에 평가 완료된 라벨 데이터(xlsx/csv)를 완벽히 유지 및 보존하면서,
신규 알고리즘 3종(cleaned_problem_proposal, keyword_tfidf, hybrid_cleaned)의
추천 결과를 기존 평가 풀에 병합하여 v2 데이터셋을 만듭니다.

실행 방법:
    python 20_update_pooled_evaluation_with_new_methods.py
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

# ── 상수 및 경로 정의 ──────────────────────────────────────────────────
ORIGINAL_EXCEL = "Sbert_output/evaluation_pooled_label_template.xlsx"
ORIGINAL_CSV = "Sbert_output/evaluation_pooled_label_template.csv"
DATASET_JSON = "test_dataset/full_dataset.json"

NEW_METHODS_FILES = {
    "cleaned_problem_proposal": "Sbert_output/topk_cleaned_problem_proposal.json",
    "keyword_tfidf": "Sbert_output/topk_keyword_tfidf.json",
    "hybrid_cleaned": "Sbert_output/topk_hybrid_cleaned.json"
}

OUTPUT_CSV = "Sbert_output/evaluation_pooled_label_template_v2.csv"
OUTPUT_JSON = "Sbert_output/evaluation_pooled_label_template_v2.json"
OUTPUT_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"

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

def load_original_labels():
    """기존 라벨 데이터를 Excel 또는 CSV에서 읽어 DataFrame으로 반환"""
    if os.path.exists(ORIGINAL_EXCEL):
        print(f"[1] 기존 Excel 라벨 파일 로드 중: {ORIGINAL_EXCEL}")
        return pd.read_excel(ORIGINAL_EXCEL, dtype=str)
    elif os.path.exists(ORIGINAL_CSV):
        print(f"[1] 기존 CSV 라벨 파일 로드 중: {ORIGINAL_CSV}")
        return pd.read_csv(ORIGINAL_CSV, encoding="utf-8-sig", dtype=str)
    else:
        print(f"[ERROR] 기존 라벨 템플릿 파일을 찾을 수 없습니다.")
        sys.exit(1)

def main():
    print("=" * 80)
    print("  신규 메소드 병합 및 라벨링 템플릿 v2 업데이트")
    print("=" * 80)
    
    # 1. 기존 라벨 데이터 로드
    df_orig = load_original_labels()
    df_orig = df_orig.fillna("")
    
    # 기존 풀에 있는 쌍을 인덱싱
    # Key: (source_bill_id, target_bill_id)
    orig_pairs = {}
    source_ids = set()
    for _, row in df_orig.iterrows():
        s_id = row["source_bill_id"]
        t_id = row["target_bill_id"]
        source_ids.add(s_id)
        
        # 보존할 인간 라벨 딕셔너리 구성
        label_dict = {col: row.get(col, "") for col in PRESERVE_COLS}
        # 기존 메소드별 랭크/스코어 정보도 일단 수집
        old_info = {}
        for m in ["raw", "structured", "problem_proposal", "weighted_field"]:
            old_info[f"{m}_rank"] = row.get(f"{m}_rank", "")
            old_info[f"{m}_score"] = row.get(f"{m}_score", "")
            
        orig_pairs[(s_id, t_id)] = {
            "labels": label_dict,
            "old_info": old_info,
            "meta": {
                "source_bill_name": row.get("source_bill_name", ""),
                "target_bill_name": row.get("target_bill_name", ""),
                "source_summary": row.get("source_summary", ""),
                "target_summary": row.get("target_summary", ""),
                "source_manual_categories": row.get("source_manual_categories", ""),
                "target_manual_categories": row.get("target_manual_categories", "")
            }
        }
        
    print(f"  - 기존 풀 데이터 수: {len(df_orig)}쌍")
    print(f"  - 평가 대상 source 법안 수: {len(source_ids)}개")
    
    # 2. full_dataset.json 로드 (신규 추가되는 쌍의 메타데이터 채우기용)
    if not os.path.exists(DATASET_JSON):
        print(f"[ERROR] 데이터셋 파일이 없습니다: {DATASET_JSON}")
        sys.exit(1)
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)
    bills_db = {b["bill_id"]: b for b in dataset_data.get("bills", [])}
    
    # 3. 신규 3개 알고리즘 추천 결과 로드
    new_results = defaultdict(lambda: defaultdict(dict))
    # 구조: new_results[(source_id, target_id)][method] = (rank, score)
    
    for method, filepath in NEW_METHODS_FILES.items():
        if not os.path.exists(filepath):
            print(f"[ERROR] 신규 메소드 결과 파일이 없습니다: {filepath}")
            print("먼저 이전 실행 스크립트들을 돌려주세요.")
            sys.exit(1)
            
        with open(filepath, "r", encoding="utf-8") as f:
            pairs_list = json.load(f)
            
        for item in pairs_list:
            s_id = item["source_bill_id"]
            t_id = item["target_bill_id"]
            
            # 오직 기존 풀에 존재하는 20개 source만 필터링 대상으로 삼음
            if s_id not in source_ids:
                continue
                
            rank = item["rank"]
            score = item["similarity"]
            new_results[(s_id, t_id)][method] = (rank, score)
            
    # 4. 병합 분석 및 신규 후보군 도출
    # 기존 쌍 중에서 신규 정보 업데이트 및 아예 새로운 추천 쌍 추출
    updated_pairs = {}
    new_candidate_pairs = defaultdict(list) # source_id별 신규 후보들
    
    # (A) 기존 쌍 업데이트
    for (s_id, t_id), data in orig_pairs.items():
        updated_data = {
            "source_bill_id": s_id,
            "target_bill_id": t_id,
            "source_bill_name": data["meta"]["source_bill_name"],
            "target_bill_name": data["meta"]["target_bill_name"],
            "source_summary": data["meta"]["source_summary"],
            "target_summary": data["meta"]["target_summary"],
            "source_manual_categories": data["meta"]["source_manual_categories"],
            "target_manual_categories": data["meta"]["target_manual_categories"]
        }
        # 기존 라벨 적용
        for col in PRESERVE_COLS:
            updated_data[col] = data["labels"][col]
            
        # 기존 알고리즘 점수 적용
        for k, v in data["old_info"].items():
            updated_data[k] = v
            
        # 신규 알고리즘 점수 적용 (있을 시)
        for m in ["cleaned_problem_proposal", "keyword_tfidf", "hybrid_cleaned"]:
            if m in new_results[(s_id, t_id)]:
                rank, score = new_results[(s_id, t_id)][m]
                updated_data[f"{m}_rank"] = str(rank)
                updated_data[f"{m}_score"] = str(score)
            else:
                updated_data[f"{m}_rank"] = ""
                updated_data[f"{m}_score"] = ""
                
        updated_pairs[(s_id, t_id)] = updated_data
        
    # (B) 신규 쌍 필터링 및 정렬용 가중치 산출
    for (s_id, t_id), methods_scores in new_results.items():
        # 기존 풀에 이미 있으면 패스
        if (s_id, t_id) in orig_pairs:
            continue
            
        # 신규 추가할 후보 데이터 구성
        source_bill = bills_db[s_id]
        target_bill = bills_db[t_id]
        
        # categories를 참고용으로 포맷
        from text_builders import format_categories
        s_cats = format_categories(source_bill.get("categories", []))
        t_cats = format_categories(target_bill.get("categories", []))
        
        candidate_data = {
            "source_bill_id": s_id,
            "target_bill_id": t_id,
            "source_bill_name": source_bill.get("bill_name", ""),
            "target_bill_name": target_bill.get("bill_name", ""),
            "source_summary": source_bill.get("summary", ""),
            "target_summary": target_bill.get("summary", ""),
            "source_manual_categories": s_cats,
            "target_manual_categories": t_cats
        }
        
        # 기존 알고리즘은 미등장이므로 빈값
        for m in ["raw", "structured", "problem_proposal", "weighted_field"]:
            candidate_data[f"{m}_rank"] = ""
            candidate_data[f"{m}_score"] = ""
            
        # 신규 알고리즘 정보
        support_count = 0
        hybrid_rank = 999
        for m in ["cleaned_problem_proposal", "keyword_tfidf", "hybrid_cleaned"]:
            if m in methods_scores:
                rank, score = methods_scores[m]
                candidate_data[f"{m}_rank"] = str(rank)
                candidate_data[f"{m}_score"] = str(score)
                support_count += 1
                if m == "hybrid_cleaned":
                    hybrid_rank = rank
            else:
                candidate_data[f"{m}_rank"] = ""
                candidate_data[f"{m}_score"] = ""
                
        # 인간 라벨은 무조건 공백
        for col in PRESERVE_COLS:
            candidate_data[col] = ""
            
        new_candidate_pairs[s_id].append({
            "data": candidate_data,
            "support_count": support_count,
            "hybrid_rank": hybrid_rank,
            "target_id": t_id
        })
        
    # 5. 각 source_id 별 신규 후보 추가 (최대 10개 제한)
    newly_added_count = 0
    for s_id, candidates in new_candidate_pairs.items():
        # 정렬 규칙: (1) support_count 내림차순, (2) hybrid_rank 오름차순, (3) target_id 오름차순
        sorted_candidates = sorted(
            candidates,
            key=lambda x: (-x["support_count"], x["hybrid_rank"], x["target_id"])
        )
        
        # 최대 10개 채택
        selected = sorted_candidates[:10]
        for item in selected:
            c_data = item["data"]
            t_id = c_data["target_bill_id"]
            updated_pairs[(s_id, t_id)] = c_data
            newly_added_count += 1
            
    # 6. 메타데이터 갱신 (appeared_methods, method_support_count, best_rank, best_score)
    final_rows = []
    for (s_id, t_id), row in updated_pairs.items():
        appeared = []
        ranks = []
        scores = []
        
        for m in ALL_METHODS:
            r_val = row.get(f"{m}_rank", "")
            s_val = row.get(f"{m}_score", "")
            if r_val != "":
                appeared.append(m)
                try:
                    ranks.append(int(r_val))
                except ValueError:
                    pass
                try:
                    scores.append(float(s_val))
                except ValueError:
                    pass
                    
        row["appeared_methods"] = ";".join(appeared)
        row["method_support_count"] = str(len(appeared))
        row["best_rank"] = str(min(ranks)) if ranks else ""
        row["best_score"] = str(max(scores)) if scores else ""
        
        final_rows.append(row)
        
    # 데이터프레임 빌드 및 정렬 (source_bill_id, 그리고 정형 순서로 정렬)
    # 컬럼 레이아웃 순서 정의
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
    
    df_final = pd.DataFrame(final_rows, columns=col_order)
    df_final = df_final.sort_values(by=["source_bill_id", "best_rank"])
    df_final = df_final.fillna("")
    
    # 7. 저장
    # CSV 저장
    df_final.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[출력] CSV v2 저장 완료: {OUTPUT_CSV}")
    
    # JSON 저장
    final_dict = {
        "description": "법률 유사도 알고리즘 평가용 라벨링 템플릿 v2 (기존 라벨 보존 및 신규 알고리즘 추가)",
        "total_pairs": len(df_final),
        "source_count": len(source_ids),
        "methods": ALL_METHODS,
        "pairs": df_final.to_dict(orient="records")
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)
    print(f"[출력] JSON v2 저장 완료: {OUTPUT_JSON}")
    
    # Excel 저장 및 스타일 적용 (가독성 서식 유지)
    writer = pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl")
    df_final.to_excel(writer, index=False, sheet_name="EvaluationSet")
    
    workbook = writer.book
    worksheet = writer.sheets["EvaluationSet"]
    
    # 헤더 스타일
    header_fill = PatternFill(start_color="E6EDF5", end_color="E6EDF5", fill_type="solid")
    header_font = Font(bold=True, size=11)
    
    for col in worksheet.columns:
        col_letter = col[0].column_letter
        col_name = df_final.columns[col[0].column - 1]
        
        # 헤더 셀 스타일 지정
        header_cell = col[0]
        header_cell.fill = header_fill
        header_cell.font = header_font
        header_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        # 데이터 기본 정렬
        align = Alignment(vertical="top", horizontal="left", wrap_text=True)
        
        # 너비 및 정렬 세부 정의
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
    print(f"[출력] Excel v2 저장 완료 (서식 적용됨): {OUTPUT_EXCEL}")
    
    # 통계 출력
    labeled_rows = df_final[df_final["human_relevance_0_to_4"] != ""]
    unlabeled_rows = df_final[df_final["human_relevance_0_to_4"] == ""]
    
    print()
    print("=" * 70)
    print("  병합 요약 통계")
    print("=" * 70)
    print(f"  기존 평가 row 수       : {len(df_orig)}개")
    print(f"  기존 라벨 완료 row 수   : {len(labeled_rows)}개")
    print(f"  새로 추가된 row 수     : {newly_added_count}개")
    print(f"  최종 v2 row 수         : {len(df_final)}개")
    print(f"  라벨 보존 row 수       : {len(labeled_rows)}개")
    print(f"  라벨 미지정 신규 row 수 : {len(unlabeled_rows)}개")
    print("=" * 70)

if __name__ == '__main__':
    main()
