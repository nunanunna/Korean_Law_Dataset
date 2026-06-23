#!/usr/bin/env python3
"""
14_build_pooled_evaluation_set.py
=================================
각 메소드(raw, structured, problem_proposal, weighted_field)의 top-k 결과를
합쳐서 사람이 평가할 라벨링용 CSV/JSON을 만든다.

실행 방법:
    python 14_build_pooled_evaluation_set.py

출력 파일:
    Sbert_output/evaluation_pooled_label_template.csv
    Sbert_output/evaluation_pooled_label_template.json
"""

import json
import sys
import os
from collections import defaultdict

import pandas as pd
from openpyxl.styles import Alignment, PatternFill, Font

# ── Windows 한글 출력 설정 ───────────────────────────────────────────
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 정의 ──────────────────────────────────────────────────────
FULL_DATASET_PATH = "test_dataset/full_dataset.json"
SOURCE_BILLS_PATH = "Sbert_output/selected_source_bills.json"

METHOD_FILES = {
    "raw": "Sbert_output/topk_raw.json",
    "structured": "Sbert_output/topk_structured.json",
    "problem_proposal": "Sbert_output/topk_problem_proposal.json",
    "weighted_field": "Sbert_output/topk_weighted_field_similarity.json",
}

OUTPUT_CSV = "Sbert_output/evaluation_pooled_label_template.csv"
OUTPUT_EXCEL = "Sbert_output/evaluation_pooled_label_template.xlsx"
OUTPUT_JSON = "Sbert_output/evaluation_pooled_label_template.json"

MAX_CANDIDATES_PER_SOURCE = 20

METHODS = ["raw", "structured", "problem_proposal", "weighted_field"]

# source/target ID 후보 필드명
SOURCE_ID_KEYS = ["source_bill_id", "bill_id_1", "source_id"]
TARGET_ID_KEYS = ["target_bill_id", "bill_id_2", "target_id"]
SOURCE_NAME_KEYS = ["source_bill_name", "bill_name_1"]
TARGET_NAME_KEYS = ["target_bill_name", "bill_name_2"]
SCORE_KEYS = ["similarity", "final_similarity", "score", "sbert_similarity"]


def load_json(path: str) -> dict | list:
    """JSON 파일을 로드한다."""
    if not os.path.exists(path):
        print(f"[ERROR] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_field(record: dict, candidates: list[str], default=None):
    """후보 필드명 중 존재하는 첫 번째 값을 반환한다."""
    for key in candidates:
        if key in record and record[key] is not None:
            return record[key]
    return default


def load_full_dataset(path: str) -> dict:
    """전체 법안 데이터를 로드하여 bill_id → bill_info 매핑을 만든다."""
    data = load_json(path)
    bills = data.get("bills", data if isinstance(data, list) else [])
    bill_map = {}
    for bill in bills:
        bid = bill.get("bill_id", "")
        bill_map[bid] = {
            "bill_id": bid,
            "bill_name": bill.get("bill_name", ""),
            "summary": bill.get("summary", ""),
            "categories": bill.get("categories", []),
        }
    return bill_map


def load_source_bills(path: str) -> list[dict]:
    """source 법안 목록을 로드한다."""
    if not os.path.exists(path):
        print(f"[ERROR] source 법안 파일을 찾을 수 없습니다: {path}")
        print("먼저 13_select_source_bills.py를 실행하세요.")
        sys.exit(1)

    data = load_json(path)

    # 가능한 구조 유연하게 처리
    if isinstance(data, list):
        source_list = data
    elif isinstance(data, dict):
        source_list = data.get("selected_source_bills", data.get("bills", []))
    else:
        print(f"[ERROR] source 법안 파일 구조를 인식할 수 없습니다.")
        sys.exit(1)

    return source_list


def load_topk_results(method: str, path: str) -> list[dict]:
    """특정 메소드의 top-k 결과를 정규화하여 로드한다."""
    if not os.path.exists(path):
        print(f"[WARNING] {method} 결과 파일을 찾을 수 없습니다: {path}")
        return []

    data = load_json(path)
    if not isinstance(data, list):
        data = data.get("results", data.get("top_k", []))

    normalized = []
    for rec in data:
        source_id = get_field(rec, SOURCE_ID_KEYS, "")
        target_id = get_field(rec, TARGET_ID_KEYS, "")
        source_name = get_field(rec, SOURCE_NAME_KEYS, "")
        target_name = get_field(rec, TARGET_NAME_KEYS, "")
        rank = rec.get("rank", 0)
        score = get_field(rec, SCORE_KEYS, 0.0)

        if source_id and target_id:
            normalized.append({
                "method": method,
                "source_bill_id": source_id,
                "source_bill_name": source_name,
                "target_bill_id": target_id,
                "target_bill_name": target_name,
                "rank": rank,
                "score": round(float(score), 6) if score else 0.0,
            })

    return normalized


def main():
    print("=" * 70)
    print("  법률 유사도 평가용 Pooled Evaluation Set 생성")
    print("=" * 70)
    print()

    # ── A. 전체 법안 데이터 로드 ───────────────────────────────────
    bill_map = load_full_dataset(FULL_DATASET_PATH)
    print(f"[1] 전체 법안 수: {len(bill_map)}개")

    # ── B. source 법안 로드 ────────────────────────────────────────
    source_bills = load_source_bills(SOURCE_BILLS_PATH)
    source_ids = set()
    source_info = {}
    for sb in source_bills:
        bid = sb.get("bill_id", "")
        source_ids.add(bid)
        source_info[bid] = {
            "bill_name": sb.get("bill_name", ""),
            "manual_categories": sb.get("manual_categories", sb.get("categories", [])),
        }
    print(f"[2] source 법안 수: {len(source_ids)}개")
    print()

    # ── C. top-k 결과 파일 로드 ────────────────────────────────────
    all_results = {}
    for method, path in METHOD_FILES.items():
        results = load_topk_results(method, path)
        all_results[method] = results
        print(f"[3] {method:20s} → {len(results):>5d}개 추천 결과 로드")
    print()

    # ── D. Pooling ──────────────────────────────────────────────────
    # source-target 쌍별로 메소드 정보를 모은다
    # key: (source_id, target_id)
    pool = defaultdict(lambda: {
        "source_bill_id": "",
        "target_bill_id": "",
        "source_bill_name": "",
        "target_bill_name": "",
        "methods": {},
    })

    for method, results in all_results.items():
        for rec in results:
            sid = rec["source_bill_id"]
            tid = rec["target_bill_id"]

            # source 법안 20개에 대해서만 pooling
            if sid not in source_ids:
                continue

            key = (sid, tid)
            entry = pool[key]
            entry["source_bill_id"] = sid
            entry["target_bill_id"] = tid
            entry["source_bill_name"] = rec.get("source_bill_name", "")
            entry["target_bill_name"] = rec.get("target_bill_name", "")
            entry["methods"][method] = {
                "rank": rec["rank"],
                "score": rec["score"],
            }

    print(f"[4] Pooling 후 전체 고유 평가쌍 수: {len(pool)}개")

    # ── E. source당 최대 20개만 남기기 ─────────────────────────────
    # 후보별 정렬 메타 계산
    source_groups = defaultdict(list)
    for (sid, tid), entry in pool.items():
        methods_info = entry["methods"]
        method_support_count = len(methods_info)
        best_rank = min(m["rank"] for m in methods_info.values())
        best_score = max(m["score"] for m in methods_info.values())
        appeared_methods = ";".join(sorted(methods_info.keys()))

        candidate = {
            **entry,
            "method_support_count": method_support_count,
            "best_rank": best_rank,
            "best_score": best_score,
            "appeared_methods": appeared_methods,
        }
        source_groups[sid].append(candidate)

    # 정렬: (1) support_count DESC, (2) best_rank ASC, (3) best_score DESC, (4) target_id ASC
    final_pairs = []
    per_source_stats = []

    for sid in sorted(source_groups.keys()):
        candidates = source_groups[sid]
        candidates.sort(key=lambda c: (
            -c["method_support_count"],
            c["best_rank"],
            -c["best_score"],
            c["target_bill_id"],
        ))

        selected = candidates[:MAX_CANDIDATES_PER_SOURCE]
        final_pairs.extend(selected)
        per_source_stats.append(len(selected))

    print(f"[5] 최종 평가쌍 수 (source당 최대 {MAX_CANDIDATES_PER_SOURCE}개): {len(final_pairs)}개")
    print()

    # ── F. 라벨링 CSV 생성 ─────────────────────────────────────────
    rows = []
    for pair in final_pairs:
        sid = pair["source_bill_id"]
        tid = pair["target_bill_id"]

        # source 정보
        s_info = bill_map.get(sid, {})
        s_name = pair.get("source_bill_name", "") or s_info.get("bill_name", "")
        s_summary = s_info.get("summary", "")
        s_categories = source_info.get(sid, {}).get(
            "manual_categories", s_info.get("categories", [])
        )

        # target 정보
        t_info = bill_map.get(tid, {})
        t_name = pair.get("target_bill_name", "") or t_info.get("bill_name", "")
        t_summary = t_info.get("summary", "")
        t_categories = t_info.get("categories", [])

        # 메소드별 rank/score
        methods_data = pair.get("methods", {})

        row = {
            "source_bill_id": sid,
            "source_bill_name": s_name,
            "target_bill_id": tid,
            "target_bill_name": t_name,
            "source_summary": s_summary,
            "target_summary": t_summary,
            "source_manual_categories": ";".join(s_categories) if isinstance(s_categories, list) else str(s_categories),
            "target_manual_categories": ";".join(t_categories) if isinstance(t_categories, list) else str(t_categories),
            "appeared_methods": pair.get("appeared_methods", ""),
            "method_support_count": pair.get("method_support_count", 0),
            "best_rank": pair.get("best_rank", ""),
            "best_score": pair.get("best_score", ""),
        }

        # 각 메소드별 rank/score
        for m in METHODS:
            m_info = methods_data.get(m, {})
            row[f"{m}_rank"] = m_info.get("rank", "")
            row[f"{m}_score"] = m_info.get("score", "")

        # 라벨링 컬럼 (빈 문자열)
        row["human_relevance_0_to_4"] = ""
        row["human_issue_match_0_to_2"] = ""
        row["human_target_match_0_to_2"] = ""
        row["human_effect_match_0_to_2"] = ""
        row["human_scope_match_0_to_2"] = ""
        row["human_article_match_0_to_2"] = ""
        row["notes"] = ""

        rows.append(row)

    # 컬럼 순서 지정
    columns = [
        "source_bill_id", "source_bill_name",
        "target_bill_id", "target_bill_name",
        "source_summary", "target_summary",
        "source_manual_categories", "target_manual_categories",
        "appeared_methods", "method_support_count",
        "best_rank", "best_score",
        "raw_rank", "raw_score",
        "structured_rank", "structured_score",
        "problem_proposal_rank", "problem_proposal_score",
        "weighted_field_rank", "weighted_field_score",
        "human_relevance_0_to_4",
        "human_issue_match_0_to_2",
        "human_target_match_0_to_2",
        "human_effect_match_0_to_2",
        "human_scope_match_0_to_2",
        "human_article_match_0_to_2",
        "notes",
    ]

    df = pd.DataFrame(rows, columns=columns)

    # CSV 저장
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[출력] CSV 저장 완료: {OUTPUT_CSV}")

    # Excel 저장 (줄바꿈/쉼표 깨짐 방지 및 가독성 최적화 서식 적용)
    os.makedirs(os.path.dirname(OUTPUT_EXCEL), exist_ok=True)
    writer = pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl")
    df.to_excel(writer, index=False, sheet_name="EvaluationSet")
    
    # openpyxl 워크시트 스타일링
    workbook = writer.book
    worksheet = writer.sheets["EvaluationSet"]
    
    # 헤더 채우기 색상 (연한 청회색) 및 폰트
    header_fill = PatternFill(start_color="E6EDF5", end_color="E6EDF5", fill_type="solid")
    header_font = Font(bold=True, size=11)
    
    for col in worksheet.columns:
        col_letter = col[0].column_letter
        col_name = df.columns[col[0].column - 1]
        
        # 헤더 스타일 적용
        header_cell = col[0]
        header_cell.fill = header_fill
        header_cell.font = header_font
        header_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        
        # 데이터 정렬 기준 설정
        # 기본값: 상단(top) 정렬 및 자동 줄 바꿈 적용
        align = Alignment(vertical="top", horizontal="left", wrap_text=True)
        
        # 컬럼 유형별 가독성 높은 넓이 설정
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
            # 기타 메타 데이터 열
            worksheet.column_dimensions[col_letter].width = 15
            
        # 데이터 셀에 정렬 적용
        for cell in col[1:]:
            cell.alignment = align
            
    writer.close()
    print(f"[출력] Excel 저장 완료 (서식 적용됨): {OUTPUT_EXCEL}")

    # JSON 저장
    json_data = {
        "description": "법률 유사도 알고리즘 평가용 라벨링 템플릿",
        "total_pairs": len(rows),
        "source_count": len(source_ids),
        "methods": METHODS,
        "max_candidates_per_source": MAX_CANDIDATES_PER_SOURCE,
        "pairs": rows,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"[출력] JSON 저장 완료: {OUTPUT_JSON}")

    # ── G. 콘솔 출력 ──────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  요약 통계")
    print("=" * 70)
    print(f"  전체 법안 수            : {len(bill_map)}개")
    print(f"  source 법안 수          : {len(source_ids)}개")
    print()
    for method, results in all_results.items():
        print(f"  {method:20s} 로드된 추천 결과 수: {len(results):>5d}개")
    print()
    print(f"  pooling 후 전체 평가쌍 수: {len(final_pairs)}개")

    if per_source_stats:
        avg_candidates = sum(per_source_stats) / len(per_source_stats)
        min_candidates = min(per_source_stats)
        max_candidates = max(per_source_stats)
        print(f"  source당 평균 후보 수    : {avg_candidates:.1f}개")
        print(f"  source당 최소 후보 수    : {min_candidates}개")
        print(f"  source당 최대 후보 수    : {max_candidates}개")

    print()
    print(f"  출력 CSV 경로           : {OUTPUT_CSV}")
    print(f"  출력 Excel 경로         : {OUTPUT_EXCEL}")
    print(f"  출력 JSON 경로          : {OUTPUT_JSON}")
    print()
    print("=" * 70)
    print("  다음 단계:")
    print(f"  1. {OUTPUT_EXCEL} 파일을 엑셀이나 구글 스프레드시트로 열어")
    print("     human_relevance_0_to_4 컬럼에 0~4점을 입력한 후 저장하세요.")
    print("     (Excel 포맷을 사용하면 문장 내 줄바꿈이나 쉼표가 전혀 깨지지 않습니다.)")
    print("  2. python 15_evaluate_methods.py 를 실행하세요.")
    print("=" * 70)


if __name__ == "__main__":
    main()
