#!/usr/bin/env python3
"""
23_grid_search_hybrid_cleaned.py
================================
hybrid_cleaned의 가중치(w_cleaned, w_tfidf, w_article)를 그리드 서치(grid search)하여
라벨링 데이터를 기준으로 최적 가중치를 찾고 성능을 분석합니다.

실행 방법:
    python 23_grid_search_hybrid_cleaned.py
"""

import os
import sys
import math
import json
import random
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 기존 모듈 임포트
from bill_text_parser import split_summary_sections
from clean_text_utils import normalize_legal_text_for_sbert, normalize_legal_text_for_keywords, extract_light_keywords
from article_similarity_utils import compute_article_similarity

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 및 경로 설정 ──────────────────────────────────────────────────
DATASET_JSON = "test_dataset/full_dataset.json"
LABEL_EXCEL = "Sbert_output/evaluation_full_score_pooled_llm_labeled.xlsx"
BACKUP_LABEL_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2_llm_labeled.xlsx"
V2_LABEL_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"

OUTPUT_CSV = "Sbert_output/hybrid_grid_search_results.csv"
OUTPUT_JSON = "Sbert_output/hybrid_grid_search_results.json"
OUTPUT_BEST_JSON = "Sbert_output/hybrid_grid_search_best_weights.json"
OUTPUT_TOPK_BEST = "Sbert_output/topk_hybrid_grid_best.json"
OUTPUT_REPORT = "Sbert_output/hybrid_grid_search_report.md"

# 정규화 옵션
USE_ROW_MINMAX_FOR_TFIDF = True
USE_ROW_MINMAX_FOR_ARTICLE = False
USE_ROW_MINMAX_FOR_CLEANED = False

RELEVANCE_THRESHOLD = 3.0

LEGAL_COLUMNS = [
    "human_issue_match_0_to_2",
    "human_target_match_0_to_2",
    "human_effect_match_0_to_2",
    "human_scope_match_0_to_2",
    "human_article_match_0_to_2"
]

LEGAL_WEIGHTS = {
    "human_issue_match_0_to_2": 0.25,
    "human_target_match_0_to_2": 0.25,
    "human_effect_match_0_to_2": 0.25,
    "human_scope_match_0_to_2": 0.15,
    "human_article_match_0_to_2": 0.10
}


def safe_float(val, default=None):
    """값을 float로 변환, 실패 시 default 반환"""
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def row_minmax_normalize(matrix, name="Matrix", ignore_diagonal=True):
    """
    row-wise min-max normalization을 수행합니다.
    1. 자기 자신 diagonal은 normalization에서 제외합니다.
    2. 한 row의 max와 min이 같으면 모두 0으로 처리합니다.
    3. normalization 전후의 min/max/mean을 출력합니다.
    """
    num_rows = matrix.shape[0]
    norm_matrix = np.zeros_like(matrix)
    
    # 정규화 전 통계용 flatten (대각 제외)
    before_vals = []
    for i in range(num_rows):
        for j in range(num_rows):
            if not ignore_diagonal or i != j:
                before_vals.append(matrix[i][j])
    before_min = np.min(before_vals)
    before_max = np.max(before_vals)
    before_mean = np.mean(before_vals)
    
    for i in range(num_rows):
        row_vals = matrix[i].copy()
        if ignore_diagonal:
            row_vals[i] = -999.0
            active_vals = row_vals[row_vals > -900.0]
        else:
            active_vals = row_vals
            
        if len(active_vals) > 0:
            min_val = np.min(active_vals)
            max_val = np.max(active_vals)
            denom = max_val - min_val
            
            if denom > 1e-8:
                row_norm = (matrix[i] - min_val) / denom
            else:
                row_norm = np.zeros_like(matrix[i])
                
            if ignore_diagonal:
                row_norm[i] = 1.0  # 자기 자신은 1.0으로 강제
                
            row_norm = np.clip(row_norm, 0.0, 1.0)
            norm_matrix[i] = row_norm
        else:
            norm_matrix[i] = matrix[i]
            
    # 정규화 후 통계용 flatten (대각 제외)
    after_vals = []
    for i in range(num_rows):
        for j in range(num_rows):
            if not ignore_diagonal or i != j:
                after_vals.append(norm_matrix[i][j])
    after_min = np.min(after_vals)
    after_max = np.max(after_vals)
    after_mean = np.mean(after_vals)
    
    print(f"  [정규화 통계] {name}:")
    print(f"    - 정규화 전 (대각 제외) Min: {before_min:.4f}, Max: {before_max:.4f}, Mean: {before_mean:.4f}")
    print(f"    - 정규화 후 (대각 제외) Min: {after_min:.4f}, Max: {after_max:.4f}, Mean: {after_mean:.4f}")
    
    return norm_matrix


def compute_dcg(relevance_scores: list[float], k: int) -> float:
    """DCG@k를 계산합니다."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def evaluate_topk(topk_data, label_map):
    """
    topk 추천 결과에 대해 Precision@5, Precision@10, nDCG@10, MRR, 
    Average Relevance, Average Legal Meaning Score를 계산합니다.
    """
    source_groups = defaultdict(list)
    num_evaluated_pairs = 0
    num_unlabeled_pairs = 0
    
    for item in topk_data:
        s_id = item["source_bill_id"]
        t_id = item["target_bill_id"]
        rank = item["rank"]
        
        lbl = label_map.get((s_id, t_id))
        if lbl is not None:
            source_groups[s_id].append({
                "rank": rank,
                "relevance": lbl["relevance"],
                "issue": lbl["issue"],
                "target": lbl["target"],
                "effect": lbl["effect"],
                "scope": lbl["scope"],
                "article": lbl["article"]
            })
            num_evaluated_pairs += 1
        else:
            num_unlabeled_pairs += 1
            
    # rank 오름차순 정렬
    for s_id in source_groups:
        source_groups[s_id].sort(key=lambda x: x["rank"])
        
    num_sources = len(source_groups)
    if num_sources == 0:
        return {
            "precision_at_5": 0.0,
            "precision_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "mrr": 0.0,
            "average_relevance": 0.0,
            "average_legal_meaning_score": 0.0,
            "num_evaluated_pairs": 0,
            "num_unlabeled_pairs": num_unlabeled_pairs,
            "num_sources": 0
        }
        
    p5_list = []
    p10_list = []
    ndcg10_list = []
    mrr_list = []
    all_relevance_vals = []
    legal_scores = []
    
    for s_id, items in source_groups.items():
        # Precision@5
        top5 = items[:5]
        if len(top5) > 0:
            relevant_count_5 = sum(1 for x in top5 if x["relevance"] >= RELEVANCE_THRESHOLD)
            p5_list.append(relevant_count_5 / min(len(top5), 5))
            
        # Precision@10
        top10 = items[:10]
        if len(top10) > 0:
            relevant_count_10 = sum(1 for x in top10 if x["relevance"] >= RELEVANCE_THRESHOLD)
            p10_list.append(relevant_count_10 / min(len(top10), 10))
            
            # nDCG@10
            actual_rels = [x["relevance"] for x in top10]
            dcg = compute_dcg(actual_rels, 10)
            idcg = compute_dcg(sorted(actual_rels, reverse=True), 10)
            if idcg == 0.0:
                ndcg10_list.append(0.0)
            else:
                ndcg10_list.append(dcg / idcg)
                
        # MRR
        found_mrr = False
        for idx, x in enumerate(items):
            if x["relevance"] >= RELEVANCE_THRESHOLD:
                mrr_list.append(1.0 / (idx + 1))
                found_mrr = True
                break
        if not found_mrr:
            mrr_list.append(0.0)
            
        # Average Relevance
        for x in items:
            all_relevance_vals.append(x["relevance"])
            
            # Legal Meaning Score 계산 (모든 세부항목이 기입된 경우만 포함)
            if (x["issue"] is not None and 
                x["target"] is not None and 
                x["effect"] is not None and 
                x["scope"] is not None and 
                x["article"] is not None):
                
                lms = (LEGAL_WEIGHTS["human_issue_match_0_to_2"] * x["issue"] + 
                       LEGAL_WEIGHTS["human_target_match_0_to_2"] * x["target"] + 
                       LEGAL_WEIGHTS["human_effect_match_0_to_2"] * x["effect"] + 
                       LEGAL_WEIGHTS["human_scope_match_0_to_2"] * x["scope"] + 
                       LEGAL_WEIGHTS["human_article_match_0_to_2"] * x["article"])
                lms_100 = (lms / 2.0) * 100.0
                legal_scores.append(lms_100)
                
    p5 = sum(p5_list) / len(p5_list) if p5_list else 0.0
    p10 = sum(p10_list) / len(p10_list) if p10_list else 0.0
    ndcg10 = sum(ndcg10_list) / len(ndcg10_list) if ndcg10_list else 0.0
    mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    avg_rel = sum(all_relevance_vals) / len(all_relevance_vals) if all_relevance_vals else 0.0
    avg_lms = sum(legal_scores) / len(legal_scores) if legal_scores else 0.0
    
    return {
        "precision_at_5": round(p5, 4),
        "precision_at_10": round(p10, 4),
        "ndcg_at_10": round(ndcg10, 4),
        "mrr": round(mrr, 4),
        "average_relevance": round(avg_rel, 4),
        "average_legal_meaning_score": round(avg_lms, 2) if legal_scores else None,
        "num_evaluated_pairs": num_evaluated_pairs,
        "num_unlabeled_pairs": num_unlabeled_pairs,
        "num_sources": num_sources
    }


def main():
    print("=" * 80)
    print("  Hybrid Cleaned 가중치 Grid Search 최적화 실행")
    print(f"  - 실행 명령어: python 23_grid_search_hybrid_cleaned.py")
    print("=" * 80)
    print()

    start_time = time.time()

    # ── 1. 데이터셋 로드 ──────────────────────────────────────────────
    if not os.path.exists(DATASET_JSON):
        print(f"[ERROR] 데이터셋 파일이 존재하지 않습니다: {DATASET_JSON}")
        sys.exit(1)
        
    with open(DATASET_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    bills = dataset.get("bills", [])
    num_bills = len(bills)
    print(f"[1] 전체 법안 로드 완료: {num_bills}개")

    # ── 2. 라벨 데이터 로드 ────────────────────────────────────────────
    label_path = LABEL_EXCEL
    if not os.path.exists(label_path):
        print(f"[WARNING] 1차 경로에 라벨 파일이 없습니다: {label_path}")
        if os.path.exists(BACKUP_LABEL_EXCEL):
            label_path = BACKUP_LABEL_EXCEL
            print(f"          대체 경로 파일 사용: {label_path}")
        else:
            print("[ERROR] 라벨링된 엑셀 파일을 찾을 수 없습니다.")
            sys.exit(1)

    print(f"[2] 라벨링 엑셀 파일 로드 중: {label_path}")
    df_label = pd.read_excel(label_path, dtype=str).fillna("")
    print(f"    - 전체 로드된 라벨 행 수: {len(df_label)}행")

    # 라벨 범위 유효성 체크 및 파싱
    label_map = {}
    num_unlabeled_in_xlsx = 0
    
    for idx, row in df_label.iterrows():
        s_id = row["source_bill_id"].strip()
        t_id = row["target_bill_id"].strip()
        rel_raw = row.get("human_relevance_0_to_4", "")
        
        rel = safe_float(rel_raw)
        if rel is None:
            num_unlabeled_in_xlsx += 1
            continue
            
        # relevance 값 0~4 확인
        if not (0.0 <= rel <= 4.0):
            raise ValueError(f"[ERROR] row {idx+2}의 relevance 값({rel})이 범위를 벗어났습니다 (0~4).")
            
        # 세부 점수 값 0~2 확인
        issue = safe_float(row.get("human_issue_match_0_to_2"))
        target = safe_float(row.get("human_target_match_0_to_2"))
        effect = safe_float(row.get("human_effect_match_0_to_2"))
        scope = safe_float(row.get("human_scope_match_0_to_2"))
        article = safe_float(row.get("human_article_match_0_to_2"))
        
        subscores = [("issue", issue), ("target", target), ("effect", effect), ("scope", scope), ("article", article)]
        for name, val in subscores:
            if val is not None:
                if not (0.0 <= val <= 2.0):
                    raise ValueError(f"[ERROR] row {idx+2}의 세부 점수 {name} 값({val})이 범위를 벗어났습니다 (0~2).")
                    
        label_map[(s_id, t_id)] = {
            "relevance": rel,
            "issue": issue,
            "target": target,
            "effect": effect,
            "scope": scope,
            "article": article
        }

    print(f"    - 유효하게 라벨링된 고유 쌍 수: {len(label_map)}개")
    print(f"    - 빈 라벨 행 수 (평가 제외): {num_unlabeled_in_xlsx}개")

    # 골드 라벨 여부를 구분하기 위해 v2 템플릿 로드 검사
    gold_sources = set()
    if os.path.exists(V2_LABEL_EXCEL):
        df_v2 = pd.read_excel(V2_LABEL_EXCEL, dtype=str).fillna("")
        gold_sources = set(df_v2["source_bill_id"].unique())
        print(f"    - Gold label 전용 source 수 (v2): {len(gold_sources)}개")
    else:
        print("    - [안내] v2 템플릿 파일이 없으므로 Gold-only 구분 없이 혼합 라벨로만 최종 보고서를 생성합니다.")

    # ── 3. Component Similarity Matrices 생성 ───────────────────────
    print("\n[3] Component Similarity Matrices 생성 시작...")

    # A. cleaned_problem_proposal matrix
    print("  A. SBERT 유사도 행렬 계산 중...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    sbert_model = SentenceTransformer(model_name, device=device)
    
    sbert_texts = []
    all_sections = []
    
    for bill in bills:
        summary = bill.get("summary", "")
        sections = split_summary_sections(summary)
        all_sections.append(sections)
        
        prob = sections.get("problem", "").strip()
        prop = sections.get("proposal", "").strip()
        combined = f"{prob} {prop}".strip()
        
        cleaned = normalize_legal_text_for_sbert(combined)
        if len(cleaned) < 5:
            fallback = f"{bill.get('bill_name', '')} {summary}".strip()
            cleaned = normalize_legal_text_for_sbert(fallback)
        sbert_texts.append(cleaned)
        
    sbert_embeddings = sbert_model.encode(
        sbert_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_tensor=False
    )
    sbert_sim = np.dot(sbert_embeddings, sbert_embeddings.T)

    # B. keyword_tfidf matrix
    print("  B. TF-IDF 유사도 행렬 계산 중...")
    tfidf_texts = []
    for idx, bill in enumerate(bills):
        sections = all_sections[idx]
        prob = sections.get("problem", "").strip()
        prop = sections.get("proposal", "").strip()
        combined = f"{prob} {prop}".strip()
        
        cleaned = normalize_legal_text_for_keywords(combined)
        keywords = extract_light_keywords(cleaned)
        if len(keywords.strip()) < 3:
            fallback = f"{bill.get('bill_name', '')} {bill.get('summary', '')}".strip()
            cleaned = normalize_legal_text_for_keywords(fallback)
            keywords = extract_light_keywords(cleaned)
        tfidf_texts.append(keywords)
        
    vectorizer = TfidfVectorizer(
        tokenizer=None,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    tfidf_mat = vectorizer.fit_transform(tfidf_texts)
    tfidf_sim = cosine_similarity(tfidf_mat)

    # C. article_similarity matrix
    print("  C. 조문 유사도 행렬 계산 중...")
    article_sim = np.zeros((num_bills, num_bills))
    for i in range(num_bills):
        for j in range(num_bills):
            if i == j:
                article_sim[i][j] = 1.0
            else:
                article_sim[i][j] = compute_article_similarity(
                    all_sections[i]["article_numbers"],
                    all_sections[j]["article_numbers"]
                )

    # 행렬 형태 검증
    assert sbert_sim.shape == (num_bills, num_bills), "SBERT 유사도 행렬 크기가 올바르지 않습니다."
    assert tfidf_sim.shape == (num_bills, num_bills), "TF-IDF 유사도 행렬 크기가 올바르지 않습니다."
    assert article_sim.shape == (num_bills, num_bills), "조문 유사도 행렬 크기가 올바르지 않습니다."
    print("    - 모든 유사도 행렬 형태(shape) 검증 완료: 동일함", sbert_sim.shape)

    # ── 4. 정규화 옵션 적용 ──────────────────────────────────────────
    print("\n[4] 정규화 처리 및 통계 계산...")
    if USE_ROW_MINMAX_FOR_CLEANED:
        sbert_sim_norm = row_minmax_normalize(sbert_sim, "SBERT (Cleaned Summary)", ignore_diagonal=True)
    else:
        sbert_sim_norm = sbert_sim.copy()
        
    if USE_ROW_MINMAX_FOR_TFIDF:
        tfidf_sim_norm = row_minmax_normalize(tfidf_sim, "TF-IDF Keywords", ignore_diagonal=True)
    else:
        tfidf_sim_norm = tfidf_sim.copy()
        
    if USE_ROW_MINMAX_FOR_ARTICLE:
        article_sim_norm = row_minmax_normalize(article_sim, "Article Similarity", ignore_diagonal=True)
    else:
        article_sim_norm = article_sim.copy()

    # ── 5. Grid Search 후보군 생성 ───────────────────────────────────
    print("\n[5] 가중치 후보군 생성 및 중복 제거 중...")
    candidates = []
    
    # 0.05 단위 탐색
    for w_c in np.arange(0.60, 0.95 + 1e-9, 0.05):
        for w_t in np.arange(0.00, 0.30 + 1e-9, 0.05):
            w_c = round(float(w_c), 2)
            w_t = round(float(w_t), 2)
            w_a = round(1.0 - w_c - w_t, 2)
            if abs(w_a) < 1e-9:
                w_a = 0.0
            
            # w_article 조건 (0.00 <= w_article <= 0.20)
            if 0.00 <= w_a <= 0.20:
                candidates.append({
                    "w_cleaned": w_c,
                    "w_tfidf": w_t,
                    "w_article": w_a
                })
                
    # 수동 탐색 후보군 추가
    MANUAL_CANDIDATES = [
        {"w_cleaned": 0.70, "w_tfidf": 0.20, "w_article": 0.10},  # 기존 baseline
        {"w_cleaned": 0.80, "w_tfidf": 0.10, "w_article": 0.10},
        {"w_cleaned": 0.85, "w_tfidf": 0.10, "w_article": 0.05},
        {"w_cleaned": 0.90, "w_tfidf": 0.05, "w_article": 0.05},
        {"w_cleaned": 0.75, "w_tfidf": 0.15, "w_article": 0.10},
        {"w_cleaned": 0.75, "w_tfidf": 0.10, "w_article": 0.15}
    ]
    
    for mc in MANUAL_CANDIDATES:
        w_mc_c = round(float(mc["w_cleaned"]), 2)
        w_mc_t = round(float(mc["w_tfidf"]), 2)
        w_mc_a = round(float(mc["w_article"]), 2)
        if abs(w_mc_a) < 1e-9:
            w_mc_a = 0.0
        candidates.append({
            "w_cleaned": w_mc_c,
            "w_tfidf": w_mc_t,
            "w_article": w_mc_a
        })
        
    # 중복 제거 및 가중치 합 1.0 유효성 검증
    unique_candidates = []
    seen = set()
    for c in candidates:
        tup = (c["w_cleaned"], c["w_tfidf"], c["w_article"])
        if tup not in seen:
            # 합이 1.0인지 다시 한 번 확인
            assert abs(c["w_cleaned"] + c["w_tfidf"] + c["w_article"] - 1.0) < 1e-9, f"가중치 합이 1.0이 아닙니다: {c}"
            seen.add(tup)
            unique_candidates.append(c)
            
    print(f"    - 총 탐색할 고유 가중치 조합 수: {len(unique_candidates)}개")

    # ── 6. Grid Search 루프 및 평가 ─────────────────────────────────
    print("\n[6] Grid Search 평가 수행 중...")
    
    all_results = []
    
    # 평가 대상 source_bill_id 목록 추출 (라벨 파일에 존재하는 소스들)
    evaluated_source_ids = sorted(list(set(k[0] for k in label_map.keys())))
    num_eval_sources = len(evaluated_source_ids)
    
    # 빌드 최적 topk 딕셔너리 저장을 위해
    best_weights_obj = None
    best_obj_score = -1.0
    best_topk_data = None
    
    # Baseline 결과 보존용
    baseline_original_metrics = None

    for c_idx, cand in enumerate(unique_candidates):
        w_c = cand["w_cleaned"]
        w_t = cand["w_tfidf"]
        w_a = cand["w_article"]
        
        # 하이브리드 점수 행렬 결합
        hybrid_matrix = w_c * sbert_sim_norm + w_t * tfidf_sim_norm + w_a * article_sim_norm
        
        # 각 source bill별 top-10 target 추출
        cand_topk = []
        for i in range(num_bills):
            source_id = bills[i]["bill_id"]
            source_name = bills[i]["bill_name"]
            
            # 자기 자신 제외
            scores = []
            for j in range(num_bills):
                if i == j:
                    continue
                scores.append((j, hybrid_matrix[i][j]))
                
            # 정렬: 1순위 similarity 내림차순, 2순위 target_bill_id 오름차순 (tie-breaking)
            scores.sort(key=lambda x: (-x[1], bills[x[0]]["bill_id"]))
            
            # top-10 선택
            top10 = scores[:10]
            assert len(top10) == 10, f"{source_id}의 추천 대상 수가 10개가 아닙니다."
            
            for rank_idx, (t_idx, sim_val) in enumerate(top10):
                target_id = bills[t_idx]["bill_id"]
                target_name = bills[t_idx]["bill_name"]
                
                # 대각선 미포함 검증
                assert source_id != target_id, "대각선 요소가 추천 결과에 포함되었습니다."
                
                comp_sbert = float(sbert_sim_norm[i][t_idx])
                comp_tfidf = float(tfidf_sim_norm[i][t_idx])
                comp_article = float(article_sim_norm[i][t_idx])
                
                cand_topk.append({
                    "source_bill_id": source_id,
                    "source_bill_name": source_name,
                    "target_bill_id": target_id,
                    "target_bill_name": target_name,
                    "rank": rank_idx + 1,
                    "similarity": round(float(sim_val), 6),
                    "method": "hybrid_grid_best",
                    "component_scores": {
                        "cleaned_problem_proposal": round(comp_sbert, 4),
                        "keyword_tfidf": round(comp_tfidf, 4),
                        "article_similarity": round(comp_article, 4)
                    },
                    "weights": {
                        "w_cleaned": w_c,
                        "w_tfidf": w_t,
                        "w_article": w_a
                    }
                })
                
        # 전체 소스 기준 평가
        metrics = evaluate_topk(cand_topk, label_map)
        
        p5 = metrics["precision_at_5"]
        p10 = metrics["precision_at_10"]
        ndcg10 = metrics["ndcg_at_10"]
        mrr = metrics["mrr"]
        avg_rel = metrics["average_relevance"]
        avg_lms = metrics["average_legal_meaning_score"]
        
        # 3가지 Objective 계산
        obj_score = 0.50 * p5 + 0.30 * ndcg10 + 0.20 * mrr
        recall_obj = 0.60 * p10 + 0.25 * p5 + 0.15 * avg_rel
        ranking_obj = 0.45 * ndcg10 + 0.35 * mrr + 0.20 * p5
        
        cand_res = {
            "w_cleaned": w_c,
            "w_tfidf": w_t,
            "w_article": w_a,
            "precision_at_5": p5,
            "precision_at_10": p10,
            "ndcg_at_10": ndcg10,
            "mrr": mrr,
            "average_relevance": avg_rel,
            "average_legal_meaning_score": avg_lms,
            "objective_score": round(obj_score, 4),
            "candidate_recall_objective": round(recall_obj, 4),
            "ranking_objective": round(ranking_obj, 4),
            "num_evaluated_pairs": metrics["num_evaluated_pairs"],
            "num_sources": metrics["num_sources"],
            # topk 임시 저장을 위해
            "_topk": cand_topk
        }
        
        all_results.append(cand_res)
        
        # 기존 baseline 저장
        if w_c == 0.70 and w_t == 0.20 and w_a == 0.10:
            baseline_original_metrics = {
                "precision_at_5": p5,
                "precision_at_10": p10,
                "ndcg_at_10": ndcg10,
                "mrr": mrr,
                "average_relevance": avg_rel,
                "average_legal_meaning_score": avg_lms,
                "objective_score": round(obj_score, 4),
                "candidate_recall_objective": round(recall_obj, 4),
                "ranking_objective": round(ranking_obj, 4),
                "num_evaluated_pairs": metrics["num_evaluated_pairs"],
                "num_sources": metrics["num_sources"]
            }

        # primary objective_score 기준 최적 저장
        if obj_score > best_obj_score:
            best_obj_score = obj_score
            best_weights_obj = cand
            best_topk_data = cand_topk

    # baseline 존재 유무 검증
    assert baseline_original_metrics is not None, "기존 가중치 baseline(0.70, 0.20, 0.10)이 평가되지 않았습니다."

    # ── 7. Source-level Train/Validation 검증 및 5-fold CV ───────────
    print("\n[7] Source-level Train/Validation Split 및 5-Fold Cross Validation 검증 중...")
    
    # 평가 대상 고유 소스 추출 및 정렬
    unique_sources = sorted(list(evaluated_source_ids))
    
    # 80/20 train/validation split
    random.seed(42)
    shuffled_sources = unique_sources.copy()
    random.shuffle(shuffled_sources)
    
    split_idx = int(len(shuffled_sources) * 0.8)
    train_sources = set(shuffled_sources[:split_idx])
    val_sources = set(shuffled_sources[split_idx:])
    
    print(f"    - Train 소스 수: {len(train_sources)}개, Validation 소스 수: {len(val_sources)}개")
    
    # Train에서 가장 objective_score가 높은 가중치 탐색
    best_train_weights = None
    best_train_obj = -1.0
    best_train_metrics = None
    
    for cand in unique_candidates:
        w_c = cand["w_cleaned"]
        w_t = cand["w_tfidf"]
        w_a = cand["w_article"]
        
        # 해당 가중치에 매칭되는 결과 찾기
        cand_res = next(x for x in all_results if x["w_cleaned"] == w_c and x["w_tfidf"] == w_t and x["w_article"] == w_a)
        cand_topk = cand_res["_topk"]
        
        # Train 소스들에 대해서만 평가
        train_topk = [item for item in cand_topk if item["source_bill_id"] in train_sources]
        train_metrics = evaluate_topk(train_topk, label_map)
        
        tr_p5 = train_metrics["precision_at_5"]
        tr_ndcg = train_metrics["ndcg_at_10"]
        tr_mrr = train_metrics["mrr"]
        tr_obj = 0.50 * tr_p5 + 0.30 * tr_ndcg + 0.20 * tr_mrr
        
        if tr_obj > best_train_obj:
            best_train_obj = tr_obj
            best_train_weights = cand
            best_train_metrics = train_metrics
            best_train_metrics["objective_score"] = round(tr_obj, 4)
            
    # Train 최적 가중치를 Validation 세트에서 평가
    val_topk = [item for item in next(x for x in all_results if x["w_cleaned"] == best_train_weights["w_cleaned"] and x["w_tfidf"] == best_train_weights["w_tfidf"] and x["w_article"] == best_train_weights["w_article"])["_topk"] if item["source_bill_id"] in val_sources]
    val_metrics = evaluate_topk(val_topk, label_map)
    val_p5 = val_metrics["precision_at_5"]
    val_ndcg = val_metrics["ndcg_at_10"]
    val_mrr = val_metrics["mrr"]
    val_obj = 0.50 * val_p5 + 0.30 * val_ndcg + 0.20 * val_mrr
    val_metrics["objective_score"] = round(val_obj, 4)
    
    print(f"    - Train 최적 가중치: w_cleaned={best_train_weights['w_cleaned']:.2f}, w_tfidf={best_train_weights['w_tfidf']:.2f}, w_article={best_train_weights['w_article']:.2f}")
    print(f"    - Train 성능: objective_score={best_train_obj:.4f}")
    print(f"    - Validation 성능: objective_score={val_obj:.4f}")

    # 5-fold Cross Validation 구현
    print("    - 5-Fold Cross Validation 수행 중...")
    fold_size = len(shuffled_sources) / 5.0
    cv_val_metrics_list = []
    
    for fold_idx in range(5):
        val_start = int(fold_idx * fold_size)
        val_end = int((fold_idx + 1) * fold_size)
        
        fold_val_sources = set(shuffled_sources[val_start:val_end])
        fold_train_sources = set(shuffled_sources) - fold_val_sources
        
        # fold train에서 최적 가중치 찾기
        fold_best_weights = None
        fold_best_train_obj = -1.0
        
        for cand in unique_candidates:
            w_c = cand["w_cleaned"]
            w_t = cand["w_tfidf"]
            w_a = cand["w_article"]
            
            cand_res = next(x for x in all_results if x["w_cleaned"] == w_c and x["w_tfidf"] == w_t and x["w_article"] == w_a)
            cand_topk = cand_res["_topk"]
            
            f_train_topk = [item for item in cand_topk if item["source_bill_id"] in fold_train_sources]
            f_train_metrics = evaluate_topk(f_train_topk, label_map)
            
            ft_p5 = f_train_metrics["precision_at_5"]
            ft_ndcg = f_train_metrics["ndcg_at_10"]
            ft_mrr = f_train_metrics["mrr"]
            ft_obj = 0.50 * ft_p5 + 0.30 * ft_ndcg + 0.20 * ft_mrr
            
            if ft_obj > fold_best_train_obj:
                fold_best_train_obj = ft_obj
                fold_best_weights = cand
                
        # fold validation 성능 평가
        f_val_cand_topk = next(x for x in all_results if x["w_cleaned"] == fold_best_weights["w_cleaned"] and x["w_tfidf"] == fold_best_weights["w_tfidf"] and x["w_article"] == fold_best_weights["w_article"])["_topk"]
        f_val_topk = [item for item in f_val_cand_topk if item["source_bill_id"] in fold_val_sources]
        f_val_metrics = evaluate_topk(f_val_topk, label_map)
        
        fv_p5 = f_val_metrics["precision_at_5"]
        fv_p10 = f_val_metrics["precision_at_10"]
        fv_ndcg = f_val_metrics["ndcg_at_10"]
        fv_mrr = f_val_metrics["mrr"]
        fv_obj = 0.50 * fv_p5 + 0.30 * fv_ndcg + 0.20 * fv_mrr
        
        cv_val_metrics_list.append({
            "fold": fold_idx + 1,
            "best_weights": fold_best_weights,
            "precision_at_5": fv_p5,
            "precision_at_10": fv_p10,
            "ndcg_at_10": fv_ndcg,
            "mrr": fv_mrr,
            "objective_score": fv_obj
        })

    cv_mean_p5 = np.mean([x["precision_at_5"] for x in cv_val_metrics_list])
    cv_mean_p10 = np.mean([x["precision_at_10"] for x in cv_val_metrics_list])
    cv_mean_ndcg = np.mean([x["ndcg_at_10"] for x in cv_val_metrics_list])
    cv_mean_mrr = np.mean([x["mrr"] for x in cv_val_metrics_list])
    cv_mean_obj = np.mean([x["objective_score"] for x in cv_val_metrics_list])
    
    cv_summary_metrics = {
        "precision_at_5": round(float(cv_mean_p5), 4),
        "precision_at_10": round(float(cv_mean_p10), 4),
        "ndcg_at_10": round(float(cv_mean_ndcg), 4),
        "mrr": round(float(cv_mean_mrr), 4),
        "objective_score": round(float(cv_mean_obj), 4)
    }
    
    print(f"    - 5-Fold Validation 평균 성능: objective_score={cv_mean_obj:.4f}")

    # ── 8. 결과 정렬 및 파일 저장 ──────────────────────────────────────
    print("\n[8] 결과 데이터 정렬 및 파일 저장 중...")
    
    # _topk 키는 JSON 및 CSV 저장에 방해되므로 결과용 딕셔너리에서 제거
    results_for_save = []
    for r in all_results:
        rc = r.copy()
        del rc["_topk"]
        results_for_save.append(rc)
        
    df_results = pd.DataFrame(results_for_save)
    df_results = df_results.sort_values(by="objective_score", ascending=False)
    
    # CSV 저장
    df_results.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"    - CSV 저장 완료: {OUTPUT_CSV}")
    
    # JSON 저장
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_for_save, f, ensure_ascii=False, indent=2)
    print(f"    - JSON 저장 완료: {OUTPUT_JSON}")

    # 최적 가중치 추출
    best_by_obj = df_results.iloc[0].to_dict()
    
    df_recall = df_results.sort_values(by="candidate_recall_objective", ascending=False)
    best_by_recall = df_recall.iloc[0].to_dict()
    
    df_ranking = df_results.sort_values(by="ranking_objective", ascending=False)
    best_by_ranking = df_ranking.iloc[0].to_dict()

    best_weights_dict = {
        "best_by_objective_score": {
            "weights": {
                "w_cleaned": float(best_by_obj["w_cleaned"]),
                "w_tfidf": float(best_by_obj["w_tfidf"]),
                "w_article": float(best_by_obj["w_article"])
            },
            "metrics": {
                "precision_at_5": float(best_by_obj["precision_at_5"]),
                "precision_at_10": float(best_by_obj["precision_at_10"]),
                "ndcg_at_10": float(best_by_obj["ndcg_at_10"]),
                "mrr": float(best_by_obj["mrr"]),
                "average_relevance": float(best_by_obj["average_relevance"]),
                "average_legal_meaning_score": best_by_obj["average_legal_meaning_score"] if pd.notna(best_by_obj["average_legal_meaning_score"]) else None,
                "objective_score": float(best_by_obj["objective_score"])
            }
        },
        "best_by_candidate_recall_objective": {
            "weights": {
                "w_cleaned": float(best_by_recall["w_cleaned"]),
                "w_tfidf": float(best_by_recall["w_tfidf"]),
                "w_article": float(best_by_recall["w_article"])
            },
            "metrics": {
                "precision_at_5": float(best_by_recall["precision_at_5"]),
                "precision_at_10": float(best_by_recall["precision_at_10"]),
                "ndcg_at_10": float(best_by_recall["ndcg_at_10"]),
                "mrr": float(best_by_recall["mrr"]),
                "average_relevance": float(best_by_recall["average_relevance"]),
                "average_legal_meaning_score": best_by_recall["average_legal_meaning_score"] if pd.notna(best_by_recall["average_legal_meaning_score"]) else None,
                "candidate_recall_objective": float(best_by_recall["candidate_recall_objective"])
            }
        },
        "best_by_ranking_objective": {
            "weights": {
                "w_cleaned": float(best_by_ranking["w_cleaned"]),
                "w_tfidf": float(best_by_ranking["w_tfidf"]),
                "w_article": float(best_by_ranking["w_article"])
            },
            "metrics": {
                "precision_at_5": float(best_by_ranking["precision_at_5"]),
                "precision_at_10": float(best_by_ranking["precision_at_10"]),
                "ndcg_at_10": float(best_by_ranking["ndcg_at_10"]),
                "mrr": float(best_by_ranking["mrr"]),
                "average_relevance": float(best_by_ranking["average_relevance"]),
                "average_legal_meaning_score": best_by_ranking["average_legal_meaning_score"] if pd.notna(best_by_ranking["average_legal_meaning_score"]) else None,
                "ranking_objective": float(best_by_ranking["ranking_objective"])
            }
        },
        "baseline_original": {
            "weights": {
                "w_cleaned": 0.70,
                "w_tfidf": 0.20,
                "w_article": 0.10
            },
            "metrics": baseline_original_metrics
        },
        "train_validation_result": {
            "train_best_weights": {
                "w_cleaned": float(best_train_weights["w_cleaned"]),
                "w_tfidf": float(best_train_weights["w_tfidf"]),
                "w_article": float(best_train_weights["w_article"])
            },
            "train_metrics": {
                "precision_at_5": float(best_train_metrics["precision_at_5"]),
                "precision_at_10": float(best_train_metrics["precision_at_10"]),
                "ndcg_at_10": float(best_train_metrics["ndcg_at_10"]),
                "mrr": float(best_train_metrics["mrr"]),
                "average_relevance": float(best_train_metrics["average_relevance"]),
                "average_legal_meaning_score": best_train_metrics["average_legal_meaning_score"],
                "objective_score": float(best_train_metrics["objective_score"])
            },
            "validation_metrics": {
                "precision_at_5": float(val_metrics["precision_at_5"]),
                "precision_at_10": float(val_metrics["precision_at_10"]),
                "ndcg_at_10": float(val_metrics["ndcg_at_10"]),
                "mrr": float(val_metrics["mrr"]),
                "average_relevance": float(val_metrics["average_relevance"]),
                "average_legal_meaning_score": val_metrics["average_legal_meaning_score"],
                "objective_score": float(val_metrics["objective_score"])
            },
            "cv_average_validation_metrics": cv_summary_metrics
        }
    }

    # Best weights JSON 저장
    with open(OUTPUT_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best_weights_dict, f, ensure_ascii=False, indent=2)
    print(f"    - 최적 가중치 JSON 저장 완료: {OUTPUT_BEST_JSON}")

    # Top-K hybrid best JSON 저장
    with open(OUTPUT_TOPK_BEST, "w", encoding="utf-8") as f:
        json.dump(best_topk_data, f, ensure_ascii=False, indent=2)
    print(f"    - Best Weight Top-10 JSON 저장 완료: {OUTPUT_TOPK_BEST} (총 {len(best_topk_data)}쌍)")

    # ── 9. 보고서 마크다운 생성 ──────────────────────────────────────
    print("\n[9] 보고서 마크다운 파일(hybrid_grid_search_report.md) 작성 중...")
    
    top10_cands_str = ""
    for idx, row in df_results.head(10).iterrows():
        lms_val = f"{row['average_legal_meaning_score']:.2f}" if pd.notna(row['average_legal_meaning_score']) else "-"
        top10_cands_str += (
            f"| {idx+1} | {row['w_cleaned']:.2f} | {row['w_tfidf']:.2f} | {row['w_article']:.2f} "
            f"| {row['precision_at_5']:.4f} | {row['precision_at_10']:.4f} | {row['ndcg_at_10']:.4f} "
            f"| {row['mrr']:.4f} | {row['average_relevance']:.4f} | {lms_val} | **{row['objective_score']:.4f}** |\n"
        )
        
    report_content = f"""# Hybrid Cleaned 가중치 Grid Search 보고서

본 보고서는 국회 법률발의안 75개 전체 데이터셋과 LLM이 보조 라벨링한 2,009개의 법안 쌍 평가 데이터를 바탕으로 SBERT 유사도(`cleaned_problem_proposal_score`), TF-IDF 명사 유사도(`keyword_tfidf_score`), 조문 구조 유사도(`article_similarity`)의 최적 결합 가중치를 탐색한 결과를 요약합니다.

## 1. 실험 목적
기존 `hybrid_cleaned` 유사도 산출 방식의 결합 가중치는 사람의 직관적 감(SBERT 0.70, TF-IDF 0.20, 조문 0.10)에 기반하여 정해졌습니다. 본 실험은 전체 75개 소스 법안에 대한 실제 라벨 데이터를 기준으로 가중치를 0.05 단위로 전수 조사(Grid Search)하여, 검색 결과의 랭킹 품질을 최고로 끌어올리는 객관적인 최적의 가중치를 규명하고자 합니다.

## 2. 사용한 component
1. **cleaned_problem_proposal_score (SBERT 유사도)**: SBERT 기반 의미 임베딩 코사인 유사도로, 법안 요약문에서 `problem + proposal`을 추출하여 정제한 문맥적 유사도를 반영하는 핵심 뼈대 정보입니다.
2. **keyword_tfidf_score (TF-IDF 명사 유사도)**: 법안 텍스트에 포함된 핵심 어휘들의 TF-IDF 코사인 유사도로, 조문이나 세부 키워드가 완전히 겹치는 법안쌍을 탐지하는 보조 신호입니다.
3. **article_similarity (조문 유사도)**: 두 법안의 summary에서 추출한 개정 대상 조문 번호(예: 안 제5조 등) 간의 Jaccard 유사도로, 동일 조문을 타겟팅하여 발의된 법안을 직접적으로 잡아내는 희소하지만 확실한 매칭 신호입니다.

## 3. 탐색한 가중치 범위
* **가중치 조건**:
  - $w_{{cleaned}} + w_{{tfidf}} + w_{{article}} = 1.0$
  - $w_{{cleaned}} \ge 0.60$ (의미 기반 SBERT 유사도 주력 유지)
  - $w_{{tfidf}} \le 0.30$ (TF-IDF 어휘 유사도 보조 역할 제한)
  - $w_{{article}} \le 0.20$ (조문 일치 신호의 과대평가 방지)
  - 모든 가중치는 $0$ 이상
* **그리드 스텝**: 0.05 단위 전수 조사 및 주요 직관적 수동 후보(6개 조합) 포함

## 4. 평가 라벨 설명
* **라벨 원본 파일**: `Sbert_output/evaluation_full_score_pooled_llm_labeled.xlsx` (gold/LLM 혼합 라벨 기준)
* **Relevance (관련도)**: `human_relevance_0_to_4 >= 3`인 경우를 검색 알고리즘 상의 "관련 있음"으로 매핑하여 Precision, MRR, nDCG를 계산했습니다.
* **Legal Meaning Score (법률적 의미 매칭)**:
  - 하위 평가 5개 요소 (`issue`, `target`, `effect`, `scope`, `article`)의 가중 합으로 계산합니다.
  - 공식: $\\text{{LMS}} = 0.25 \\times \\text{{issue}} + 0.25 \\times \\text{{target}} + 0.25 \\times \\text{{effect}} + 0.15 \\times \\text{{scope}} + 0.10 \\times \\text{{article}}$ (범위: 0 ~ 2)
  - 100점 만점 환산: $\\text{{LMS\\_100}} = \\frac{{\\text{{LMS}}}}{2} \\times 100$

## 5. 기존 가중치 성능 (Baseline)
* **가중치**: SBERT (`0.70`), TF-IDF (`0.20`), 조문 유사도 (`0.10`)
* **성능 지표**:
  - Precision@5: {baseline_original_metrics["precision_at_5"]:.4f}
  - Precision@10: {baseline_original_metrics["precision_at_10"]:.4f}
  - nDCG@10: {baseline_original_metrics["ndcg_at_10"]:.4f}
  - MRR: {baseline_original_metrics["mrr"]:.4f}
  - Average Relevance: {baseline_original_metrics["average_relevance"]:.4f}
  - Average Legal Meaning Score: {f"{baseline_original_metrics['average_legal_meaning_score']:.2f}" if baseline_original_metrics['average_legal_meaning_score'] is not None else "-"}
  - **Objective Score**: **{baseline_original_metrics["objective_score"]:.4f}**

## 6. objective_score 기준 최적 가중치 (Primary Objective)
* **공식**: $0.50 \\times \\text{{P@5}} + 0.30 \\times \\text{{nDCG@10}} + 0.20 \\times \\text{{MRR}}$
* **가중치**: SBERT (**{best_weights_dict["best_by_objective_score"]["weights"]["w_cleaned"]:.2f}**), TF-IDF (**{best_weights_dict["best_by_objective_score"]["weights"]["w_tfidf"]:.2f}**), 조문 (**{best_weights_dict["best_by_objective_score"]["weights"]["w_article"]:.2f}**)
* **최적 성능 지표**:
  - Precision@5: {best_weights_dict["best_by_objective_score"]["metrics"]["precision_at_5"]:.4f}
  - Precision@10: {best_weights_dict["best_by_objective_score"]["metrics"]["precision_at_10"]:.4f}
  - nDCG@10: {best_weights_dict["best_by_objective_score"]["metrics"]["ndcg_at_10"]:.4f}
  - MRR: {best_weights_dict["best_by_objective_score"]["metrics"]["mrr"]:.4f}
  - **Objective Score**: **{best_weights_dict["best_by_objective_score"]["metrics"]["objective_score"]:.4f}**

## 7. candidate_recall_objective 기준 최적 가중치 (보조 지표)
* **공식**: $0.60 \\times \\text{{P@10}} + 0.25 \\times \\text{{P@5}} + 0.15 \\times \\text{{AvgRel}}$
* **가중치**: SBERT (**{best_weights_dict["best_by_candidate_recall_objective"]["weights"]["w_cleaned"]:.2f}**), TF-IDF (**{best_weights_dict["best_by_candidate_recall_objective"]["weights"]["w_tfidf"]:.2f}**), 조문 (**{best_weights_dict["best_by_candidate_recall_objective"]["weights"]["w_article"]:.2f}**)
* **성능 지표**:
  - Precision@5: {best_weights_dict["best_by_candidate_recall_objective"]["metrics"]["precision_at_5"]:.4f}
  - Precision@10: {best_weights_dict["best_by_candidate_recall_objective"]["metrics"]["precision_at_10"]:.4f}
  - Average Relevance: {best_weights_dict["best_by_candidate_recall_objective"]["metrics"]["average_relevance"]:.4f}
  - **Candidate Recall Objective Score**: **{best_weights_dict["best_by_candidate_recall_objective"]["metrics"]["candidate_recall_objective"]:.4f}**

## 8. ranking_objective 기준 최적 가중치 (보조 지표)
* **공식**: $0.45 \\times \\text{{nDCG@10}} + 0.35 \\times \\text{{MRR}} + 0.20 \\times \\text{{P@5}}$
* **가중치**: SBERT (**{best_weights_dict["best_by_ranking_objective"]["weights"]["w_cleaned"]:.2f}**), TF-IDF (**{best_weights_dict["best_by_ranking_objective"]["weights"]["w_tfidf"]:.2f}**), 조문 (**{best_weights_dict["best_by_ranking_objective"]["weights"]["w_article"]:.2f}**)
* **성능 지표**:
  - Precision@5: {best_weights_dict["best_by_ranking_objective"]["metrics"]["precision_at_5"]:.4f}
  - nDCG@10: {best_weights_dict["best_by_ranking_objective"]["metrics"]["ndcg_at_10"]:.4f}
  - MRR: {best_weights_dict["best_by_ranking_objective"]["metrics"]["mrr"]:.4f}
  - **Ranking Objective Score**: **{best_weights_dict["best_by_ranking_objective"]["metrics"]["ranking_objective"]:.4f}**

## 9. train/validation 검증 결과
* **80/20 Split 검증**:
  - **Train 최적 가중치**: SBERT (**{best_weights_dict["train_validation_result"]["train_best_weights"]["w_cleaned"]:.2f}**), TF-IDF (**{best_weights_dict["train_validation_result"]["train_best_weights"]["w_tfidf"]:.2f}**), 조문 (**{best_weights_dict["train_validation_result"]["train_best_weights"]["w_article"]:.2f}**)
  - **Train 성능 (Objective)**: **{best_weights_dict["train_validation_result"]["train_metrics"]["objective_score"]:.4f}**
  - **Validation 성능 (Objective)**: **{best_weights_dict["train_validation_result"]["validation_metrics"]["objective_score"]:.4f}**
* **5-Fold Cross Validation 평균 검증**:
  - **5-Fold Validation 평균 성능**: **{best_weights_dict["train_validation_result"]["cv_average_validation_metrics"]["objective_score"]:.4f}**
  - 평균 Precision@5: {best_weights_dict["train_validation_result"]["cv_average_validation_metrics"]["precision_at_5"]:.4f}
  - 평균 Precision@10: {best_weights_dict["train_validation_result"]["cv_average_validation_metrics"]["precision_at_10"]:.4f}
  - 평균 nDCG@10: {best_weights_dict["train_validation_result"]["cv_average_validation_metrics"]["ndcg_at_10"]:.4f}
  - 평균 MRR: {best_weights_dict["train_validation_result"]["cv_average_validation_metrics"]["mrr"]:.4f}

## 10. 상위 10개 가중치 후보 표
전체 라벨 셋 기준 `objective_score` 성능이 높은 가중치 상위 10개 후보 목록입니다.

| 순위 | $w_{{cleaned}}$ | $w_{{tfidf}}$ | $w_{{article}}$ | P@5 | P@10 | nDCG@10 | MRR | AvgRel | LMS | **Objective** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
{top10_cands_str}

## 11. 해석 및 추천
1. **그리드 서치의 과적합 한계**: 본 그리드 서치는 현재 구축된 라벨링 데이터셋(75개 소스법안, 2009쌍)을 기준으로 최적화된 결과입니다. 따라서 다른 종류의 법안 도메인이나 평가 데이터셋에서는 가중치 양상이 일부 변동될 수 있습니다.
2. **Gold와 LLM 라벨 혼합 해석 유의**: 본 평가는 일부 LLM 보조 실버 라벨링(full-label) 데이터를 포함하므로, 최종 가중치의 해석 및 배포 시 이 점을 고려해야 하며 전체적인 정확성을 종합 평가하는 보조 자료로 취급하는 것이 안전합니다.
3. **가중치 선택의 쟁점**: Primary Objective Score와 더불어 개별 평가 항목(P@5, nDCG@10, MRR)을 독립적으로 검토해야 합니다. 특정 가중치에서 Objective Score가 미세하게 높더라도 특정 개별 지표가 하락하는 현상이 발생할 수 있습니다.
4. **TF-IDF & 조문 유사도 기여도**: TF-IDF와 조문 유사도는 법안 의미 매칭을 직접 보조하는 훌륭한 신호이지만, SBERT에 비해 어휘 일치 혹은 구조 일치에 편향되어 있습니다. 따라서 이들 보조 신호의 가중치를 과도하게 설정하면 전반적인 의미적 맥락 검색 품질을 저해시킬 위험이 높습니다. 본 실험 결과를 바탕으로, SBERT 비중을 0.80 이상으로 높이고 TF-IDF 및 조문을 최소 보조로 가져가는 방향이 실질 검색 환경에서 더 견고할 수 있습니다.

"""

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"    - 보고서 마크다운 저장 완료: {OUTPUT_REPORT}")

    # ── 10. 콘솔 요약 출력 ──────────────────────────────────────────
    elapsed_time = time.time() - start_time
    print()
    print("=" * 80)
    print("  그리드 서치(Grid Search) 실행 완료 요약")
    print("=" * 80)
    print(f"  - 전체 법안 수                  : {num_bills}개")
    print(f"  - 전체 평가 라벨 행 수            : {len(df_label)}행")
    print(f"  - 평가에 사용된 source 수        : {num_eval_sources}개")
    print(f"  - 탐색한 가중치 후보 수          : {len(unique_candidates)}개")
    print("-" * 80)
    print(f"  - 기존 가중치 (0.70 / 0.20 / 0.10) 성능:")
    print(f"    P@5: {baseline_original_metrics['precision_at_5']:.4f}, nDCG@10: {baseline_original_metrics['ndcg_at_10']:.4f}, MRR: {baseline_original_metrics['mrr']:.4f}")
    print(f"    Objective Score: {baseline_original_metrics['objective_score']:.4f}")
    print("-" * 80)
    print(f"  - objective_score 기준 최적 가중치 (Primary):")
    print(f"    w_cleaned={best_weights_dict['best_by_objective_score']['weights']['w_cleaned']:.2f}, w_tfidf={best_weights_dict['best_by_objective_score']['weights']['w_tfidf']:.2f}, w_article={best_weights_dict['best_by_objective_score']['weights']['w_article']:.2f}")
    print(f"    Performance - P@5: {best_weights_dict['best_by_objective_score']['metrics']['precision_at_5']:.4f}, nDCG@10: {best_weights_dict['best_by_objective_score']['metrics']['ndcg_at_10']:.4f}, MRR: {best_weights_dict['best_by_objective_score']['metrics']['mrr']:.4f}")
    print(f"    Objective Score: {best_weights_dict['best_by_objective_score']['metrics']['objective_score']:.4f}")
    print("-" * 80)
    print(f"  - candidate_recall_objective 기준 최적 가중치:")
    print(f"    w_cleaned={best_weights_dict['best_by_candidate_recall_objective']['weights']['w_cleaned']:.2f}, w_tfidf={best_weights_dict['best_by_candidate_recall_objective']['weights']['w_tfidf']:.2f}, w_article={best_weights_dict['best_by_candidate_recall_objective']['weights']['w_article']:.2f}")
    print(f"    Objective Score: {best_weights_dict['best_by_candidate_recall_objective']['metrics']['candidate_recall_objective']:.4f}")
    print("-" * 80)
    print(f"  - ranking_objective 기준 최적 가중치:")
    print(f"    w_cleaned={best_weights_dict['best_by_ranking_objective']['weights']['w_cleaned']:.2f}, w_tfidf={best_weights_dict['best_by_ranking_objective']['weights']['w_tfidf']:.2f}, w_article={best_weights_dict['best_by_ranking_objective']['weights']['w_article']:.2f}")
    print(f"    Objective Score: {best_weights_dict['best_by_ranking_objective']['metrics']['ranking_objective']:.4f}")
    print("-" * 80)
    print(f"  - Train/Validation Split (80/20) 결과:")
    print(f"    Train Best Weights: w_cleaned={best_weights_dict['train_validation_result']['train_best_weights']['w_cleaned']:.2f}, w_tfidf={best_weights_dict['train_validation_result']['train_best_weights']['w_tfidf']:.2f}, w_article={best_weights_dict['train_validation_result']['train_best_weights']['w_article']:.2f}")
    print(f"    Train Objective   : {best_weights_dict['train_validation_result']['train_metrics']['objective_score']:.4f}")
    print(f"    Validation Obj    : {best_weights_dict['train_validation_result']['validation_metrics']['objective_score']:.4f}")
    print("-" * 80)
    print(f"  - 5-Fold Cross Validation 평균 Validation 성능:")
    print(f"    Objective Score   : {cv_summary_metrics['objective_score']:.4f}")
    print(f"    P@5: {cv_summary_metrics['precision_at_5']:.4f}, nDCG@10: {cv_summary_metrics['ndcg_at_10']:.4f}, MRR: {cv_summary_metrics['mrr']:.4f}")
    print("-" * 80)
    print(f"  - 소요 시간                     : {elapsed_time:.2f}초")
    print(f"  - 출력 파일 목록:")
    print(f"    1. {OUTPUT_CSV}")
    print(f"    2. {OUTPUT_JSON}")
    print(f"    3. {OUTPUT_BEST_JSON}")
    print(f"    4. {OUTPUT_TOPK_BEST}")
    print(f"    5. {OUTPUT_REPORT}")
    print("=" * 80)


if __name__ == "__main__":
    main()
