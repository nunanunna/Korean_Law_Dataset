#!/usr/bin/env python3
"""
24_grid_search_weighted_field_and_final_ensemble.py
===================================================
Weighted Field 및 Final Ensemble 가중치를 그리드 서치(grid search)하여
라벨링 데이터를 기준으로 최적 가중치를 찾고 성능을 분석합니다.
"""

import os
import sys
import math
import json
import random
import time
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 기존 모듈 임포트
from bill_text_parser import split_summary_sections, normalize_summary, extract_article_numbers
from clean_text_utils import normalize_legal_text_for_sbert, normalize_legal_text_for_keywords, extract_light_keywords
from article_similarity_utils import compute_article_similarity
from text_builders import build_raw_text, build_structured_text, build_problem_proposal_text

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 상수 및 경로 설정 ──────────────────────────────────────────────────
DATASET_JSON = "test_dataset/full_dataset.json"
LABEL_EXCEL = "Sbert_output/evaluation_full_score_pooled_llm_labeled.xlsx"
V2_LABEL_EXCEL = "Sbert_output/evaluation_pooled_label_template_v2.xlsx"

OUTPUT_CSV = "Sbert_output/weighted_and_ensemble_grid_search_results.csv"
OUTPUT_JSON = "Sbert_output/weighted_and_ensemble_grid_search_results.json"
OUTPUT_BEST_JSON = "Sbert_output/weighted_and_ensemble_best_weights.json"
OUTPUT_TOPK_WF_BEST = "Sbert_output/topk_weighted_field_grid_best.json"
OUTPUT_TOPK_FE_BEST = "Sbert_output/topk_final_ensemble_grid_best.json"
OUTPUT_REPORT = "Sbert_output/weighted_and_ensemble_grid_search_report.md"

RELEVANCE_THRESHOLD = 3.0

LEGAL_WEIGHTS = {
    "human_issue_match_0_to_2": 0.25,
    "human_target_match_0_to_2": 0.25,
    "human_effect_match_0_to_2": 0.25,
    "human_scope_match_0_to_2": 0.15,
    "human_article_match_0_to_2": 0.10
}

# 정규화 여부 옵션 (기본값 True)
NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD = True
NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE = True

# 수동 탐색 후보군 정의
WEIGHTED_FIELD_MANUAL_CANDIDATES = [
    {
        "name": "WF_original",
        "w_title": 0.00, "w_full": 0.20, "w_current": 0.05, "w_problem": 0.35, "w_proposal": 0.35, "w_article": 0.05
    },
    {
        "name": "WF_balanced_context",
        "w_title": 0.10, "w_full": 0.25, "w_current": 0.05, "w_problem": 0.30, "w_proposal": 0.25, "w_article": 0.05
    },
    {
        "name": "WF_raw_context_preserved",
        "w_title": 0.15, "w_full": 0.30, "w_current": 0.05, "w_problem": 0.25, "w_proposal": 0.20, "w_article": 0.05
    },
    {
        "name": "WF_problem_proposal_strong",
        "w_title": 0.10, "w_full": 0.10, "w_current": 0.00, "w_problem": 0.40, "w_proposal": 0.35, "w_article": 0.05
    },
    {
        "name": "WF_no_article",
        "w_title": 0.10, "w_full": 0.25, "w_current": 0.05, "w_problem": 0.30, "w_proposal": 0.30, "w_article": 0.00
    }
]

FINAL_ENSEMBLE_MANUAL_CANDIDATES = [
    {
        "name": "FE_balanced",
        "w_raw": 0.10, "w_structured": 0.25, "w_problem_proposal": 0.20, "w_weighted_field": 0.10, "w_cleaned": 0.10, "w_hybrid": 0.25
    },
    {
        "name": "FE_hybrid_p5",
        "w_raw": 0.05, "w_structured": 0.20, "w_problem_proposal": 0.20, "w_weighted_field": 0.10, "w_cleaned": 0.10, "w_hybrid": 0.35
    },
    {
        "name": "FE_structured_ranking",
        "w_raw": 0.05, "w_structured": 0.35, "w_problem_proposal": 0.20, "w_weighted_field": 0.10, "w_cleaned": 0.05, "w_hybrid": 0.25
    },
    {
        "name": "FE_context_preserved",
        "w_raw": 0.15, "w_structured": 0.25, "w_problem_proposal": 0.15, "w_weighted_field": 0.15, "w_cleaned": 0.05, "w_hybrid": 0.25
    },
    {
        "name": "FE_structured_hybrid_only",
        "w_raw": 0.10, "w_structured": 0.35, "w_problem_proposal": 0.10, "w_weighted_field": 0.00, "w_cleaned": 0.00, "w_hybrid": 0.45
    }
]


def safe_float(val, default=None):
    if val is None or (isinstance(val, str) and val.strip() == ""):
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def normalize_bill_name(name: str) -> str:
    cleaned = name.strip()
    cleaned = re.sub(r'(?:일부개정법률안|전부개정법률안|개정법률안|법률안|일부개정안|개정안|안)$', '', cleaned)
    return cleaned.strip()


def row_minmax_normalize(matrix, name="Matrix", ignore_diagonal=True):
    num_rows = matrix.shape[0]
    norm_matrix = np.zeros_like(matrix)
    
    before_vals = []
    for i in range(num_rows):
        for j in range(num_rows):
            if not ignore_diagonal or i != j:
                before_vals.append(matrix[i][j])
    before_min = np.min(before_vals) if before_vals else 0.0
    before_max = np.max(before_vals) if before_vals else 0.0
    before_mean = np.mean(before_vals) if before_vals else 0.0
    
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
                row_norm[i] = 1.0  # 자기 자신은 1.0으로 강제 유지
                
            row_norm = np.clip(row_norm, 0.0, 1.0)
            norm_matrix[i] = row_norm
        else:
            norm_matrix[i] = matrix[i]
            
    after_vals = []
    for i in range(num_rows):
        for j in range(num_rows):
            if not ignore_diagonal or i != j:
                after_vals.append(norm_matrix[i][j])
    after_min = np.min(after_vals) if after_vals else 0.0
    after_max = np.max(after_vals) if after_vals else 0.0
    after_mean = np.mean(after_vals) if after_vals else 0.0
    
    print(f"  [정규화 통계] {name}:")
    print(f"    - 정규화 전 (대각 제외) Min: {before_min:.4f}, Max: {before_max:.4f}, Mean: {before_mean:.4f}")
    print(f"    - 정규화 후 (대각 제외) Min: {after_min:.4f}, Max: {after_max:.4f}, Mean: {after_mean:.4f}")
    
    stats = {
        "before_min": float(before_min),
        "before_max": float(before_max),
        "before_mean": float(before_mean),
        "after_min": float(after_min),
        "after_max": float(after_max),
        "after_mean": float(after_mean)
    }
    return norm_matrix, stats


def compute_dcg(relevance_scores: list[float], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k]):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


def evaluate_topk(topk_data, label_map, filter_sources=None):
    source_groups = defaultdict(list)
    num_evaluated_pairs = 0
    num_unlabeled_pairs = 0
    
    for item in topk_data:
        s_id = item["source_bill_id"]
        t_id = item["target_bill_id"]
        rank = item["rank"]
        
        # 만약 특정 source만 필터링한다면 (Train/Val 또는 Gold-only)
        if filter_sources is not None and s_id not in filter_sources:
            continue
            
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
            
            # Legal Meaning Score 계산
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


def compute_objectives(m):
    p5 = m["precision_at_5"]
    p10 = m["precision_at_10"]
    ndcg10 = m["ndcg_at_10"]
    mrr = m["mrr"]
    avg_rel = m["average_relevance"]
    
    obj_score = 0.50 * p5 + 0.30 * ndcg10 + 0.20 * mrr
    cand_recall = 0.60 * p10 + 0.25 * p5 + 0.15 * avg_rel
    ranking = 0.45 * ndcg10 + 0.35 * mrr + 0.20 * p5
    
    return {
        "objective_score": round(obj_score, 4),
        "candidate_recall_objective": round(cand_recall, 4),
        "ranking_objective": round(ranking, 4)
    }


def extract_topk_pairs(matrix, bills, k=10, weights=None, method_name="method", comp_matrices=None, comp_names=None):
    num_bills = len(bills)
    topk_data = []
    for i in range(num_bills):
        source_id = bills[i]["bill_id"]
        source_name = bills[i]["bill_name"]
        
        scores = []
        for j in range(num_bills):
            if i == j:
                continue
            scores.append((j, matrix[i][j]))
            
        # Tie breaking: similarity desc, target_bill_id asc
        scores.sort(key=lambda x: (-x[1], bills[x[0]]["bill_id"]))
        
        top10 = scores[:k]
        for rank_idx, (t_idx, sim_val) in enumerate(top10):
            target_id = bills[t_idx]["bill_id"]
            target_name = bills[t_idx]["bill_name"]
            
            comp_scores = {}
            if comp_matrices is not None and comp_names is not None:
                for name, mat in zip(comp_names, comp_matrices):
                    comp_scores[name] = round(float(mat[i][t_idx]), 4)
            
            item = {
                "source_bill_id": source_id,
                "source_bill_name": source_name,
                "target_bill_id": target_id,
                "target_bill_name": target_name,
                "rank": rank_idx + 1,
                "similarity": round(float(sim_val), 6),
                "method": method_name
            }
            if comp_scores:
                item["component_scores"] = comp_scores
            if weights:
                item["weights"] = weights
            topk_data.append(item)
    return topk_data


def main():
    print("=" * 80)
    print("  Weighted Field & Final Ensemble Grid Search 최적화 실행")
    print("=" * 80)
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
    if not os.path.exists(LABEL_EXCEL):
        print(f"[ERROR] 라벨 파일이 없습니다: {LABEL_EXCEL}")
        sys.exit(1)

    print(f"[2] 라벨링 엑셀 파일 로드 중: {LABEL_EXCEL}")
    df_label = pd.read_excel(LABEL_EXCEL, dtype=str).fillna("")
    print(f"    - 전체 로드된 라벨 행 수: {len(df_label)}행")

    label_map = {}
    for idx, row in df_label.iterrows():
        s_id = row["source_bill_id"].strip()
        t_id = row["target_bill_id"].strip()
        rel_raw = row.get("human_relevance_0_to_4", "")
        
        rel = safe_float(rel_raw)
        if rel is None:
            continue
            
        if not (0.0 <= rel <= 4.0):
            raise ValueError(f"[ERROR] row {idx+2}의 relevance 값({rel})이 범위를 벗어났습니다.")
            
        issue = safe_float(row.get("human_issue_match_0_to_2"))
        target = safe_float(row.get("human_target_match_0_to_2"))
        effect = safe_float(row.get("human_effect_match_0_to_2"))
        scope = safe_float(row.get("human_scope_match_0_to_2"))
        article = safe_float(row.get("human_article_match_0_to_2"))
        
        label_map[(s_id, t_id)] = {
            "relevance": rel,
            "issue": issue,
            "target": target,
            "effect": effect,
            "scope": scope,
            "article": article
        }

    print(f"    - 유효하게 라벨링된 고유 쌍 수: {len(label_map)}개")

    # Gold 라벨 여부를 구분하기 위해 v2 템플릿 로드 검사
    gold_sources = set()
    if os.path.exists(V2_LABEL_EXCEL):
        df_v2 = pd.read_excel(V2_LABEL_EXCEL, dtype=str).fillna("")
        gold_sources = set(df_v2["source_bill_id"].unique())
        print(f"    - Gold label 전용 source 수 (v2): {len(gold_sources)}개")
    else:
        print("    - [안내] v2 템플릿 파일이 없으므로 Gold-only 구분 없이 혼합 라벨로만 진행합니다.")

    # ── 3. SBERT 모델 및 기본 텍스트 파싱 ──────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    print(f"\n[3] SBERT 모델 로드 중: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("    SBERT 모델 로드 완료!")

    # ── 4. 공통 / Weighted Field Components 텍스트 및 임베딩 생성 ──
    print("\n[4] Component Text 준비 및 유사도 행렬 생성...")
    
    # 영역 파싱
    all_sections = []
    for idx, bill in enumerate(bills):
        summary = bill.get("summary", "")
        sections = split_summary_sections(summary)
        all_sections.append(sections)

    # A. SBERT 공통 / Weighted Field 입력 텍스트 리스트
    raw_texts = []
    structured_texts = []
    problem_proposal_texts = []
    cleaned_texts = []
    
    wf_title_texts = []
    wf_full_texts = []
    wf_current_texts = []
    wf_problem_texts = []
    wf_proposal_texts = []

    for idx, bill in enumerate(bills):
        summary = bill.get("summary", "")
        sections = all_sections[idx]
        
        # 1. raw
        raw_texts.append(build_raw_text(bill))
        # 2. structured
        structured_texts.append(build_structured_text(bill, sections))
        # 3. problem_proposal
        problem_proposal_texts.append(build_problem_proposal_text(bill, sections))
        # 4. cleaned_problem_proposal
        prob = sections.get("problem", "").strip()
        prop = sections.get("proposal", "").strip()
        combined = f"{prob} {prop}".strip()
        cleaned = normalize_legal_text_for_sbert(combined)
        if len(cleaned) < 5:
            fallback = f"{bill.get('bill_name', '')} {summary}".strip()
            cleaned = normalize_legal_text_for_sbert(fallback)
        cleaned_texts.append(cleaned)
        
        # 5. Weighted Field: title (normalize_bill_name)
        wf_title_texts.append(normalize_bill_name(bill.get("bill_name", "")))
        # 6. Weighted Field: full_text (bill_name + summary)
        wf_full_texts.append(f"{bill.get('bill_name', '')} {summary}".strip())
        # 7. Weighted Field: current_law
        cur_text = sections.get("current_law", "").strip()
        wf_current_texts.append(cur_text if cur_text else "[내용 없음]")
        # 8. Weighted Field: problem
        prob_text = sections.get("problem", "").strip()
        wf_problem_texts.append(prob_text if prob_text else "[내용 없음]")
        # 9. Weighted Field: proposal
        prop_text = sections.get("proposal", "").strip()
        wf_proposal_texts.append(prop_text if prop_text else "[내용 없음]")

    # B. SBERT 임베딩 인코딩 및 코사인 유사도 연산
    def encode_and_get_similarity(text_list, desc):
        print(f"    - {desc} 인코딩 및 코사인 유사도 계산 중...")
        embs = model.encode(text_list, batch_size=32, show_progress_bar=False,
                            normalize_embeddings=True, convert_to_tensor=True)
        sim = embs @ embs.T
        return sim.cpu().numpy()

    raw_sim = encode_and_get_similarity(raw_texts, "Raw Text")
    structured_sim = encode_and_get_similarity(structured_texts, "Structured Text")
    problem_proposal_sim = encode_and_get_similarity(problem_proposal_texts, "Problem Proposal Text")
    cleaned_problem_proposal_sim = encode_and_get_similarity(cleaned_texts, "Cleaned Problem Proposal SBERT")

    title_sim = encode_and_get_similarity(wf_title_texts, "WF Title Law Name")
    full_sim = encode_and_get_similarity(wf_full_texts, "WF Full Text")
    current_sim = encode_and_get_similarity(wf_current_texts, "WF Current Law")
    problem_sim = encode_and_get_similarity(wf_problem_texts, "WF Problem")
    proposal_sim = encode_and_get_similarity(wf_proposal_texts, "WF Proposal")

    # C. TF-IDF & Article Jaccard 유사도
    print("    - TF-IDF 키워드 유사도 계산 중...")
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

    print("    - Article Jaccard 유사도 계산 중...")
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

    # ── 5. Component Score Normalization ──────────────────────────────
    print("\n[5] Component Score Normalization 적용...")
    normalization_stats = {}

    def get_norm(matrix, name, flag):
        if flag:
            norm_mat, stats = row_minmax_normalize(matrix, name, ignore_diagonal=True)
            normalization_stats[name] = stats
            return norm_mat
        else:
            print(f"    - [안내] {name} 정규화 비적용 (원시 점수 사용)")
            return matrix.copy()

    # TF-IDF 및 Article 하이브리드 최적화를 위한 정규화 (hybrid_cleaned_best용)
    # 기존 grid search 결과 hybrid_cleaned_best = 0.9*cleaned + 0.1*tfidf_norm + 0.0*article_sim 이었음
    tfidf_sim_norm_hybrid, _ = row_minmax_normalize(tfidf_sim, "Hybrid Keyword TF-IDF", ignore_diagonal=True)
    hybrid_cleaned_best_sim = 0.90 * cleaned_problem_proposal_sim + 0.10 * tfidf_sim_norm_hybrid + 0.00 * article_sim
    hybrid_cleaned_original_sim = 0.70 * cleaned_problem_proposal_sim + 0.20 * tfidf_sim_norm_hybrid + 0.10 * article_sim

    # Weighted Field Components 정규화
    title_sim_wf = get_norm(title_sim, "WF Title Law Name", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)
    full_sim_wf = get_norm(full_sim, "WF Full Text", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)
    current_sim_wf = get_norm(current_sim, "WF Current Law", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)
    problem_sim_wf = get_norm(problem_sim, "WF Problem", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)
    proposal_sim_wf = get_norm(proposal_sim, "WF Proposal", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)
    article_sim_wf = get_norm(article_sim, "WF Article Jaccard", NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD)

    # ── 6. Weighted Field Grid Search ─────────────────────────────────
    print("\n[6] Weighted Field Grid Search 시작...")
    
    # 0.05 step grid search candidates 생성
    wf_candidates = []
    # w_title: 0.00 ~ 0.20, w_full: 0.10 ~ 0.35, w_current: 0.00 ~ 0.10, w_problem: 0.20 ~ 0.45, w_proposal: 0.20 ~ 0.45, w_article: 0.00 ~ 0.10
    for w_title in np.arange(0.00, 0.20 + 1e-9, 0.05):
        for w_full in np.arange(0.10, 0.35 + 1e-9, 0.05):
            for w_current in np.arange(0.00, 0.10 + 1e-9, 0.05):
                for w_problem in np.arange(0.20, 0.45 + 1e-9, 0.05):
                    for w_proposal in np.arange(0.20, 0.45 + 1e-9, 0.05):
                        for w_article in np.arange(0.00, 0.10 + 1e-9, 0.05):
                            w_t = round(float(w_title), 2)
                            w_fu = round(float(w_full), 2)
                            w_c = round(float(w_current), 2)
                            w_pr = round(float(w_problem), 2)
                            w_po = round(float(w_proposal), 2)
                            w_a = round(float(w_article), 2)
                            if abs(w_t + w_fu + w_c + w_pr + w_po + w_a - 1.0) < 1e-9:
                                wf_candidates.append({
                                    "w_title": w_t, "w_full": w_fu, "w_current": w_c,
                                    "w_problem": w_pr, "w_proposal": w_po, "w_article": w_a
                                })

    # 수동 후보 추가 및 중복 제거
    for mc in WEIGHTED_FIELD_MANUAL_CANDIDATES:
        wf_candidates.append({
            "w_title": mc["w_title"], "w_full": mc["w_full"], "w_current": mc["w_current"],
            "w_problem": mc["w_problem"], "w_proposal": mc["w_proposal"], "w_article": mc["w_article"]
        })

    unique_wf_candidates = []
    seen_wf = set()
    for c in wf_candidates:
        tup = (c["w_title"], c["w_full"], c["w_current"], c["w_problem"], c["w_proposal"], c["w_article"])
        if tup not in seen_wf:
            assert abs(sum(tup) - 1.0) < 1e-9, f"가중치 합이 1.0이 아닙니다: {c}"
            seen_wf.add(tup)
            unique_wf_candidates.append(c)

    print(f"    - Weighted Field 총 고유 가중치 조합 수: {len(unique_wf_candidates)}개")

    evaluated_source_ids = sorted(list(set(k[0] for k in label_map.keys())))
    
    wf_results = []
    best_wf_by_obj = None
    best_wf_by_recall = None
    best_wf_by_rank = None
    
    best_wf_score_obj = -1.0
    best_wf_score_recall = -1.0
    best_wf_score_rank = -1.0

    for c_idx, cand in enumerate(unique_wf_candidates):
        w_t = cand["w_title"]
        w_fu = cand["w_full"]
        w_c = cand["w_current"]
        w_pr = cand["w_problem"]
        w_po = cand["w_proposal"]
        w_a = cand["w_article"]

        # 가중치 행렬 결합
        combined_wf_sim = (w_t * title_sim_wf + w_fu * full_sim_wf + w_c * current_sim_wf +
                           w_pr * problem_sim_wf + w_po * proposal_sim_wf + w_a * article_sim_wf)

        # Top-10 target 추출
        topk_data = extract_topk_pairs(combined_wf_sim, bills, k=10, weights=cand, method_name="weighted_field_candidate")
        metrics = evaluate_topk(topk_data, label_map)
        objs = compute_objectives(metrics)
        
        res = {**cand, **metrics, **objs}
        wf_results.append(res)

        # Best 가중치 트래킹
        if objs["objective_score"] > best_wf_score_obj:
            best_wf_score_obj = objs["objective_score"]
            best_wf_by_obj = res
        if objs["candidate_recall_objective"] > best_wf_score_recall:
            best_wf_score_recall = objs["candidate_recall_objective"]
            best_wf_by_recall = res
        if objs["ranking_objective"] > best_wf_score_rank:
            best_wf_score_rank = objs["ranking_objective"]
            best_wf_by_rank = res

    print(f"    - Weighted Field 최적 탐색 완료!")
    print(f"      * Best Objective Score: {best_wf_score_obj:.4f} (Weights: {best_wf_by_obj})")

    # Weighted Field Best Matrix 고정 (Final Ensemble용)
    w_t_best = best_wf_by_obj["w_title"]
    w_fu_best = best_wf_by_obj["w_full"]
    w_c_best = best_wf_by_obj["w_current"]
    w_pr_best = best_wf_by_obj["w_problem"]
    w_po_best = best_wf_by_obj["w_proposal"]
    w_a_best = best_wf_by_obj["w_article"]

    weighted_field_best_sim = (w_t_best * title_sim_wf + w_fu_best * full_sim_wf + w_c_best * current_sim_wf +
                               w_pr_best * problem_sim_wf + w_po_best * proposal_sim_wf + w_a_best * article_sim_wf)

    # ── 7. Final Ensemble Components Normalization ────────────────────
    # Final Ensemble에 사용될 6개 matrix 정규화
    raw_sim_fe = get_norm(raw_sim, "FE Raw Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)
    structured_sim_fe = get_norm(structured_sim, "FE Structured Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)
    problem_proposal_sim_fe = get_norm(problem_proposal_sim, "FE Problem Proposal Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)
    weighted_field_best_sim_fe = get_norm(weighted_field_best_sim, "FE Weighted Field Best Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)
    cleaned_problem_proposal_sim_fe = get_norm(cleaned_problem_proposal_sim, "FE Cleaned Problem Proposal Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)
    hybrid_cleaned_best_sim_fe = get_norm(hybrid_cleaned_best_sim, "FE Hybrid Cleaned Best Score", NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE)

    # ── 8. Final Ensemble Grid Search ─────────────────────────────────
    print("\n[8] Final Ensemble Grid Search 시작...")
    
    # w_raw: 0.00 ~ 0.25, w_structured: 0.10 ~ 0.40, w_problem_proposal: 0.10 ~ 0.30, w_weighted_field: 0.00 ~ 0.20, w_cleaned: 0.00 ~ 0.20, w_hybrid: 0.15 ~ 0.45
    fe_candidates = []
    for w_raw in np.arange(0.00, 0.25 + 1e-9, 0.05):
        for w_structured in np.arange(0.10, 0.40 + 1e-9, 0.05):
            for w_problem_proposal in np.arange(0.10, 0.30 + 1e-9, 0.05):
                for w_weighted_field in np.arange(0.00, 0.20 + 1e-9, 0.05):
                    for w_cleaned in np.arange(0.00, 0.20 + 1e-9, 0.05):
                        for w_hybrid in np.arange(0.15, 0.45 + 1e-9, 0.05):
                            w_r = round(float(w_raw), 2)
                            w_s = round(float(w_structured), 2)
                            w_pp = round(float(w_problem_proposal), 2)
                            w_wf = round(float(w_weighted_field), 2)
                            w_cl = round(float(w_cleaned), 2)
                            w_hy = round(float(w_hybrid), 2)
                            if abs(w_r + w_s + w_pp + w_wf + w_cl + w_hy - 1.0) < 1e-9:
                                fe_candidates.append({
                                    "w_raw": w_r, "w_structured": w_s, "w_problem_proposal": w_pp,
                                    "w_weighted_field": w_wf, "w_cleaned": w_cl, "w_hybrid": w_hy
                                })

    # 수동 후보 추가
    for mc in FINAL_ENSEMBLE_MANUAL_CANDIDATES:
        fe_candidates.append({
            "w_raw": mc["w_raw"], "w_structured": mc["w_structured"], "w_problem_proposal": mc["w_problem_proposal"],
            "w_weighted_field": mc["w_weighted_field"], "w_cleaned": mc["w_cleaned"], "w_hybrid": mc["w_hybrid"]
        })

    unique_fe_candidates = []
    seen_fe = set()
    for c in fe_candidates:
        tup = (c["w_raw"], c["w_structured"], c["w_problem_proposal"], c["w_weighted_field"], c["w_cleaned"], c["w_hybrid"])
        if tup not in seen_fe:
            assert abs(sum(tup) - 1.0) < 1e-9, f"가중치 합이 1.0이 아닙니다: {c}"
            seen_fe.add(tup)
            unique_fe_candidates.append(c)

    print(f"    - Final Ensemble 총 고유 가중치 조합 수: {len(unique_fe_candidates)}개")

    fe_results = []
    best_fe_by_obj = None
    best_fe_by_recall = None
    best_fe_by_rank = None
    
    best_fe_score_obj = -1.0
    best_fe_score_recall = -1.0
    best_fe_score_rank = -1.0

    for c_idx, cand in enumerate(unique_fe_candidates):
        w_r = cand["w_raw"]
        w_s = cand["w_structured"]
        w_pp = cand["w_problem_proposal"]
        w_wf = cand["w_weighted_field"]
        w_cl = cand["w_cleaned"]
        w_hy = cand["w_hybrid"]

        combined_fe_sim = (w_r * raw_sim_fe + w_s * structured_sim_fe + w_pp * problem_proposal_sim_fe +
                           w_wf * weighted_field_best_sim_fe + w_cl * cleaned_problem_proposal_sim_fe +
                           w_hy * hybrid_cleaned_best_sim_fe)

        topk_data = extract_topk_pairs(combined_fe_sim, bills, k=10, weights=cand, method_name="final_ensemble_candidate")
        metrics = evaluate_topk(topk_data, label_map)
        objs = compute_objectives(metrics)

        res = {**cand, **metrics, **objs}
        fe_results.append(res)

        if objs["objective_score"] > best_fe_score_obj:
            best_fe_score_obj = objs["objective_score"]
            best_fe_by_obj = res
        if objs["candidate_recall_objective"] > best_fe_score_recall:
            best_fe_score_recall = objs["candidate_recall_objective"]
            best_fe_by_recall = res
        if objs["ranking_objective"] > best_fe_score_rank:
            best_fe_score_rank = objs["ranking_objective"]
            best_fe_by_rank = res

    print(f"    - Final Ensemble 최적 탐색 완료!")
    print(f"      * Best Objective Score: {best_fe_score_obj:.4f} (Weights: {best_fe_by_obj})")

    # ── 9. Train / Validation split (80/20) 및 5-Fold Cross Validation ─
    print("\n[9] Cross Validation (Train/Val 분할) 수행 중...")
    
    # Source Bill 수준 5-Fold 분할
    random.seed(42)
    shuffled_sources = evaluated_source_ids.copy()
    random.shuffle(shuffled_sources)
    
    num_sources = len(shuffled_sources)
    k_folds = 5
    fold_size = num_sources // k_folds
    
    cv_wf_val_scores = []
    cv_fe_val_scores = []

    # 80/20 train/val 결과 저장용 (Fold 0 기준)
    fold0_val_wf_metrics = None
    fold0_val_fe_metrics = None
    fold0_train_wf_weights = None
    fold0_train_fe_weights = None

    for fold in range(k_folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold != k_folds - 1 else num_sources
        
        val_sources = set(shuffled_sources[val_start:val_end])
        train_sources = set(shuffled_sources) - val_sources
        
        # A. Weighted Field Fold Optimization
        best_wf_fold_obj = -1.0
        best_wf_fold_cand = None
        for cand in unique_wf_candidates:
            w_t = cand["w_title"]
            w_fu = cand["w_full"]
            w_c = cand["w_current"]
            w_pr = cand["w_problem"]
            w_po = cand["w_proposal"]
            w_a = cand["w_article"]
            
            combined_wf_sim = (w_t * title_sim_wf + w_fu * full_sim_wf + w_c * current_sim_wf +
                               w_pr * problem_sim_wf + w_po * proposal_sim_wf + w_a * article_sim_wf)
            
            topk_data = extract_topk_pairs(combined_wf_sim, bills, k=10)
            metrics_tr = evaluate_topk(topk_data, label_map, filter_sources=train_sources)
            objs_tr = compute_objectives(metrics_tr)
            
            if objs_tr["objective_score"] > best_wf_fold_obj:
                best_wf_fold_obj = objs_tr["objective_score"]
                best_wf_fold_cand = cand
                
        # Val Evaluation for WF
        w_t = best_wf_fold_cand["w_title"]
        w_fu = best_wf_fold_cand["w_full"]
        w_c = best_wf_fold_cand["w_current"]
        w_pr = best_wf_fold_cand["w_problem"]
        w_po = best_wf_fold_cand["w_proposal"]
        w_a = best_wf_fold_cand["w_article"]
        combined_wf_sim = (w_t * title_sim_wf + w_fu * full_sim_wf + w_c * current_sim_wf +
                           w_pr * problem_sim_wf + w_po * proposal_sim_wf + w_a * article_sim_wf)
        topk_data_val = extract_topk_pairs(combined_wf_sim, bills, k=10)
        metrics_val = evaluate_topk(topk_data_val, label_map, filter_sources=val_sources)
        objs_val = compute_objectives(metrics_val)
        cv_wf_val_scores.append(objs_val["objective_score"])

        # B. Final Ensemble Fold Optimization
        best_fe_fold_obj = -1.0
        best_fe_fold_cand = None
        for cand in unique_fe_candidates:
            w_r = cand["w_raw"]
            w_s = cand["w_structured"]
            w_pp = cand["w_problem_proposal"]
            w_wf = cand["w_weighted_field"]
            w_cl = cand["w_cleaned"]
            w_hy = cand["w_hybrid"]
            
            combined_fe_sim = (w_r * raw_sim_fe + w_s * structured_sim_fe + w_pp * problem_proposal_sim_fe +
                               w_wf * weighted_field_best_sim_fe + w_cl * cleaned_problem_proposal_sim_fe +
                               w_hy * hybrid_cleaned_best_sim_fe)
            
            topk_data = extract_topk_pairs(combined_fe_sim, bills, k=10)
            metrics_tr = evaluate_topk(topk_data, label_map, filter_sources=train_sources)
            objs_tr = compute_objectives(metrics_tr)
            
            if objs_tr["objective_score"] > best_fe_fold_obj:
                best_fe_fold_obj = objs_tr["objective_score"]
                best_fe_fold_cand = cand
                
        # Val Evaluation for FE
        w_r = best_fe_fold_cand["w_raw"]
        w_s = best_fe_fold_cand["w_structured"]
        w_pp = best_fe_fold_cand["w_problem_proposal"]
        w_wf = best_fe_fold_cand["w_weighted_field"]
        w_cl = best_fe_fold_cand["w_cleaned"]
        w_hy = best_fe_fold_cand["w_hybrid"]
        combined_fe_sim = (w_r * raw_sim_fe + w_s * structured_sim_fe + w_pp * problem_proposal_sim_fe +
                           w_wf * weighted_field_best_sim_fe + w_cl * cleaned_problem_proposal_sim_fe +
                           w_hy * hybrid_cleaned_best_sim_fe)
        topk_data_val_fe = extract_topk_pairs(combined_fe_sim, bills, k=10)
        metrics_val_fe = evaluate_topk(topk_data_val_fe, label_map, filter_sources=val_sources)
        objs_val_fe = compute_objectives(metrics_val_fe)
        cv_fe_val_scores.append(objs_val_fe["objective_score"])

        # 80/20 train/val split (Fold 0) 상세 저장
        if fold == 0:
            fold0_val_wf_metrics = {**metrics_val, **objs_val}
            fold0_val_fe_metrics = {**metrics_val_fe, **objs_val_fe}
            fold0_train_wf_weights = best_wf_fold_cand
            fold0_train_fe_weights = best_fe_fold_cand

    cv_wf_mean_val = np.mean(cv_wf_val_scores)
    cv_fe_mean_val = np.mean(cv_fe_val_scores)

    print(f"    - 5-Fold CV 완료:")
    print(f"      * Weighted Field 평균 Validation Objective Score: {cv_wf_mean_val:.4f}")
    print(f"      * Final Ensemble 평균 Validation Objective Score: {cv_fe_mean_val:.4f}")

    # ── 10. Baseline 및 기존 모델 평가 ─────────────────────────────────
    print("\n[10] Baseline 모델들 평가 수행...")
    
    baselines = {}
    
    # 1. raw
    raw_topk = extract_topk_pairs(raw_sim, bills, k=10, method_name="raw")
    baselines["raw"] = {**evaluate_topk(raw_topk, label_map), **compute_objectives(evaluate_topk(raw_topk, label_map))}
    
    # 2. structured
    struct_topk = extract_topk_pairs(structured_sim, bills, k=10, method_name="structured")
    baselines["structured"] = {**evaluate_topk(struct_topk, label_map), **compute_objectives(evaluate_topk(struct_topk, label_map))}
    
    # 3. problem_proposal
    pp_topk = extract_topk_pairs(problem_proposal_sim, bills, k=10, method_name="problem_proposal")
    baselines["problem_proposal"] = {**evaluate_topk(pp_topk, label_map), **compute_objectives(evaluate_topk(pp_topk, label_map))}
    
    # 4. cleaned_problem_proposal
    cleaned_topk = extract_topk_pairs(cleaned_problem_proposal_sim, bills, k=10, method_name="cleaned_problem_proposal")
    baselines["cleaned_problem_proposal"] = {**evaluate_topk(cleaned_topk, label_map), **compute_objectives(evaluate_topk(cleaned_topk, label_map))}
    
    # 5. hybrid_cleaned_original
    hybrid_orig_topk = extract_topk_pairs(hybrid_cleaned_original_sim, bills, k=10, method_name="hybrid_cleaned_original")
    baselines["hybrid_cleaned_original"] = {**evaluate_topk(hybrid_orig_topk, label_map), **compute_objectives(evaluate_topk(hybrid_orig_topk, label_map))}
    
    # 6. hybrid_cleaned_best
    hybrid_best_topk = extract_topk_pairs(hybrid_cleaned_best_sim, bills, k=10, method_name="hybrid_cleaned_best")
    baselines["hybrid_cleaned_best"] = {**evaluate_topk(hybrid_best_topk, label_map), **compute_objectives(evaluate_topk(hybrid_best_topk, label_map))}

    # 7. weighted_field_original (수동 original 가중치 WF_original 적용)
    w_t_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_title"]
    w_fu_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_full"]
    w_c_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_current"]
    w_pr_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_problem"]
    w_po_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_proposal"]
    w_a_orig = WEIGHTED_FIELD_MANUAL_CANDIDATES[0]["w_article"]
    wf_orig_sim = (w_t_orig * title_sim_wf + w_fu_orig * full_sim_wf + w_c_orig * current_sim_wf +
                   w_pr_orig * problem_sim_wf + w_po_orig * proposal_sim_wf + w_a_orig * article_sim_wf)
    wf_orig_topk = extract_topk_pairs(wf_orig_sim, bills, k=10, weights=WEIGHTED_FIELD_MANUAL_CANDIDATES[0], method_name="weighted_field_original")
    baselines["weighted_field_original"] = {**evaluate_topk(wf_orig_topk, label_map), **compute_objectives(evaluate_topk(wf_orig_topk, label_map))}

    # ── 11. Gold-only에 대한 성능 리포팅 (v2 20개 소스 기준) ──────────────
    gold_report_data = {}
    if gold_sources:
        print("\n[11] Gold-only (20개 소스 법안) 기준 추가 성능 리포팅 계산 중...")
        for name, bl_m in baselines.items():
            topk_f = None
            if name == "raw": topk_f = raw_topk
            elif name == "structured": topk_f = struct_topk
            elif name == "problem_proposal": topk_f = pp_topk
            elif name == "cleaned_problem_proposal": topk_f = cleaned_topk
            elif name == "hybrid_cleaned_original": topk_f = hybrid_orig_topk
            elif name == "hybrid_cleaned_best": topk_f = hybrid_best_topk
            elif name == "weighted_field_original": topk_f = wf_orig_topk
            
            if topk_f:
                g_met = evaluate_topk(topk_f, label_map, filter_sources=gold_sources)
                g_obj = compute_objectives(g_met)
                gold_report_data[name] = {**g_met, **g_obj}

        # Weighted Field Best 및 Final Ensemble Best도 Gold-only 적용
        wf_best_topk = extract_topk_pairs(weighted_field_best_sim, bills, k=10)
        g_wf_met = evaluate_topk(wf_best_topk, label_map, filter_sources=gold_sources)
        g_wf_obj = compute_objectives(g_wf_met)
        gold_report_data["weighted_field_best"] = {**g_wf_met, **g_wf_obj}

        w_r = best_fe_by_obj["w_raw"]
        w_s = best_fe_by_obj["w_structured"]
        w_pp = best_fe_by_obj["w_problem_proposal"]
        w_wf = best_fe_by_obj["w_weighted_field"]
        w_cl = best_fe_by_obj["w_cleaned"]
        w_hy = best_fe_by_obj["w_hybrid"]
        combined_fe_sim_best = (w_r * raw_sim_fe + w_s * structured_sim_fe + w_pp * problem_proposal_sim_fe +
                                w_wf * weighted_field_best_sim_fe + w_cl * cleaned_problem_proposal_sim_fe +
                                w_hy * hybrid_cleaned_best_sim_fe)
        fe_best_topk = extract_topk_pairs(combined_fe_sim_best, bills, k=10)
        g_fe_met = evaluate_topk(fe_best_topk, label_map, filter_sources=gold_sources)
        g_fe_obj = compute_objectives(g_fe_met)
        gold_report_data["final_ensemble_best"] = {**g_fe_met, **g_fe_obj}

    # ── 12. 결과 출력 및 파일 저장 ────────────────────────────────────
    print("\n[12] 결과 저장 및 파일 작성 시작...")
    
    # A. JSON / CSV 결과 테이블 구성
    results_all = []
    # Weighted Field 결과들
    for r in wf_results:
        results_all.append({
            "type": "weighted_field",
            **r
        })
    # Final Ensemble 결과들
    for r in fe_results:
        results_all.append({
            "type": "final_ensemble",
            **r
        })
        
    df_res = pd.DataFrame(results_all)
    df_res.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_all, f, ensure_ascii=False, indent=4)

    # B. Best 가중치 상세 정보를 저장할 JSON
    best_weights_json = {
        "weighted_field": {
            "best_by_objective_score": {
                "weights": {
                    "w_title": best_wf_by_obj["w_title"],
                    "w_full": best_wf_by_obj["w_full"],
                    "w_current": best_wf_by_obj["w_current"],
                    "w_problem": best_wf_by_obj["w_problem"],
                    "w_proposal": best_wf_by_obj["w_proposal"],
                    "w_article": best_wf_by_obj["w_article"]
                },
                "metrics": {k: best_wf_by_obj[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "objective_score"]}
            },
            "best_by_candidate_recall_objective": {
                "weights": {
                    "w_title": best_wf_by_recall["w_title"],
                    "w_full": best_wf_by_recall["w_full"],
                    "w_current": best_wf_by_recall["w_current"],
                    "w_problem": best_wf_by_recall["w_problem"],
                    "w_proposal": best_wf_by_recall["w_proposal"],
                    "w_article": best_wf_by_recall["w_article"]
                },
                "metrics": {k: best_wf_by_recall[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "candidate_recall_objective"]}
            },
            "best_by_ranking_objective": {
                "weights": {
                    "w_title": best_wf_by_rank["w_title"],
                    "w_full": best_wf_by_rank["w_full"],
                    "w_current": best_wf_by_rank["w_current"],
                    "w_problem": best_wf_by_rank["w_problem"],
                    "w_proposal": best_wf_by_rank["w_proposal"],
                    "w_article": best_wf_by_rank["w_article"]
                },
                "metrics": {k: best_wf_by_rank[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "ranking_objective"]}
            },
            "train_validation_result": {
                "fold0_train_best_weights": fold0_train_wf_weights,
                "fold0_val_metrics": fold0_val_wf_metrics,
                "cv_average_validation_objective_score": round(cv_wf_mean_val, 4)
            }
        },
        "final_ensemble": {
            "best_by_objective_score": {
                "weights": {
                    "w_raw": best_fe_by_obj["w_raw"],
                    "w_structured": best_fe_by_obj["w_structured"],
                    "w_problem_proposal": best_fe_by_obj["w_problem_proposal"],
                    "w_weighted_field": best_fe_by_obj["w_weighted_field"],
                    "w_cleaned": best_fe_by_obj["w_cleaned"],
                    "w_hybrid": best_fe_by_obj["w_hybrid"]
                },
                "metrics": {k: best_fe_by_obj[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "objective_score"]}
            },
            "best_by_candidate_recall_objective": {
                "weights": {
                    "w_raw": best_fe_by_recall["w_raw"],
                    "w_structured": best_fe_by_recall["w_structured"],
                    "w_problem_proposal": best_fe_by_recall["w_problem_proposal"],
                    "w_weighted_field": best_fe_by_recall["w_weighted_field"],
                    "w_cleaned": best_fe_by_recall["w_cleaned"],
                    "w_hybrid": best_fe_by_recall["w_hybrid"]
                },
                "metrics": {k: best_fe_by_recall[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "candidate_recall_objective"]}
            },
            "best_by_ranking_objective": {
                "weights": {
                    "w_raw": best_fe_by_rank["w_raw"],
                    "w_structured": best_fe_by_rank["w_structured"],
                    "w_problem_proposal": best_fe_by_rank["w_problem_proposal"],
                    "w_weighted_field": best_fe_by_rank["w_weighted_field"],
                    "w_cleaned": best_fe_by_rank["w_cleaned"],
                    "w_hybrid": best_fe_by_rank["w_hybrid"]
                },
                "metrics": {k: best_fe_by_rank[k] for k in ["precision_at_5", "precision_at_10", "ndcg_at_10", "mrr", "average_relevance", "average_legal_meaning_score", "ranking_objective"]}
            },
            "train_validation_result": {
                "fold0_train_best_weights": fold0_train_fe_weights,
                "fold0_val_metrics": fold0_val_fe_metrics,
                "cv_average_validation_objective_score": round(cv_fe_mean_val, 4)
            }
        },
        "normalization_applied": {
            "weighted_field": NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD,
            "final_ensemble": NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE,
            "normalization_stats": normalization_stats
        }
    }
    
    with open(OUTPUT_BEST_JSON, "w", encoding="utf-8") as f:
        json.dump(best_weights_json, f, ensure_ascii=False, indent=4)

    # C. Top-K Best JSON 파일 생성 및 저장
    # 1) Weighted Field Best
    wf_best_topk_export = extract_topk_pairs(
        weighted_field_best_sim, bills, k=10, 
        weights={
            "w_title": best_wf_by_obj["w_title"],
            "w_full": best_wf_by_obj["w_full"],
            "w_current": best_wf_by_obj["w_current"],
            "w_problem": best_wf_by_obj["w_problem"],
            "w_proposal": best_wf_by_obj["w_proposal"],
            "w_article": best_wf_by_obj["w_article"]
        }, 
        method_name="weighted_field_grid_best",
        comp_matrices=[title_sim_wf, full_sim_wf, current_sim_wf, problem_sim_wf, proposal_sim_wf, article_sim_wf],
        comp_names=["title_law_name_similarity", "full_text_similarity", "current_law_similarity", "problem_similarity", "proposal_similarity", "article_similarity"]
    )
    with open(OUTPUT_TOPK_WF_BEST, "w", encoding="utf-8") as f:
        json.dump(wf_best_topk_export, f, ensure_ascii=False, indent=4)

    # 2) Final Ensemble Best
    w_r = best_fe_by_obj["w_raw"]
    w_s = best_fe_by_obj["w_structured"]
    w_pp = best_fe_by_obj["w_problem_proposal"]
    w_wf = best_fe_by_obj["w_weighted_field"]
    w_cl = best_fe_by_obj["w_cleaned"]
    w_hy = best_fe_by_obj["w_hybrid"]
    combined_fe_sim_best = (w_r * raw_sim_fe + w_s * structured_sim_fe + w_pp * problem_proposal_sim_fe +
                            w_wf * weighted_field_best_sim_fe + w_cl * cleaned_problem_proposal_sim_fe +
                            w_hy * hybrid_cleaned_best_sim_fe)
    fe_best_topk_export = extract_topk_pairs(
        combined_fe_sim_best, bills, k=10,
        weights={
            "w_raw": w_r,
            "w_structured": w_s,
            "w_problem_proposal": w_pp,
            "w_weighted_field": w_wf,
            "w_cleaned": w_cl,
            "w_hybrid": w_hy
        },
        method_name="final_ensemble_grid_best",
        comp_matrices=[raw_sim_fe, structured_sim_fe, problem_proposal_sim_fe, weighted_field_best_sim_fe, cleaned_problem_proposal_sim_fe, hybrid_cleaned_best_sim_fe],
        comp_names=["raw_score", "structured_score", "problem_proposal_score", "weighted_field_best_score", "cleaned_problem_proposal_score", "hybrid_cleaned_best_score"]
    )
    with open(OUTPUT_TOPK_FE_BEST, "w", encoding="utf-8") as f:
        json.dump(fe_best_topk_export, f, ensure_ascii=False, indent=4)

    # D. Markdown 리포트 작성
    print("    - Markdown 보고서 생성 중...")
    
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("# 법률발의안 유사도 알고리즘 가중치 최적화 결과 보고서\n\n")
        f.write("> **평가 기준:** gold/LLM 혼합 라벨 기준 (75개 법안 전체셋)\n")
        if gold_sources:
            f.write("> **추가 비교:** Gold-only 라벨 기준 (20개 법안 서브셋)\n")
        f.write(f"> **Component Normalization:** Weighted Field ({NORMALIZE_COMPONENTS_FOR_WEIGHTED_FIELD}), Final Ensemble ({NORMALIZE_COMPONENTS_FOR_FINAL_ENSEMBLE}) (Method: row_minmax)\n\n")
        
        f.write("## 1. 최적 가중치 요약\n\n")
        f.write("### A. Weighted Field 최적 가중치\n")
        f.write("| Objective | w_title | w_full | w_current | w_problem | w_proposal | w_article | Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(f"| **best_by_objective_score** | {best_wf_by_obj['w_title']:.2f} | {best_wf_by_obj['w_full']:.2f} | {best_wf_by_obj['w_current']:.2f} | {best_wf_by_obj['w_problem']:.2f} | {best_wf_by_obj['w_proposal']:.2f} | {best_wf_by_obj['w_article']:.2f} | {best_wf_by_obj['objective_score']:.4f} |\n")
        f.write(f"| **best_by_candidate_recall_objective** | {best_wf_by_recall['w_title']:.2f} | {best_wf_by_recall['w_full']:.2f} | {best_wf_by_recall['w_current']:.2f} | {best_wf_by_recall['w_problem']:.2f} | {best_wf_by_recall['w_proposal']:.2f} | {best_wf_by_recall['w_article']:.2f} | {best_wf_by_recall['candidate_recall_objective']:.4f} |\n")
        f.write(f"| **best_by_ranking_objective** | {best_wf_by_rank['w_title']:.2f} | {best_wf_by_rank['w_full']:.2f} | {best_wf_by_rank['w_current']:.2f} | {best_wf_by_rank['w_problem']:.2f} | {best_wf_by_rank['w_proposal']:.2f} | {best_wf_by_rank['w_article']:.2f} | {best_wf_by_rank['ranking_objective']:.4f} |\n\n")
        
        f.write("### B. Final Ensemble 최적 가중치\n")
        f.write("| Objective | w_raw | w_structured | w_problem_proposal | w_weighted_field | w_cleaned | w_hybrid | Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        f.write(f"| **best_by_objective_score** | {best_fe_by_obj['w_raw']:.2f} | {best_fe_by_obj['w_structured']:.2f} | {best_fe_by_obj['w_problem_proposal']:.2f} | {best_fe_by_obj['w_weighted_field']:.2f} | {best_fe_by_obj['w_cleaned']:.2f} | {best_fe_by_obj['w_hybrid']:.2f} | {best_fe_by_obj['objective_score']:.4f} |\n")
        f.write(f"| **best_by_candidate_recall_objective** | {best_fe_by_recall['w_raw']:.2f} | {best_fe_by_recall['w_structured']:.2f} | {best_fe_by_recall['w_problem_proposal']:.2f} | {best_fe_by_recall['w_weighted_field']:.2f} | {best_fe_by_recall['w_cleaned']:.2f} | {best_fe_by_recall['w_hybrid']:.2f} | {best_fe_by_recall['candidate_recall_objective']:.4f} |\n")
        f.write(f"| **best_by_ranking_objective** | {best_fe_by_rank['w_raw']:.2f} | {best_fe_by_rank['w_structured']:.2f} | {best_fe_by_rank['w_problem_proposal']:.2f} | {best_fe_by_rank['w_weighted_field']:.2f} | {best_fe_by_rank['w_cleaned']:.2f} | {best_fe_by_rank['w_hybrid']:.2f} | {best_fe_by_rank['ranking_objective']:.4f} |\n\n")

        f.write("## 2. 전체 성능 비교표 (75개 법안 혼합 라벨 전체셋)\n\n")
        f.write("| Method | P@5 | P@10 | nDCG@10 | MRR | Avg Relevance | Avg LMS | Objective Score |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
        
        # Baselines
        for name, bl_m in baselines.items():
            f.write(f"| {name} | {bl_m['precision_at_5']:.4f} | {bl_m['precision_at_10']:.4f} | {bl_m['ndcg_at_10']:.4f} | {bl_m['mrr']:.4f} | {bl_m['average_relevance']:.4f} | {bl_m['average_legal_meaning_score']} | {bl_m['objective_score']:.4f} |\n")
        
        # Best WF and FE
        wf_best_full = evaluate_topk(wf_best_topk_export, label_map)
        wf_best_obj = compute_objectives(wf_best_full)
        f.write(f"| **weighted_field_best** | {wf_best_full['precision_at_5']:.4f} | {wf_best_full['precision_at_10']:.4f} | {wf_best_full['ndcg_at_10']:.4f} | {wf_best_full['mrr']:.4f} | {wf_best_full['average_relevance']:.4f} | {wf_best_full['average_legal_meaning_score']} | {wf_best_obj['objective_score']:.4f} |\n")
        
        fe_best_full = evaluate_topk(fe_best_topk_export, label_map)
        fe_best_obj = compute_objectives(fe_best_full)
        f.write(f"| **final_ensemble_best** | {fe_best_full['precision_at_5']:.4f} | {fe_best_full['precision_at_10']:.4f} | {fe_best_full['ndcg_at_10']:.4f} | {fe_best_full['mrr']:.4f} | {fe_best_full['average_relevance']:.4f} | {fe_best_full['average_legal_meaning_score']} | {fe_best_obj['objective_score']:.4f} |\n\n")

        if gold_sources:
            f.write("## 3. Gold-only 성능 비교표 (20개 법안 골드 라벨)\n\n")
            f.write("| Method | P@5 | P@10 | nDCG@10 | MRR | Avg Relevance | Avg LMS | Objective Score |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for name, m_g in gold_report_data.items():
                f.write(f"| {name} | {m_g['precision_at_5']:.4f} | {m_g['precision_at_10']:.4f} | {m_g['ndcg_at_10']:.4f} | {m_g['mrr']:.4f} | {m_g['average_relevance']:.4f} | {m_g['average_legal_meaning_score']} | {m_g['objective_score']:.4f} |\n")
            f.write("\n")

        f.write("## 4. Train / Validation 검증 결과\n\n")
        f.write("### A. Fold-0 80/20 Split 검증\n")
        f.write("- **Weighted Field**:\n")
        f.write(f"  - Train 최적 가중치: {fold0_train_wf_weights}\n")
        f.write(f"  - Validation P@5: {fold0_val_wf_metrics['precision_at_5']:.4f}, nDCG@10: {fold0_val_wf_metrics['ndcg_at_10']:.4f}, MRR: {fold0_val_wf_metrics['mrr']:.4f}, Objective: {fold0_val_wf_metrics['objective_score']:.4f}\n")
        f.write("- **Final Ensemble**:\n")
        f.write(f"  - Train 최적 가중치: {fold0_train_fe_weights}\n")
        f.write(f"  - Validation P@5: {fold0_val_fe_metrics['precision_at_5']:.4f}, nDCG@10: {fold0_val_fe_metrics['ndcg_at_10']:.4f}, MRR: {fold0_val_fe_metrics['mrr']:.4f}, Objective: {fold0_val_fe_metrics['objective_score']:.4f}\n\n")
        
        f.write("### B. 5-Fold Cross Validation 평균 성능\n")
        f.write(f"- Weighted Field 평균 Validation Objective Score: **{cv_wf_mean_val:.4f}**\n")
        f.write(f"- Final Ensemble 평균 Validation Objective Score: **{cv_fe_mean_val:.4f}**\n\n")

        f.write("## 5. Component Normalization 상세 통계\n\n")
        f.write("| Component | Pre-Min | Pre-Max | Pre-Mean | Post-Min | Post-Max | Post-Mean |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for c_name, st in normalization_stats.items():
            f.write(f"| {c_name} | {st['before_min']:.4f} | {st['before_max']:.4f} | {st['before_mean']:.4f} | {st['after_min']:.4f} | {st['after_max']:.4f} | {st['after_mean']:.4f} |\n")

    print(f"\n모든 작업 완료! (소요 시간: {time.time() - start_time:.2f}초)")


if __name__ == "__main__":
    main()
