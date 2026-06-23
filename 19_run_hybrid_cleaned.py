#!/usr/bin/env python3
"""
19_run_hybrid_cleaned.py
========================
cleaned_problem_proposal SBERT 유사도, keyword_tfidf 유사도, 조문 유사도를 가중합하여
최종 하이브리드 유사도를 계산하고 top-10을 추천합니다.

식:
    hybrid_score = 0.70 * SBERT_sim + 0.20 * TFIDF_sim + 0.10 * Article_sim

실행 방법:
    python 19_run_hybrid_cleaned.py
"""

import os
import json
import sys
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bill_text_parser import split_summary_sections
from clean_text_utils import normalize_legal_text_for_sbert, normalize_legal_text_for_keywords, extract_light_keywords
from article_similarity_utils import compute_article_similarity

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ── 하이퍼파라미터 및 옵션 정의 ────────────────────────────────────────
USE_ROW_MINMAX_FOR_TFIDF = True  # TF-IDF 유사도의 row별 min-max 정규화 여부

def calculate_jaccard(cats1, cats2):
    """두 카테고리 리스트 간의 Jaccard 유사도 계산"""
    set1 = set([c.strip() for c in cats1 if c and c.strip()])
    set2 = set([c.strip() for c in cats2 if c and c.strip()])
    
    intersection = sorted(list(set1.intersection(set2)))
    union = sorted(list(set1.union(set2)))
    
    jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
    return intersection, jaccard_sim

def main():
    print("=" * 70)
    print("  hybrid_cleaned 하이브리드 법안 유사도 연산")
    print(f"  - TF-IDF Min-Max 정규화 옵션: {USE_ROW_MINMAX_FOR_TFIDF}")
    print("=" * 70)
    
    # ── 데이터 로드 ──────────────────────────────────────────────────
    dataset_path = 'test_dataset/full_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 데이터셋 파일이 없습니다: {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    num_bills = len(bills)
    print(f"[1] 전체 법안 수: {num_bills}개")
    
    # ── 1. SBERT 유사도 행렬 계산 ─────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    print(f"[2] SBERT 모델 로딩 및 임베딩 생성 중... (장치: {device})")
    model = SentenceTransformer(model_name, device=device)
    
    sbert_texts = []
    all_sections = []
    
    for bill in bills:
        summary = bill.get('summary', '')
        sections = split_summary_sections(summary)
        all_sections.append(sections)
        
        problem = sections.get('problem', '').strip()
        proposal = sections.get('proposal', '').strip()
        combined_text = f"{problem} {proposal}".strip()
        
        cleaned_text = normalize_legal_text_for_sbert(combined_text)
        if len(cleaned_text) < 5:
            fallback_raw = f"{bill.get('bill_name', '')} {summary}".strip()
            cleaned_text = normalize_legal_text_for_sbert(fallback_raw)
        sbert_texts.append(cleaned_text)
        
    sbert_embeddings = model.encode(
        sbert_texts,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_tensor=False
    )
    # Cosine Similarity (Normalized embeddings의 내적)
    sbert_sim_matrix = np.dot(sbert_embeddings, sbert_embeddings.T)
    
    # ── 2. TF-IDF 유사도 행렬 계산 ─────────────────────────────────────
    print("[3] TF-IDF 유사도 행렬 생성 중...")
    tfidf_texts = []
    for idx, bill in enumerate(bills):
        sections = all_sections[idx]
        problem = sections.get('problem', '').strip()
        proposal = sections.get('proposal', '').strip()
        combined_text = f"{problem} {proposal}".strip()
        
        cleaned_text = normalize_legal_text_for_keywords(combined_text)
        keyword_text = extract_light_keywords(cleaned_text)
        if len(keyword_text.strip()) < 3:
            fallback_raw = f"{bill.get('bill_name', '')} {bill.get('summary', '')}".strip()
            cleaned_text = normalize_legal_text_for_keywords(fallback_raw)
            keyword_text = extract_light_keywords(cleaned_text)
        tfidf_texts.append(keyword_text)
        
    vectorizer = TfidfVectorizer(
        tokenizer=None,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    tfidf_matrix = vectorizer.fit_transform(tfidf_texts)
    tfidf_sim_matrix = cosine_similarity(tfidf_matrix)
    
    # TF-IDF Row별 Min-Max 정규화 적용
    if USE_ROW_MINMAX_FOR_TFIDF:
        norm_tfidf_sim_matrix = np.zeros_like(tfidf_sim_matrix)
        for i in range(num_bills):
            # 자기 자신과의 코사인 유사도(1.0)를 제외하고 정규화하기 위해 대각선을 제외한 인덱스 추출
            row_vals = tfidf_sim_matrix[i].copy()
            row_vals[i] = -999.0  # 대각선 마스킹
            active_vals = row_vals[row_vals > -900.0]
            
            if len(active_vals) > 0:
                min_val = np.min(active_vals)
                max_val = np.max(active_vals)
                denom = max_val - min_val
                
                # 0 나누기 예외 처리
                if denom > 1e-8:
                    row_norm = (tfidf_sim_matrix[i] - min_val) / denom
                else:
                    row_norm = tfidf_sim_matrix[i] - min_val
                
                row_norm[i] = 1.0  # 자기 자신은 1.0으로 강제
                # 범위 보정 (0 ~ 1)
                row_norm = np.clip(row_norm, 0.0, 1.0)
                norm_tfidf_sim_matrix[i] = row_norm
            else:
                norm_tfidf_sim_matrix[i] = tfidf_sim_matrix[i]
        tfidf_sim_matrix = norm_tfidf_sim_matrix
        
    # ── 3. 조문 유사도 행렬 계산 ─────────────────────────────────────
    print("[4] 조문 유사도 행렬 생성 중...")
    article_sim_matrix = np.zeros((num_bills, num_bills))
    for i in range(num_bills):
        for j in range(num_bills):
            if i == j:
                article_sim_matrix[i][j] = 1.0
            else:
                article_sim_matrix[i][j] = compute_article_similarity(
                    all_sections[i]["article_numbers"],
                    all_sections[j]["article_numbers"]
                )
                
    # ── 4. 하이브리드 행렬 결합 ──────────────────────────────────────
    print("[5] 세 가지 유사도 결합 중...")
    hybrid_matrix = 0.70 * sbert_sim_matrix + 0.20 * tfidf_sim_matrix + 0.10 * article_sim_matrix
    
    # 자기 자신 제외
    masked_hybrid = hybrid_matrix.copy()
    np.fill_diagonal(masked_hybrid, -999.0)
    
    results = []
    for i in range(num_bills):
        source_bill = bills[i]
        source_cats = source_bill.get('categories', [])
        source_sims = masked_hybrid[i]
        
        # 상위 10개 인덱스 추출
        topk_idxs = np.argsort(source_sims)[::-1][:10]
        
        for rank_idx, target_idx in enumerate(topk_idxs):
            similarity_val = source_sims[target_idx]
            if target_idx == i:
                continue
                
            target_bill = bills[target_idx]
            target_cats = target_bill.get('categories', [])
            
            # 카테고리 자카드 유사도
            intersection, jaccard_sim = calculate_jaccard(source_cats, target_cats)
            
            # 컴포넌트 점수 추출
            comp_sbert = float(sbert_sim_matrix[i][target_idx])
            comp_tfidf = float(tfidf_sim_matrix[i][target_idx])
            comp_article = float(article_sim_matrix[i][target_idx])
            
            item = {
                "source_bill_id": source_bill["bill_id"],
                "source_bill_name": source_bill["bill_name"],
                "target_bill_id": target_bill["bill_id"],
                "target_bill_name": target_bill["bill_name"],
                "rank": rank_idx + 1,
                "similarity": round(float(similarity_val), 6),
                "method": "hybrid_cleaned",
                "component_scores": {
                    "cleaned_problem_proposal": round(comp_sbert, 4),
                    "keyword_tfidf": round(comp_tfidf, 4),
                    "article_similarity": round(comp_article, 4)
                },
                "source_categories": source_cats,
                "target_categories": target_cats,
                "category_intersection": intersection,
                "jaccard_similarity": round(jaccard_sim, 6)
            }
            results.append(item)
            
    # 결과 저장
    output_dir = 'Sbert_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topk_hybrid_cleaned.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"[출력] 하이브리드 결과 저장 완료: {output_path} (총 {len(results)}쌍)")
    print("=" * 70)

if __name__ == '__main__':
    main()
