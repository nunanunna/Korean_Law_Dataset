#!/usr/bin/env python3
"""
18_run_keyword_tfidf.py
=======================
problem + proposal에 대해 TF-IDF cosine similarity를 계산하여 top-10을 추천합니다.

실행 방법:
    python 18_run_keyword_tfidf.py
"""

import os
import json
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bill_text_parser import split_summary_sections
from clean_text_utils import normalize_legal_text_for_keywords, extract_light_keywords

# Windows 한글 출력 깨짐 방지
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

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
    print("  keyword_tfidf 기반 코사인 유사도 연산")
    print("=" * 70)
    
    # 데이터 로드
    dataset_path = 'test_dataset/full_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 데이터셋 파일이 없습니다: {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    print(f"[1] 전체 법안 수: {len(bills)}개")
    
    # 키워드 텍스트 생성
    texts = []
    print("[2] 키워드 추출 및 텍스트 정제(TF-IDF용) 중...")
    for bill in bills:
        summary = bill.get('summary', '')
        sections = split_summary_sections(summary)
        
        problem = sections.get('problem', '').strip()
        proposal = sections.get('proposal', '').strip()
        
        # problem + proposal 결합
        combined_text = f"{problem} {proposal}".strip()
        
        # 1차 정제 및 2차 가벼운 키워드 추출 조합
        cleaned_text = normalize_legal_text_for_keywords(combined_text)
        keyword_text = extract_light_keywords(cleaned_text)
        
        # fallback: 정제 후 너무 짧으면 bill_name + summary 사용
        if len(keyword_text.strip()) < 3:
            fallback_raw = f"{bill.get('bill_name', '')} {summary}".strip()
            cleaned_text = normalize_legal_text_for_keywords(fallback_raw)
            keyword_text = extract_light_keywords(cleaned_text)
            
        texts.append(keyword_text)
        
    # TF-IDF 벡터화
    print("[3] TfidfVectorizer 학습 및 변환 중...")
    vectorizer = TfidfVectorizer(
        tokenizer=None,
        token_pattern=r"(?u)\b[가-힣A-Za-z0-9]{2,}\b",
        min_df=1,
        max_df=0.85,
        ngram_range=(1, 2)
    )
    
    tfidf_matrix = vectorizer.fit_transform(texts)
    print(f"    TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    
    # Cosine Similarity 계산
    print("[4] Cosine Similarity 연산 중...")
    sim_matrix = cosine_similarity(tfidf_matrix)
    
    # 자기 자신 제외 (대각 성분을 0 또는 충분히 작은 값으로 설정)
    np.fill_diagonal(sim_matrix, -999.0)
    
    num_bills = len(bills)
    results = []
    
    print("[5] 각 법안별 top-10 추출 중...")
    for i in range(num_bills):
        source_bill = bills[i]
        source_cats = source_bill.get('categories', [])
        
        # 현재 source 법안의 유사도
        source_sims = sim_matrix[i]
        
        # 내림차순 정렬하여 상위 10개 인덱스 추출
        topk_idxs = np.argsort(source_sims)[::-1][:10]
        
        for rank_idx, target_idx in enumerate(topk_idxs):
            similarity_val = source_sims[target_idx]
            # 만약 자기 자신인 경우를 완벽 방지
            if target_idx == i:
                continue
                
            target_bill = bills[target_idx]
            target_cats = target_bill.get('categories', [])
            
            # 카테고리 자카드 유사도
            intersection, jaccard_sim = calculate_jaccard(source_cats, target_cats)
            
            item = {
                "source_bill_id": source_bill["bill_id"],
                "source_bill_name": source_bill["bill_name"],
                "target_bill_id": target_bill["bill_id"],
                "target_bill_name": target_bill["bill_name"],
                "rank": rank_idx + 1,
                "similarity": round(float(similarity_val), 6),
                "method": "keyword_tfidf",
                "source_categories": source_cats,
                "target_categories": target_cats,
                "category_intersection": intersection,
                "jaccard_similarity": round(jaccard_sim, 6)
            }
            results.append(item)
            
    # 출력 저장
    output_dir = 'Sbert_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topk_keyword_tfidf.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"[출력] 결과 저장 완료: {output_path} (총 {len(results)}쌍)")
    print("=" * 70)

if __name__ == '__main__':
    main()
