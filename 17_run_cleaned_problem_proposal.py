#!/usr/bin/env python3
"""
17_run_cleaned_problem_proposal.py
==================================
cleaned_problem_proposal SBERT 유사도 실험을 실행합니다.

실행 방법:
    python 17_run_cleaned_problem_proposal.py
"""

import os
import json
import sys
import io
import torch
from sentence_transformers import SentenceTransformer
from bill_text_parser import split_summary_sections
from clean_text_utils import normalize_legal_text_for_sbert

# Windows 환경 한글 출력 깨짐 방지
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
    print("  cleaned_problem_proposal SBERT 임베딩 및 유사도 연산")
    print("=" * 70)
    
    # 장치 및 모델 로드
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    print(f"[1] 모델 로딩 중: {model_name} (장치: {device})")
    model = SentenceTransformer(model_name, device=device)
    print("    모델 로드 완료!")
    
    # 데이터 로드
    dataset_path = 'test_dataset/full_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"[ERROR] 데이터셋 파일이 없습니다: {dataset_path}")
        sys.exit(1)
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    print(f"[2] 전체 법안 수: {len(bills)}개")
    
    # 텍스트 빌딩 및 정제
    texts = []
    print("[3] 텍스트 정제(SBERT용) 수행 중...")
    for idx, bill in enumerate(bills):
        summary = bill.get('summary', '')
        sections = split_summary_sections(summary)
        
        problem = sections.get('problem', '').strip()
        proposal = sections.get('proposal', '').strip()
        
        # problem + proposal 결합
        combined_text = f"{problem} {proposal}".strip()
        
        # normalize_legal_text_for_sbert 적용
        cleaned_text = normalize_legal_text_for_sbert(combined_text)
        
        # fallback: 텍스트가 너무 짧은 경우 bill_name + summary 정제본 사용
        if len(cleaned_text) < 5:
            fallback_raw = f"{bill.get('bill_name', '')} {summary}".strip()
            cleaned_text = normalize_legal_text_for_sbert(fallback_raw)
            
        texts.append(cleaned_text)
        
    # SBERT 임베딩 생성
    print("[4] SBERT 임베딩 인코딩 중...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True
    )
    
    # 유사도 계산
    print("[5] 유사도 매트릭스 계산 중...")
    sim_matrix = embeddings @ embeddings.T
    
    # 자기 자신 제외
    sim_matrix.fill_diagonal_(-999.0)
    
    num_bills = len(bills)
    topk_vals, topk_idxs = torch.topk(sim_matrix, k=min(10, num_bills - 1), dim=1)
    
    # 결과 구성
    results = []
    for i in range(num_bills):
        source_bill = bills[i]
        source_cats = source_bill.get('categories', [])
        
        for rank_idx in range(topk_vals.shape[1]):
            target_idx = topk_idxs[i][rank_idx].item()
            similarity_val = topk_vals[i][rank_idx].item()
            
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
                "similarity": round(similarity_val, 6),
                "method": "cleaned_problem_proposal",
                "source_categories": source_cats,
                "target_categories": target_cats,
                "category_intersection": intersection,
                "jaccard_similarity": round(jaccard_sim, 6)
            }
            results.append(item)
            
    # 출력 저장
    output_dir = 'Sbert_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topk_cleaned_problem_proposal.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"[출력] 결과 저장 완료: {output_path} (총 {len(results)}쌍)")
    print("=" * 70)

if __name__ == '__main__':
    main()
