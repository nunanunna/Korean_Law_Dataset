"""
실행 방법:
python 05_weighted_field_similarity.py

결과 파일:
Sbert_output/topk_weighted_field_similarity.json
"""

import os
import json
import sys
import io
import torch
from sentence_transformers import SentenceTransformer
from bill_text_parser import normalize_summary, split_summary_sections

# Windows 환경 한글 출력 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def calculate_jaccard(cats1, cats2):
    """
    두 카테고리 리스트 간의 교집합과 Jaccard 유사도를 계산합니다.
    """
    set1 = set([c.strip() for c in cats1 if c and c.strip()])
    set2 = set([c.strip() for c in cats2 if c and c.strip()])
    
    intersection = sorted(list(set1.intersection(set2)))
    union = sorted(list(set1.union(set2)))
    
    jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0.0
    return intersection, jaccard_sim

def main():
    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")
    
    # 모델 로딩
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    print(f"SBERT 모델 로딩 중: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("모델 로드 완료!")
    
    # 데이터 로딩
    dataset_path = 'test_dataset/full_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} 파일이 존재하지 않습니다.")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    num_bills = len(bills)
    print(f"전체 법안 수: {num_bills}")
    
    # 필드 및 가중치 정의
    fields = ["full_text", "current_law", "problem", "proposal", "article_text"]
    weights = {
        "full_text": 0.20,
        "current_law": 0.05,
        "problem": 0.35,
        "proposal": 0.35,
        "article_text": 0.05
    }
    
    # 1. 각 필드별 텍스트 목록 생성
    field_texts = {field: [] for field in fields}
    
    for bill in bills:
        summary = bill.get('summary', '')
        # full_text: '제안이유 및 주요내용'이 제거된 정규화 텍스트
        full_text = normalize_summary(summary)
        # sections: current_law, problem, proposal, article_text
        sections = split_summary_sections(summary)
        
        parsed_fields = {
            "full_text": full_text,
            "current_law": sections.get('current_law', ''),
            "problem": sections.get('problem', ''),
            "proposal": sections.get('proposal', ''),
            "article_text": sections.get('article_text', '')
        }
        
        for field in fields:
            val = parsed_fields[field].strip()
            # 빈 텍스트일 때 대체용 라벨링
            if not val:
                val = "[내용 없음]"
            field_texts[field].append(val)
            
    # 2. 각 필드별 임베딩 생성 및 similarity matrix 구축
    sim_matrices = {}
    for field in fields:
        print(f"[{field}] 필드 임베딩 생성 및 유사도 계산 중...")
        embeddings = model.encode(
            field_texts[field],
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_tensor=True
        )
        sim_matrix = embeddings @ embeddings.T
        sim_matrices[field] = sim_matrix
        
    # 3. 가중합 기반 final similarity matrix 생성
    final_sim_matrix = torch.zeros((num_bills, num_bills), device=device)
    for field in fields:
        final_sim_matrix += weights[field] * sim_matrices[field]
        
    # 자기 자신과의 유사도 제외
    final_sim_matrix.fill_diagonal_(-999.0)
    
    # 각 법안별 top-10 추출
    topk_vals, topk_idxs = torch.topk(final_sim_matrix, k=min(10, num_bills - 1), dim=1)
    
    # 4. 결과 데이터 구조화
    results = []
    for i in range(num_bills):
        source_bill = bills[i]
        source_cats = source_bill.get('categories', [])
        
        for rank_idx in range(topk_vals.shape[1]):
            target_idx = topk_idxs[i][rank_idx].item()
            final_similarity_score = topk_vals[i][rank_idx].item()
            
            target_bill = bills[target_idx]
            target_cats = target_bill.get('categories', [])
            
            # 카테고리 비교
            intersection, jaccard_sim = calculate_jaccard(source_cats, target_cats)
            
            # 개별 필드별 유사도 점수 모음
            field_scores = {}
            for field in fields:
                field_scores[field] = round(sim_matrices[field][i][target_idx].item(), 6)
                
            item = {
                "source_bill_id": source_bill["bill_id"],
                "source_bill_name": source_bill["bill_name"],
                "target_bill_id": target_bill["bill_id"],
                "target_bill_name": target_bill["bill_name"],
                "rank": rank_idx + 1,
                "final_similarity": round(final_similarity_score, 6),
                "field_scores": field_scores,
                "category_intersection": intersection,
                "jaccard_similarity": round(jaccard_sim, 6)
            }
            results.append(item)
            
    # 5. JSON 저장
    output_dir = 'Sbert_output'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'topk_weighted_field_similarity.json')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"\n가중합 유사도 저장 완료: {output_path} (총 {len(results)}개 페어)")

if __name__ == '__main__':
    main()
