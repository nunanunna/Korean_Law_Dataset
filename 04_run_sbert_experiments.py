"""
실행 방법:
python 04_run_sbert_experiments.py

결과 파일:
Sbert_output/topk_raw.json
Sbert_output/topk_structured.json
Sbert_output/topk_problem_proposal.json
"""

import os
import json
import sys
import io
import torch
from sentence_transformers import SentenceTransformer
from bill_text_parser import split_summary_sections
from text_builders import build_raw_text, build_structured_text, build_problem_proposal_text

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

def run_experiment_for_version(version_name, text_builder_fn, bills, model, output_path):
    print(f"\n[{version_name.upper()}] 버전 임베딩 생성 및 유사도 계산 중...")
    
    # 1. 입력 텍스트 생성
    texts = []
    for bill in bills:
        if version_name == 'raw':
            txt = text_builder_fn(bill)
        else:
            summary = bill.get('summary', '')
            sections = split_summary_sections(summary)
            txt = text_builder_fn(bill, sections)
        texts.append(txt)
        
    # 2. SBERT 임베딩 인코딩
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True
    )
    
    # 3. 유사도 행렬 계산 (normalize_embeddings=True이므로 행렬 곱으로 코사인 유사도 연산)
    sim_matrix = embeddings @ embeddings.T
    
    # 자기 자신과의 유사도 제외 (대각 성분을 충분히 작은 값으로 설정)
    sim_matrix.fill_diagonal_(-999.0)
    
    # 각 법안별 top-10 인덱스와 값 추출
    num_bills = len(bills)
    topk_vals, topk_idxs = torch.topk(sim_matrix, k=min(10, num_bills - 1), dim=1)
    
    # 4. 결과 구조화
    results = []
    for i in range(num_bills):
        source_bill = bills[i]
        source_cats = source_bill.get('categories', [])
        
        for rank_idx in range(topk_vals.shape[1]):
            target_idx = topk_idxs[i][rank_idx].item()
            similarity_val = topk_vals[i][rank_idx].item()
            
            target_bill = bills[target_idx]
            target_cats = target_bill.get('categories', [])
            
            # 카테고리 교집합 및 Jaccard 유사도 계산
            intersection, jaccard_sim = calculate_jaccard(source_cats, target_cats)
            
            item = {
                "source_bill_id": source_bill["bill_id"],
                "source_bill_name": source_bill["bill_name"],
                "target_bill_id": target_bill["bill_id"],
                "target_bill_name": target_bill["bill_name"],
                "rank": rank_idx + 1,
                "similarity": round(similarity_val, 6),
                "source_categories": source_cats,
                "target_categories": target_cats,
                "category_intersection": intersection,
                "jaccard_similarity": round(jaccard_sim, 6)
            }
            results.append(item)
            
    # 5. JSON 저장
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
        
    print(f"[{version_name.upper()}] 결과 저장 완료: {output_path} (총 {len(results)}개 페어)")

def main():
    # 장치 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"사용 장치: {device}")
    
    # 모델 로드
    model_name = 'woong0322/ko-legal-sbert-finetuned'
    print(f"SBERT 모델 로딩 중: {model_name}...")
    model = SentenceTransformer(model_name, device=device)
    print("모델 로드 완료!")
    
    # 데이터 로드
    dataset_path = 'test_dataset/full_dataset.json'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} 파일이 존재하지 않습니다.")
        return
        
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    print(f"전체 법안 수: {len(bills)}")
    
    output_dir = 'Sbert_output'
    os.makedirs(output_dir, exist_ok=True)
    
    # 실험 수행
    experiments = [
        ('raw', build_raw_text, os.path.join(output_dir, 'topk_raw.json')),
        ('structured', build_structured_text, os.path.join(output_dir, 'topk_structured.json')),
        ('problem_proposal', build_problem_proposal_text, os.path.join(output_dir, 'topk_problem_proposal.json'))
    ]
    
    for version_name, builder_fn, output_path in experiments:
        run_experiment_for_version(version_name, builder_fn, bills, model, output_path)
        
    print("\n모든 SBERT 임베딩 실험이 성공적으로 완료되었습니다!")

if __name__ == '__main__':
    main()
