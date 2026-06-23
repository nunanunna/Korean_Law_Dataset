"""
실행 방법:
python 01_inspect_dataset.py

결과 파일:
Sbert_output/dataset_inspection.json
"""

import os
import json
import random
import sys
import io

# Windows 환경 한글 출력 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def inspect_dataset():
    input_path = 'test_dataset/full_dataset.json'
    output_dir = 'Sbert_output'
    output_path = os.path.join(output_dir, 'dataset_inspection.json')
    
    # 1. 파일 읽기
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 파일이 존재하지 않습니다.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    total_bills = len(bills)
    print(f"=== 데이터셋 구조 점검 ===")
    print(f"전체 법안 수: {total_bills}")
    
    # 2. 필드 확인 및 통계 변수 초기화
    required_fields = ['bill_id', 'bill_name', 'summary', 'categories']
    missing_fields_count = {field: 0 for field in required_fields}
    empty_summary_count = 0
    non_list_categories_count = 0
    
    # 키워드 카운트 초기화
    keywords = ["현행법", "그런데", "그러나", "이에", "안 제", "신설", "개정", "삭제"]
    keyword_counts = {kw: 0 for kw in keywords}
    
    summary_lengths = []
    
    for bill in bills:
        # 필드 존재 확인
        for field in required_fields:
            if field not in bill:
                missing_fields_count[field] += 1
                
        # summary 검사
        summary = bill.get('summary')
        if summary is None or (isinstance(summary, str) and not summary.strip()):
            empty_summary_count += 1
        elif isinstance(summary, str):
            summary_lengths.append(len(summary))
            # 키워드 통계
            for kw in keywords:
                keyword_counts[kw] += summary.count(kw)
                
        # categories 검사
        categories = bill.get('categories')
        if categories is not None and not isinstance(categories, list):
            non_list_categories_count += 1
            
    # summary 길이 통계 계산
    if summary_lengths:
        min_len = min(summary_lengths)
        max_len = max(summary_lengths)
        mean_len = sum(summary_lengths) / len(summary_lengths)
    else:
        min_len = max_len = mean_len = 0
        
    print("\n[필드 존재 여부 확인]")
    for field, count in missing_fields_count.items():
        print(f"- {field} 누락 법안 수: {count}")
        
    print(f"\nsummary가 비어 있는 법안 수: {empty_summary_count}")
    print(f"categories가 list가 아닌 법안 수: {non_list_categories_count}")
    
    print("\n[summary 길이 통계]")
    print(f"- 최소 길이: {min_len}")
    print(f"- 최대 길이: {max_len}")
    print(f"- 평균 길이: {mean_len:.2f}")
    
    print("\n[키워드 등장 빈도 통계]")
    for kw, count in keyword_counts.items():
        print(f"- '{kw}': {count}회")
        
    # 3. 임의 샘플 5개 선택 및 출력
    random.seed(42)  # 결과 재현성을 위해 시드 고정
    sample_bills = random.sample(bills, min(5, len(bills)))
    
    print("\n[임의 샘플 5개 출력]")
    samples_data = []
    for i, bill in enumerate(sample_bills, 1):
        bill_id = bill.get('bill_id', 'N/A')
        bill_name = bill.get('bill_name', 'N/A')
        categories = bill.get('categories', [])
        summary = bill.get('summary', '')
        summary_preview = summary[:500] if isinstance(summary, str) else ''
        
        print(f"\n--- 샘플 {i} ---")
        print(f"bill_id: {bill_id}")
        print(f"bill_name: {bill_name}")
        print(f"categories: {categories}")
        print(f"summary 앞 500자:\n{summary_preview}")
        
        samples_data.append({
            "bill_id": bill_id,
            "bill_name": bill_name,
            "categories": categories,
            "summary_preview": summary_preview
        })
        
    # 4. JSON 파일 저장
    os.makedirs(output_dir, exist_ok=True)
    
    output_json = {
        "total_bills": total_bills,
        "missing_fields": missing_fields_count,
        "empty_summary_count": empty_summary_count,
        "non_list_categories_count": non_list_categories_count,
        "summary_length_stats": {
            "min": min_len,
            "max": max_len,
            "mean": round(mean_len, 2)
        },
        "keyword_counts": keyword_counts,
        "samples": samples_data
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
        
    print(f"\n검사 결과가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    inspect_dataset()
