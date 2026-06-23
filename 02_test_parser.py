"""
실행 방법:
python 02_test_parser.py

결과 파일:
Sbert_output/parsed_summary_samples.json
"""

import os
import json
import sys
import io
from bill_text_parser import split_summary_sections

# Windows 환경 한글 출력 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_parser():
    input_path = 'test_dataset/full_dataset.json'
    output_dir = 'Sbert_output'
    output_path = os.path.join(output_dir, 'parsed_summary_samples.json')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 파일이 존재하지 않습니다.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    test_bills = bills[:20]  # 앞 20개 법안 선택
    
    parsed_results = []
    
    print(f"=== 국회 발의안 요약문 파싱 테스트 (앞 {len(test_bills)}개 법안) ===")
    
    for i, bill in enumerate(test_bills, 1):
        bill_id = bill.get('bill_id', 'N/A')
        bill_name = bill.get('bill_name', 'N/A')
        summary = bill.get('summary', '')
        
        parsed = split_summary_sections(summary)
        
        # 출력
        print(f"\n[{i}/20] 법안 ID: {bill_id} | 법안명: {bill_name}")
        print(f"  - 현행법 설명 (current_law): {parsed['current_law']}")
        print(f"  - 문제점 (problem): {parsed['problem']}")
        print(f"  - 개정안 (proposal): {parsed['proposal']}")
        print(f"  - 조문 텍스트 (article_text): {parsed['article_text']}")
        print(f"  - 조문 번호 리스트 (article_numbers): {parsed['article_numbers']}")
        
        # 저장용 결과 객체 구성
        parsed_results.append({
            "bill_id": bill_id,
            "bill_name": bill_name,
            "parsed_data": parsed
        })
        
    # JSON 파일로 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n파싱 테스트 결과가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    test_parser()
