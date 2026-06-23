"""
실행 방법:
python 03_test_text_builders.py

결과 파일:
Sbert_output/text_builder_samples.json
"""

import os
import json
import sys
import io
from bill_text_parser import split_summary_sections
from text_builders import build_raw_text, build_structured_text, build_problem_proposal_text

# Windows 환경 한글 출력 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def test_builders():
    input_path = 'test_dataset/full_dataset.json'
    output_dir = 'Sbert_output'
    output_path = os.path.join(output_dir, 'text_builder_samples.json')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 파일이 존재하지 않습니다.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    test_bills = bills[:10]  # 앞 10개 법안
    
    builder_results = []
    
    print(f"=== 구조화 텍스트 생성 테스트 (앞 {len(test_bills)}개 법안) ===")
    
    for i, bill in enumerate(test_bills, 1):
        bill_id = bill.get('bill_id', 'N/A')
        bill_name = bill.get('bill_name', 'N/A')
        summary = bill.get('summary', '')
        
        # 1. 2단계 파서를 이용해 섹션 분리
        sections = split_summary_sections(summary)
        
        # 2. 텍스트 빌더들을 이용해 버전별 텍스트 생성
        raw_txt = build_raw_text(bill)
        structured_txt = build_structured_text(bill, sections)
        problem_proposal_txt = build_problem_proposal_text(bill, sections)
        
        # 출력
        print(f"\n[{i}/10] 법안 ID: {bill_id} | 법안명: {bill_name}")
        print("-" * 50)
        print("[raw_text] (일부)\n" + (raw_txt[:150] + "..." if len(raw_txt) > 150 else raw_txt))
        print("-" * 50)
        print("[structured_text] (일부)\n" + (structured_txt[:250] + "..." if len(structured_txt) > 250 else structured_txt))
        print("-" * 50)
        print("[problem_proposal_text] (일부)\n" + (problem_proposal_txt[:150] + "..." if len(problem_proposal_txt) > 150 else problem_proposal_txt))
        print("=" * 50)
        
        # 저장용 데이터 구성
        builder_results.append({
            "bill_id": bill_id,
            "bill_name": bill_name,
            "raw_text": raw_txt,
            "structured_text": structured_txt,
            "problem_proposal_text": problem_proposal_txt
        })
        
    # JSON 파일로 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(builder_results, f, ensure_ascii=False, indent=4)
        
    print(f"\n구조화 텍스트 생성 결과가 {output_path}에 저장되었습니다.")

if __name__ == '__main__':
    test_builders()
