"""
실행 방법:
python 13_select_source_bills.py

입력 파일:
test_dataset/full_dataset.json

출력 파일:
Sbert_output/selected_source_bills.json
"""

import os
import json
import sys
import io

# Windows 환경 한글 출력 깨짐 방지
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# 분야별 키워드 정의
DOMAIN_KEYWORDS = {
    "노동": ["근로", "노동", "고용", "임금", "산재", "노무", "사업장"],
    "교육": ["교육", "학교", "학생", "교원", "평생교육", "교육기관"],
    "주거": ["주택", "임대차", "전세", "보증금", "임차인", "부동산"],
    "복지": ["복지", "장애인", "노인", "아동", "청소년", "지원"],
    "디지털": ["개인정보", "정보통신", "온라인", "플랫폼", "인공지능", "데이터"],
    "형사": ["처벌", "벌칙", "범죄", "수사", "피해자", "보호명령"],
    "환경": ["환경", "폐기물", "탄소", "기후", "오염", "에너지"],
    "보건의료": ["의료", "병원", "환자", "보건", "질병", "의약품"],
    "행정": ["지방자치단체", "국가", "공공기관", "행정", "위원회"],
    "경제": ["사업자", "소상공인", "기업", "금융", "세금", "시장"]
}

# 수동 카테고리 -> 10대 도메인 매핑 (키워드 매칭이 없을 때 백업용)
CATEGORY_TO_DOMAIN = {
    "노동": "노동",
    "교육": "교육",
    "주거": "주거",
    "복지": "복지",
    "디지털": "디지털",
    "환경·기후": "환경",
    "보건": "보건의료",
    "정치·행정": "행정",
    "경제": "경제",
    "생활안전": "형사"
}

# 평가 품질 강화를 위한 요약문 패턴 리스트
PATTERNS = ["현행법", "그런데", "그러나", "이에", "안 제", "신설", "개정"]


def estimate_bill_domain(bill):
    """
    법안명과 요약문 키워드 매칭을 통해 10대 분야(estimated_domain)를 추정합니다.
    """
    bill_name = bill.get("bill_name", "")
    summary = bill.get("summary", "")
    
    scores = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            # 법안명에 키워드가 나타나면 높은 가중치 부여
            score += bill_name.count(kw) * 5
            score += summary.count(kw)
        scores[domain] = score
        
    max_domain = max(scores, key=scores.get)
    if scores[max_domain] > 0:
        return max_domain
        
    # 키워드 매칭이 안 되는 경우 수동 카테고리 기반 백업 매핑
    manual_cats = bill.get("categories", [])
    if manual_cats:
        for cat in manual_cats:
            if cat in CATEGORY_TO_DOMAIN:
                return CATEGORY_TO_DOMAIN[cat]
                
    return "행정"  # 최후의 기본값


def estimate_difficulty(bill):
    """
    법안의 난이도(easy, medium, hard)를 추정합니다.
    - hard: 법적 구조가 복잡한 키워드가 포함되거나, 요약문이 아주 길거나, 다중 카테고리인 경우
    - easy: 구조가 명확하고, 적당한 길이에 1개 카테고리만 있는 경우
    - medium: 그 외 중간 영역
    """
    summary = bill.get("summary", "")
    manual_cats = bill.get("categories", [])
    length = len(summary)
    
    complex_terms = ["준용", "예외", "제외", "면제", "부담금", "특례", "과태료", "3배", "징벌적"]
    has_complex_term = any(term in summary for term in complex_terms)
    
    # 1. Hard 조건
    if length > 1200 or len(manual_cats) >= 3 or has_complex_term:
        return "hard"
        
    # 2. Easy 조건
    has_clear_structure = "현행법" in summary and "이에" in summary and ("하고자" in summary or "하려는 것" in summary)
    if 300 <= length <= 700 and len(manual_cats) == 1 and has_clear_structure:
        return "easy"
        
    # 3. Medium 조건
    return "medium"


def select_source_bills():
    input_path = 'test_dataset/full_dataset.json'
    output_dir = 'Sbert_output'
    output_path = os.path.join(output_dir, 'selected_source_bills.json')
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} 파일이 존재하지 않습니다.")
        return
        
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    bills = data.get('bills', [])
    total_bills = len(bills)
    
    # 1. 필터링: 요약문이 없거나 너무 짧은 법안 제외
    # (평가의 신뢰성을 위해 요약문이 최소 300자 이상인 법안만 선별)
    valid_bills = []
    excluded_count = 0
    for bill in bills:
        summary = bill.get("summary", "")
        if not summary or len(summary) < 300:
            excluded_count += 1
            continue
        valid_bills.append(bill)
        
    # 2. 법안 속성 전처리
    preprocessed_bills = []
    for bill in valid_bills:
        summary = bill.get("summary", "")
        detected = [p for p in PATTERNS if p in summary]
        estimated_domain = estimate_bill_domain(bill)
        difficulty = estimate_difficulty(bill)
        
        preprocessed_bills.append({
            "bill_id": bill.get("bill_id"),
            "bill_name": bill.get("bill_name"),
            "summary": summary,
            "manual_categories": bill.get("categories", []),
            "estimated_domain": estimated_domain,
            "difficulty": difficulty,
            "summary_length": len(summary),
            "detected_patterns": detected
        })
        
    # 3. 그리디 선택 알고리즘 실행 (20개 선정)
    selected_indices = set()
    selected_bills_data = []
    
    # 분야별, 난이도별 균형을 맞추기 위한 선택 상태 트래킹 변수
    selected_domains_count = {d: 0 for d in DOMAIN_KEYWORDS}
    selected_categories_count = {}
    selected_difficulties_count = {"easy": 0, "medium": 0, "hard": 0}
    
    for rank in range(1, 21):
        best_idx = -1
        best_score = -99999.0
        best_score_breakdown = {}
        
        for idx, bill in enumerate(preprocessed_bills):
            if idx in selected_indices:
                continue
                
            # 3.1. 요약문 길이 점수 (최대 12점, 1200자 기준 선형 가산)
            length_score = min(bill["summary_length"], 1200) / 100.0
            
            # 3.2. 패턴 감지 점수 (최대 14점, 패턴당 2점)
            pattern_score = len(bill["detected_patterns"]) * 2.0
            
            # 3.3. 분야 다양성 점수 (이미 많이 뽑힌 분야는 감점, 미선택 분야는 높은 가점)
            domain_count = selected_domains_count.get(bill["estimated_domain"], 0)
            domain_diversity_score = 10.0 - (domain_count * 4.0)
            
            # 3.4. 수동 카테고리 다양성 점수
            cat_scores = []
            for cat in bill["manual_categories"]:
                cat_count = selected_categories_count.get(cat, 0)
                cat_scores.append(10.0 - (cat_count * 2.0))
            category_diversity_score = sum(cat_scores) / len(cat_scores) if cat_scores else 5.0
            
            # 3.5. 난이도 균형 점수 (easy, medium, hard의 고른 분포 지향)
            diff_count = selected_difficulties_count[bill["difficulty"]]
            difficulty_balance_score = 10.0 - (diff_count * 2.5)
            
            total_score = (
                length_score + 
                pattern_score + 
                domain_diversity_score + 
                category_diversity_score + 
                difficulty_balance_score
            )
            
            if total_score > best_score:
                best_score = total_score
                best_idx = idx
                best_score_breakdown = {
                    "length_score": length_score,
                    "pattern_score": pattern_score,
                    "domain_diversity_score": domain_diversity_score,
                    "category_diversity_score": category_diversity_score,
                    "difficulty_balance_score": difficulty_balance_score
                }
                
        if best_idx != -1:
            selected_indices.add(best_idx)
            chosen = preprocessed_bills[best_idx]
            
            # 상태 업데이트
            selected_domains_count[chosen["estimated_domain"]] += 1
            for cat in chosen["manual_categories"]:
                selected_categories_count[cat] = selected_categories_count.get(cat, 0) + 1
            selected_difficulties_count[chosen["difficulty"]] += 1
            
            reason = (
                f"요약문 길이 {chosen['summary_length']}자, 패턴 {len(chosen['detected_patterns'])}개 감지. "
                f"추정 분야 '{chosen['estimated_domain']}'(누적 {selected_domains_count[chosen['estimated_domain']]}개) 및 "
                f"난이도 '{chosen['difficulty']}'의 법안으로서 균형적 평가를 위해 선정됨."
            )
            
            selected_bills_data.append({
                "priority_rank": rank,
                "bill_id": chosen["bill_id"],
                "bill_name": chosen["bill_name"],
                "manual_categories": chosen["manual_categories"],
                "estimated_domain": chosen["estimated_domain"],
                "difficulty": chosen["difficulty"],
                "selection_score": round(best_score, 2),
                "selection_reason": reason,
                "summary_length": chosen["summary_length"],
                "detected_patterns": chosen["detected_patterns"]
            })
            
    # 4. Quick 10 Subset 선정
    # 선정된 20개 법안 중에서, 10대 도메인별로 가장 대표성(선정 우선순위가 높은)을 띠는 1개씩을 골라 구성
    quick_subset = []
    domain_to_best_bill = {}
    
    # 20개 선정 법안은 priority_rank 순으로 정렬되어 있으므로, 
    # 각 도메인별로 처음 마주치는 법안이 해당 도메인 내 최고 우선순위 법안임
    for bill in selected_bills_data:
        domain = bill["estimated_domain"]
        if domain not in domain_to_best_bill:
            domain_to_best_bill[domain] = bill
            
    # 도메인별 1개씩 총 10개 법안 정렬 및 수집
    sorted_quick_bills = sorted(domain_to_best_bill.values(), key=lambda x: x["priority_rank"])
    for i, bill in enumerate(sorted_quick_bills, 1):
        quick_subset.append({
            "priority_rank": i,
            "bill_id": bill["bill_id"],
            "bill_name": bill["bill_name"],
            "reason": f"추정 분야 '{bill['estimated_domain']}'을 대표하는 우선 평가 대상 법안 ({bill['difficulty']} 난이도)"
        })
        
    # 5. 분포 요약 생성
    manual_category_dist = {cat: count for cat, count in selected_categories_count.items()}
    
    notes = (
        f"유효 법안 {len(valid_bills)}개 중 평가용 소스 법안 20개를 자동 선정했습니다. "
        f"10대 도메인 분포를 {min(selected_domains_count.values())}~{max(selected_domains_count.values())}개 범위로 유지하여 분야의 다변화를 달성했으며, "
        f"난이도는 easy {selected_difficulties_count['easy']}개, medium {selected_difficulties_count['medium']}개, hard {selected_difficulties_count['hard']}개로 적절히 분배되었습니다."
    )
    
    output_json = {
        "selected_source_bills": selected_bills_data,
        "quick_10_subset": quick_subset,
        "coverage_summary": {
            "domain_distribution": selected_domains_count,
            "manual_category_distribution": manual_category_dist,
            "difficulty_distribution": selected_difficulties_count,
            "notes": notes
        }
    }
    
    # 결과 저장
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=2)
        
    # 콘솔 출력 요구사항 구현
    print("=== 법안 선정 결과 요약 ===")
    print(f"전체 법안 수: {total_bills}")
    print(f"선정된 source 법안 수: {len(selected_bills_data)}")
    print(f"quick subset 수: {len(quick_subset)}")
    print("\n[추정 도메인(domain) 분포]")
    for domain, count in selected_domains_count.items():
        print(f"- {domain}: {count}개")
    print("\n[난이도(difficulty) 분포]")
    for diff, count in selected_difficulties_count.items():
        print(f"- {diff}: {count}개")
    print(f"\n출력 파일 경로: {output_path}")


if __name__ == '__main__':
    select_source_bills()
