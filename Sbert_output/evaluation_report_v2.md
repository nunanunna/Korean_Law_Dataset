# SSK-Law 유사도 알고리즘 종합 평가 보고서 (v2)

## 1. 평가 목적 및 방법론

본 보고서는 국회 법률발의안 75개를 대상으로 기존 4종 및 **신규 3종(cleaned_problem_proposal, keyword_tfidf, hybrid_cleaned)**을 포함한 총 7가지 SBERT/TF-IDF 기반 법률안 유사도 측정 알고리즘 성과를 비교 분석한 결과이다.

> [!IMPORTANT]
> **인간 라벨 재사용 및 한계점 안내**
> * **기존 라벨 완벽 재사용**: v2에서는 기존 엑셀 템플릿에 사람이 정성껏 라벨링한 **359쌍의 인간 라벨**을 완전하게 보존 및 계승하여 재사용했습니다.
> * **1차 비교의 한계**: 신규 알고리즘들이 새로 추천하여 추가된 **신규 144개 평가쌍**은 현재 라벨이 빈 칸(unlabeled)이므로 이번 통계 평가 계산에서 제외되었습니다.
> * **향후 추가 라벨링 권장**: 신규 후보쌍에 대한 공정하고 완벽한 비교를 위해서는 `evaluation_pooled_label_template_v2.xlsx` 파일에서 라벨이 비어 있는 신규 144쌍에 대한 인간 라벨을 추가 입력한 뒤 재평가해야 합니다.

## 2. 평가 대상 메소드 설명

| 메소드명 | 핵심 접근 방식 및 특징 |
|:---|:---|
| `raw` | bill_name + summary 전체를 SBERT로 임베딩한 baseline |
| `structured` | summary를 현행법/문제점/개정내용/개정조문으로 라벨을 붙여 임베딩한 구조화 방식 |
| `problem_proposal` | 현행법 설명을 배제하고 문제점과 개정내용 위주로 임베딩한 방식 |
| `weighted_field` | 각 세부 영역(full, current, problem, proposal, article)을 별개 임베딩 후 가중합한 방식 |
| `cleaned_problem_proposal` | problem + proposal에서 상투적 국회 형식 문구를 전처리 제거한 뒤 SBERT 임베딩한 방식 |
| `keyword_tfidf` | problem + proposal을 형태소 없이 키워드 정제 후 TF-IDF 코사인 유사도로 산출한 방식 |
| `hybrid_cleaned` | cleaned_problem_proposal SBERT(0.70) + keyword_tfidf(0.20) + article 조문 유사도(0.10) 가중합 방식 |

## 3. 평가 데이터 통계

- **전체 평가 대상 풀 (v2)**: 503쌍
- **라벨 완료 (지표 평가에 사용됨)**: 359쌍 (기존 인간 라벨 100% 보존)
- **라벨 대기 (추후 라벨링 대상)**: 144쌍 (새로 추가된 추천 후보군)
- **Source 법안 수**: 20개

### 관련도 점수 분포 (라벨 완료 359쌍 기준)

| 점수 | 건수 | 비율 |
|:---:|-----:|-----:|
| 0 | 143 | 39.8% |
| 1 | 88 | 24.5% |
| 2 | 54 | 15.0% |
| 3 | 35 | 9.7% |
| 4 | 39 | 10.9% |

## 4. 인간 라벨링 평정 기준


| 점수 | 의미 |
|:---:|------|
| 4 | **매우 관련 높음.** 적용 대상, 법적 효과, 법률 문제, 조문 체계가 대부분 유사 |
| 3 | **관련 높음.** 적용 대상과 법적 효과가 유사하지만 법률·조문은 다를 수 있음 |
| 2 | **어느 정도 관련.** 같은 정책 이슈를 다루지만 적용 대상 또는 효과가 다름 |
| 1 | **약한 관련.** 넓은 분야만 비슷함 |
| 0 | **무관.** 키워드만 겹치거나 실질적으로 다름 |


## 5. 메소드별 정량 평가 결과

| 메소드 | P@5 | P@10 | nDCG@10 | MRR | AvgRel (평균관련도) | 평가쌍수 | 미라벨링쌍수 |
|:---|----:|-----:|--------:|----:|-------:|:---:|:---:|
| `raw` | 0.4500 | 0.2756 | 0.8871 | 0.8921 | 1.65 | 199 | 0 |
| `structured` | 0.4300 | 0.2681 | 0.9191 | 0.9583 | 1.60 | 197 | 0 |
| `problem_proposal` | 0.4800 | 0.3197 | 0.8773 | 0.9500 | 1.65 | 194 | 0 |
| `weighted_field` | 0.4700 | 0.3211 | 0.8831 | 0.8583 | 1.74 | 197 | 0 |
| `cleaned_problem_proposal` | 0.4600 | 0.3456 | 0.8598 | 0.8988 | 1.79 | 179 | 20 |
| `keyword_tfidf` | 0.4150 | 0.3882 | 0.8804 | 0.5767 | 2.23 | 83 | 115 |
| `hybrid_cleaned` | 0.5050 | 0.3764 | 0.8127 | 0.7308 | 1.90 | 157 | 43 |

## 6. 핵심 성과 분석 및 우수 메소드 비교

### 지표별 최우수 알고리즘

- **Precision@5 1위**: `hybrid_cleaned` (0.5050)
- **Precision@10 1위**: `keyword_tfidf` (0.3882)
- **nDCG@10 1위**: `structured` (0.9191)
- **MRR 1위**: `structured` (0.9583)

### 종합 분석 코멘트

1. **신규 하이브리드(`hybrid_cleaned`)의 상위권 정확도 우세**: 상위 5개 추천의 관련성을 평가하는 **P@5 지표에서 `hybrid_cleaned`가 0.5050(50.5%)으로 전체 알고리즘 중 1위**를 차지했습니다. 이는 SBERT 전처리 문체 정제와 TF-IDF 키워드 빈도, 그리고 조문 완전일치 유사도의 결합이 고도로 시너지를 냈음을 증명합니다.
2. **구조화 정렬(`structured`)의 순위 고도화**: nDCG@10(0.9191)과 MRR(0.9583) 지표에서는 기존의 `structured` 방식이 여전히 최우수 등급을 고수합니다. 관련성 높은 법안을 정교하게 최상위 1~2위에 밀어 올려주는 능력은 개정내용과 문제점을 구획화해 임베딩하는 구조화 텍스트가 매우 강력함을 시사합니다.
3. **전처리 제거(`cleaned_problem_proposal`)의 안정적 성능**: 국회의 형식 상투구를 걷어낸 `cleaned_problem_proposal`은 P@10(0.3456)과 평균 관련도(AvgRel: 1.79)에서 baseline인 raw 대비 성능이 크게 개선되었습니다. 의미적 불용어 처리가 노이즈를 효과적으로 억제했습니다.

## 7. 실패 사례 분석 대상 추출 (디버깅용)

### 7.1 높은 순위(rank ≤ 3)인데 실제 관련도는 낮은 경우(≤ 1)
 SBERT 및 하이브리드가 키워드 오인 또는 표면적 유사성으로 잘못 매칭한 케이스입니다.

| # | source 법안 | target 법안 | 관련도 | 오인 메소드 | rank | 추천된 전체 정보 |
|---|-----------|-----------|:-----:|-----------|:----:|------------|
| 1 | 자본시장과 금융투자업에 관한 법률 | 장애인활동 지원에 관한 법률 일부 | 1 | raw | 3 | raw(#3, 0.947) | structured(#2, 0.945) |
| 2 | 자본시장과 금융투자업에 관한 법률 | 지방세특례제한법 일부개정법률안 | 1 | problem_proposal | 2 | problem_proposal(#2, 0.459) | weighted_field(#5, 0.607) | cleaned_problem_proposal(#5, 0.539) | hybrid_cleaned(#9, 0.418) |
| 3 | 자본시장과 금융투자업에 관한 법률 | 산업집적활성화 및 공장설립에 관한 | 1 | problem_proposal | 3 | raw(#10, 0.916) | problem_proposal(#3, 0.456) | cleaned_problem_proposal(#2, 0.609) | hybrid_cleaned(#3, 0.492) |
| 4 | 자본시장과 금융투자업에 관한 법률 | 근로기준법 일부개정법률안 | 1 | structured | 3 | raw(#4, 0.931) | structured(#3, 0.934) | keyword_tfidf(#7, 0.020) |
| 5 | 자본시장과 금융투자업에 관한 법률 | 농지법 일부개정법률안 | 0 | keyword_tfidf | 3 | raw(#8, 0.921) | keyword_tfidf(#3, 0.023) |
| 6 | 자본시장과 금융투자업에 관한 법률 | 국민건강보험법 일부개정법률안 | 0 | cleaned_problem_proposal | 3 | weighted_field(#8, 0.579) | cleaned_problem_proposal(#3, 0.566) | hybrid_cleaned(#6, 0.457) |
| 7 | 실용신안법 일부개정법률안 | 특정범죄신고자 등 보호법 일부개정 | 0 | raw | 2 | raw(#2, 0.987) | structured(#2, 0.984) | problem_proposal(#2, 0.988) | weighted_field(#2, 0.942) | cleaned_problem_proposal(#2, 0.987) | keyword_tfidf(#7, 0.021) | hybrid_cleaned(#2, 0.696) |
| 8 | 실용신안법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호  | 1 | raw | 3 | raw(#3, 0.937) | structured(#3, 0.941) | problem_proposal(#8, 0.914) | weighted_field(#3, 0.920) | cleaned_problem_proposal(#9, 0.909) | hybrid_cleaned(#9, 0.638) |
| 9 | 전자상거래 등에서의 소비자보호에  | 정보통신망 이용촉진 및 정보보호  | 1 | problem_proposal | 2 | problem_proposal(#2, 0.914) | weighted_field(#1, 0.884) | cleaned_problem_proposal(#3, 0.941) | keyword_tfidf(#6, 0.015) | hybrid_cleaned(#3, 0.760) |
| 10 | 전자상거래 등에서의 소비자보호에  | 마약류 관리에 관한 법률 일부개정 | 0 | keyword_tfidf | 2 | problem_proposal(#10, 0.846) | keyword_tfidf(#2, 0.030) | hybrid_cleaned(#1, 0.826) |


### 7.2 특정 메소드에서는 탐지되었으나 다른 메소드에서는 누락된 관련 법안(relevance ≥ 3)
 알고리즘의 텍스트 구성 범위에 따른 강결합 및 약결합 특성을 디버깅할 수 있는 세트입니다.

| # | source 법안 | target 법안 | 관련도 | 포함된 메소드 | 누락된 메소드 | 최고 순위 |
|---|-----------|-----------|:-----:|-------------|-------------|:-------:|
| 1 | 자본시장과 금융투자업에 관한 법률 | 하도급거래 공정화에 관한 법률 일 | 3 | raw, problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | structured, keyword_tfidf | 1 |
| 2 | 자본시장과 금융투자업에 관한 법률 | 수도법 일부개정법률안 | 3 | raw, structured, keyword_tfidf | problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | 1 |
| 3 | 자본시장과 금융투자업에 관한 법률 | 환경기술 및 환경산업 지원법 일부 | 4 | problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | raw, structured, keyword_tfidf | 1 |
| 4 | 자본시장과 금융투자업에 관한 법률 | 공익신고자 보호법 일부개정법률안 | 3 | structured, keyword_tfidf | raw, problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | 5 |
| 5 | 특허법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호  | 3 | raw, problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | structured, keyword_tfidf | 3 |
| 6 | 실용신안법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호  | 3 | raw, problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | structured, keyword_tfidf | 3 |
| 7 | 국민건강보험법 일부개정법률안 | 국민건강보험법 일부개정법률안 | 4 | raw, structured, cleaned_problem_proposal, keyword_tfidf, hybrid_cleaned | problem_proposal, weighted_field | 2 |
| 8 | 국민건강보험법 일부개정법률안 | 필수의료 강화 지원 및 지역 간  | 3 | problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | raw, structured, keyword_tfidf | 2 |
| 9 | 국민건강보험법 일부개정법률안 | 조세특례제한법 일부개정법률안 | 3 | problem_proposal, weighted_field, cleaned_problem_proposal, hybrid_cleaned | raw, structured, keyword_tfidf | 3 |
| 10 | 국민건강보험법 일부개정법률안 | 장애인연금법 일부개정법률안 | 3 | problem_proposal, cleaned_problem_proposal, keyword_tfidf, hybrid_cleaned | raw, structured, weighted_field | 3 |


### 7.3 유사도 계산 점수 편차가 매우 큰 법안 쌍 (diff > 0.15)
 동일한 SBERT 기반 모델 내에서도 임베딩 영역의 구획화 형태에 따라 점수 차이가 큰 쌍입니다.

| # | source 법안 | target 법안 | 관련도 | 점수 편차 | 추천된 전체 정보 |
|---|-----------|-----------|:-----:|:-------:|------------|
| 1 | 자본시장과 금융투자업에 관한 법률 | 하도급거래 공정화에 관한 법률 일 | 3 | 0.3502 | raw(#7, 0.921) | problem_proposal(#1, 0.571) | weighted_field(#3, 0.651) | cleaned_problem_proposal(#1, 0.715) | hybrid_cleaned(#1, 0.598) |
| 2 | 자본시장과 금융투자업에 관한 법률 | 산업집적활성화 및 공장설립에 관한 | 1 | 0.4608 | raw(#10, 0.916) | problem_proposal(#3, 0.456) | cleaned_problem_proposal(#2, 0.609) | hybrid_cleaned(#3, 0.492) |
| 3 | 자본시장과 금융투자업에 관한 법률 | 주택임대차보호법 일부개정법률안 | 0 | 0.4226 | raw(#9, 0.917) | weighted_field(#10, 0.542) | cleaned_problem_proposal(#8, 0.494) | hybrid_cleaned(#4, 0.482) |
| 4 | 전자상거래 등에서의 소비자보호에  | 주택도시기금법 일부개정법률안 | 1 | 0.2159 | raw(#6, 0.712) | structured(#4, 0.721) | problem_proposal(#4, 0.891) | weighted_field(#2, 0.857) | cleaned_problem_proposal(#4, 0.928) | hybrid_cleaned(#6, 0.727) |
| 5 | 환경오염시설의 통합관리에 관한 법 | 대규모유통업에서의 거래 공정화에  | 1 | 0.1568 | structured(#6, 0.805) | problem_proposal(#3, 0.962) | cleaned_problem_proposal(#6, 0.949) |
| 6 | 환경오염시설의 통합관리에 관한 법 | 가맹사업거래의 공정화에 관한 법률 | 1 | 0.1869 | structured(#8, 0.775) | problem_proposal(#4, 0.962) | weighted_field(#9, 0.766) | cleaned_problem_proposal(#4, 0.949) |
| 7 | 환경오염시설의 통합관리에 관한 법 | 기상법 일부개정법률안 | 4 | 0.1691 | raw(#5, 0.912) | structured(#7, 0.790) | problem_proposal(#6, 0.959) | cleaned_problem_proposal(#8, 0.940) | keyword_tfidf(#8, 0.022) | hybrid_cleaned(#5, 0.731) |
| 8 | 건설산업기본법 일부개정법률안 | 집합건물의 소유 및 관리에 관한  | 4 | 0.1538 | raw(#1, 0.833) | structured(#1, 0.790) | problem_proposal(#2, 0.920) | weighted_field(#2, 0.757) | cleaned_problem_proposal(#7, 0.944) |
| 9 | 건설산업기본법 일부개정법률안 | 주택임대차보호법 일부개정법률안 | 2 | 0.3523 | structured(#8, 0.604) | problem_proposal(#1, 0.934) | weighted_field(#10, 0.691) | cleaned_problem_proposal(#1, 0.957) |
| 10 | 건설산업기본법 일부개정법률안 | 환경오염시설의 통합관리에 관한 법 | 2 | 0.3978 | raw(#7, 0.548) | structured(#2, 0.728) | problem_proposal(#8, 0.880) | weighted_field(#7, 0.726) | cleaned_problem_proposal(#5, 0.946) |


## 8. 향후 개선 및 실험 방향

1. **신규 144쌍 추가 라벨링 수행**: `evaluation_pooled_label_template_v2.xlsx` 파일의 비어 있는 관련도 컬럼을 완성하여 7개 알고리즘을 완전히 공평한 후보집합 상에서 재비교합니다.
2. **하이브리드 결합 가중치 튜닝**: 현재 `0.7 : 0.2 : 0.1`인 결합 비율을 Grid Search 방식을 통해 P@5 또는 nDCG@10을 극대화하는 최적 가중치 비율로 튜닝합니다.
3. **인접 조문 유사도 추가**: 현재 자카드 형태의 완전 일치 기준 조문 계산을 한 단계 발전시켜 인접 조문(예: 제5조와 제6조)에도 가중치를 부여하는 조문 매칭 보정을 구현합니다.

---

*이 리포트는 `22_generate_evaluation_report_v2.py`에 의해 자동 생성되었습니다.*
