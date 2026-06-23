# SSK-Law 유사도 알고리즘 평가 리포트

## 1. 평가 목적

본 리포트는 국회 법률발의안 75개를 대상으로 여러 SBERT 기반 유사도 알고리즘의 성과를 비교·분석한 결과를 정리한 것이다.
사람이 직접 0~4점 관련도 라벨을 부여한 뒤, 메소드별 Precision, nDCG, MRR 등 표준 IR 지표를 계산하여 어떤 방식이 법률 유사도 측정에 가장 효과적인지 판단한다.

## 2. 평가 대상 메소드

| 메소드 | 설명 |
|--------|------|
| `raw` | bill_name + summary 전체를 그대로 SBERT 임베딩한 baseline |
| `structured` | summary를 현행법, 문제점, 개정내용, 개정조문으로 분리하고 라벨을 붙여 임베딩한 방식 |
| `problem_proposal` | 현행법 설명을 제외하고 문제점과 개정내용 중심으로 임베딩한 방식 |
| `weighted_field` | full_text, current_law, problem, proposal, article_text를 각각 임베딩하고 가중합한 방식 |

## 3. 평가 데이터 구성

- **전체 평가쌍 수**: 359쌍
- **라벨 입력 완료**: 359쌍
- **source 법안 수**: 20개
- **평가 방식**: 4개 메소드의 top-10 결과를 pooling하여 중복 제거 후 source당 최대 20개 후보 선정

### 관련도 점수 분포

| 점수 | 건수 | 비율 |
|:---:|-----:|-----:|
| 0 | 143 | 39.8% |
| 1 | 88 | 24.5% |
| 2 | 54 | 15.0% |
| 3 | 35 | 9.7% |
| 4 | 39 | 10.9% |

## 4. 라벨링 기준


| 점수 | 의미 |
|:---:|------|
| 4 | **매우 관련 높음.** 적용 대상, 법적 효과, 법률 문제, 조문 체계가 대부분 유사 |
| 3 | **관련 높음.** 적용 대상과 법적 효과가 유사하지만 법률·조문은 다를 수 있음 |
| 2 | **어느 정도 관련.** 같은 정책 이슈를 다루지만 적용 대상 또는 효과가 다름 |
| 1 | **약한 관련.** 넓은 분야만 비슷함 |
| 0 | **무관.** 키워드만 겹치거나 실질적으로 다름 |


## 5. 메소드별 정량 평가 결과

| 메소드 | P@5 | P@10 | nDCG@10 | MRR | AvgRel | Legal Meaning |
|--------|----:|-----:|--------:|----:|-------:|--------------:|
| `raw` | 0.4500 | 0.2756 | 0.8871 | 0.8921 | 1.65 | nan |
| `structured` | 0.4300 | 0.2681 | 0.9191 | 0.9583 | 1.60 | nan |
| `problem_proposal` | 0.4800 | 0.3197 | 0.8773 | 0.9500 | 1.65 | nan |
| `weighted_field` | 0.4700 | 0.3211 | 0.8831 | 0.8583 | 1.74 | nan |

## 6. 가장 성능이 좋은 메소드

종합 성능 기준 **`problem_proposal`** 메소드가 가장 우수한 것으로 나타났다.

- **설명**: 현행법 설명을 제외하고 문제점과 개정내용 중심으로 임베딩한 방식
- **nDCG@10**: 0.8773
- **MRR**: 0.9500

## 7. 지표별 해석

### Precision@K
상위 K개 추천 결과 중 관련도 3점 이상인 법안의 비율이다. 사용자가 상위 결과만 보는 시나리오에서의 실용적 성능을 나타낸다.

### nDCG@10
Normalized Discounted Cumulative Gain. 관련도 점수를 순위 가중치와 함께 반영하여 순위 품질을 종합적으로 평가한다. 가장 중요한 지표로 볼 수 있다.

### MRR (Mean Reciprocal Rank)
처음으로 관련 법안(관련도 ≥ 3)이 등장하는 순위의 역수 평균이다. 사용자가 첫 번째 관련 결과를 얼마나 빨리 찾을 수 있는지를 나타낸다.

### Average Relevance
추천 결과 전체의 평균 관련도 점수이다. 전체적인 추천 품질을 나타내는 보조 지표이다.

### Legal Meaning Score
법률적 의미 유사성을 세부 항목(이슈 일치, 대상 일치, 효과 일치, 범위 일치, 조문 일치)별로 평가한 복합 점수이다. 0~100점 범위이다.

## 8. 실패 사례 분석 후보

### 8.1 높은 순위(rank ≤ 3)인데 관련도가 낮은 경우(≤ 1)

| # | source 법안 | target 법안 | 관련도 | 문제 메소드 | rank | 메소드별 정보 |
|---|-----------|-----------|:-----:|-----------|:----:|------------|
| 1 | 자본시장과 금융투자업에 관한 법률 일 | 장애인활동 지원에 관한 법률 일부개정 | 1 | raw | 3 | raw(#3, 0.9466) | structured(#2, 0.9449) | problem_proposal(N/A, N/A) | weighted_field(N/A, N/A) |
| 2 | 자본시장과 금융투자업에 관한 법률 일 | 지방세특례제한법 일부개정법률안 | 1 | problem_proposal | 2 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#2, 0.4587) | weighted_field(#5, 0.6069) |
| 3 | 자본시장과 금융투자업에 관한 법률 일 | 근로기준법 일부개정법률안 | 1 | structured | 3 | raw(#4, 0.9314) | structured(#3, 0.9342) | problem_proposal(N/A, N/A) | weighted_field(N/A, N/A) |
| 4 | 자본시장과 금융투자업에 관한 법률 일 | 산업집적활성화 및 공장설립에 관한 법 | 1 | problem_proposal | 3 | raw(#10, 0.9164) | structured(N/A, N/A) | problem_proposal(#3, 0.4556) | weighted_field(N/A, N/A) |
| 5 | 실용신안법 일부개정법률안 | 특정범죄신고자 등 보호법 일부개정법률 | 0 | raw | 2 | raw(#2, 0.9868) | structured(#2, 0.9842) | problem_proposal(#2, 0.9883) | weighted_field(#2, 0.9415) |
| 6 | 실용신안법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호 등에 | 1 | raw | 3 | raw(#3, 0.9371) | structured(#3, 0.9414) | problem_proposal(#8, 0.9143) | weighted_field(#3, 0.9195) |
| 7 | 전자상거래 등에서의 소비자보호에 관한 | 주택도시기금법 일부개정법률안 | 1 | weighted_field | 2 | raw(#6, 0.7117) | structured(#4, 0.7209) | problem_proposal(#4, 0.8915) | weighted_field(#2, 0.8570) |
| 8 | 전자상거래 등에서의 소비자보호에 관한 | 정보통신망 이용촉진 및 정보보호 등에 | 1 | problem_proposal | 2 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#2, 0.9140) | weighted_field(#1, 0.8841) |
| 9 | 전자상거래 등에서의 소비자보호에 관한 | 중소기업진흥에 관한 법률 일부개정법률 | 1 | problem_proposal | 3 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#3, 0.8965) | weighted_field(#8, 0.7985) |
| 10 | 주택임대차보호법 일부개정법률안 | 하도급거래 공정화에 관한 법률 일부개 | 0 | raw | 1 | raw(#1, 0.9572) | structured(#6, 0.8664) | problem_proposal(N/A, N/A) | weighted_field(#7, 0.7866) |

> 분석 메모: (수동 기입 필요)

### 8.2 특정 메소드에서 누락되었지만 관련도 높은 경우(≥ 3)

| # | source 법안 | target 법안 | 관련도 | 포함된 메소드 | 누락된 메소드 | 최고 순위 |
|---|-----------|-----------|:-----:|-------------|-------------|:-------:|
| 1 | 자본시장과 금융투자업에 관한 법률 일 | 하도급거래 공정화에 관한 법률 일부개 | 3 | raw, problem_proposal, weighted_field | structured | 1 |
| 2 | 자본시장과 금융투자업에 관한 법률 일 | 수도법 일부개정법률안 | 3 | raw, structured | problem_proposal, weighted_field | 1 |
| 3 | 자본시장과 금융투자업에 관한 법률 일 | 환경기술 및 환경산업 지원법 일부개정 | 4 | problem_proposal, weighted_field | raw, structured | 1 |
| 4 | 특허법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호 등에 | 3 | raw, problem_proposal, weighted_field | structured | 4 |
| 5 | 실용신안법 일부개정법률안 | 정보통신망 이용촉진 및 정보보호 등에 | 3 | raw, problem_proposal, weighted_field | structured | 4 |
| 6 | 국민건강보험법 일부개정법률안 | 국민건강보험법 일부개정법률안 | 4 | raw, structured, problem_proposal | weighted_field | 5 |
| 7 | 국민건강보험법 일부개정법률안 | 국민건강보험법 일부개정법률안 | 4 | raw, structured | problem_proposal, weighted_field | 2 |
| 8 | 국민건강보험법 일부개정법률안 | 필수의료 강화 지원 및 지역 간 의료 | 3 | problem_proposal, weighted_field | raw, structured | 2 |
| 9 | 국민건강보험법 일부개정법률안 | 조세특례제한법 일부개정법률안 | 3 | problem_proposal, weighted_field | raw, structured | 3 |
| 10 | 국민건강보험법 일부개정법률안 | 장애인연금법 일부개정법률안 | 3 | problem_proposal | raw, structured, weighted_field | 5 |

> 분석 메모: (수동 기입 필요)

### 8.3 메소드 간 점수 차이가 큰 경우(diff > 0.15)

| # | source 법안 | target 법안 | 관련도 | 점수 차이 | 메소드별 정보 |
|---|-----------|-----------|:-----:|:-------:|------------|
| 1 | 자본시장과 금융투자업에 관한 법률 일 | 하도급거래 공정화에 관한 법률 일부개 | 3 | 0.3502 | raw(#7, 0.9211) | structured(N/A, N/A) | problem_proposal(#1, 0.5709) | weighted_field(#3, 0.6510) |
| 2 | 자본시장과 금융투자업에 관한 법률 일 | 환경기술 및 환경산업 지원법 일부개정 | 4 | 0.2407 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#5, 0.4421) | weighted_field(#1, 0.6829) |
| 3 | 자본시장과 금융투자업에 관한 법률 일 | 개인정보 보호법 일부개정법률안 | 2 | 0.2316 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#4, 0.4506) | weighted_field(#2, 0.6822) |
| 4 | 자본시장과 금융투자업에 관한 법률 일 | 산업집적활성화 및 공장설립에 관한 법 | 1 | 0.4608 | raw(#10, 0.9164) | structured(N/A, N/A) | problem_proposal(#3, 0.4556) | weighted_field(N/A, N/A) |
| 5 | 자본시장과 금융투자업에 관한 법률 일 | 농촌융복합산업 육성 및 지원에 관한  | 0 | 0.1676 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#6, 0.4421) | weighted_field(#4, 0.6098) |
| 6 | 자본시장과 금융투자업에 관한 법률 일 | 주택임대차보호법 일부개정법률안 | 0 | 0.3747 | raw(#9, 0.9165) | structured(N/A, N/A) | problem_proposal(N/A, N/A) | weighted_field(#10, 0.5418) |
| 7 | 국민건강보험법 일부개정법률안 | 기초연금법 일부개정법률안 | 3 | 0.1506 | raw(#8, 0.8916) | structured(N/A, N/A) | problem_proposal(#9, 0.8149) | weighted_field(#8, 0.7410) |
| 8 | 국민건강보험법 일부개정법률안 | 조세특례제한법 일부개정법률안 | 3 | 0.1971 | raw(N/A, N/A) | structured(N/A, N/A) | problem_proposal(#3, 0.9177) | weighted_field(#10, 0.7205) |
| 9 | 전자상거래 등에서의 소비자보호에 관한 | 주택도시기금법 일부개정법률안 | 1 | 0.1798 | raw(#6, 0.7117) | structured(#4, 0.7209) | problem_proposal(#4, 0.8915) | weighted_field(#2, 0.8570) |
| 10 | 주택임대차보호법 일부개정법률안 | 농지법 일부개정법률안 | 2 | 0.1534 | raw(#6, 0.9487) | structured(#9, 0.8469) | problem_proposal(#8, 0.9331) | weighted_field(#6, 0.7953) |

> 분석 메모: (수동 기입 필요)

## 9. 향후 개선 방향

1. **라벨 데이터 확장**: 평가 source 법안을 20개에서 전체 75개로 확대하여 통계적 신뢰도를 높인다.
2. **Legal Meaning Score 활용**: 세부 항목(이슈/대상/효과/범위/조문)별 가중치를 튜닝하여 법률 도메인 특화 지표를 정교하게 개선한다.
3. **하이브리드 방식 탐구**: 실패 사례 분석 결과를 토대로 메소드 간 앙상블 전략을 설계한다.
4. **임베딩 모델 실험**: ko-sbert 외에 legal-BERT 등 법률 도메인 특화 모델을 실험한다.
5. **요약문 구조 파싱 개선**: 현행법/문제점/개정내용 구분의 정확도를 높여 structured/problem_proposal 방식의 성능을 개선한다.

---

*이 리포트는 `16_generate_evaluation_report.py`에 의해 자동 생성되었습니다.*
