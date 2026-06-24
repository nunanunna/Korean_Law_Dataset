# 법률발의안 유사도 알고리즘 가중치 최적화 결과 보고서

> **평가 기준:** gold/LLM 혼합 라벨 기준 (75개 법안 전체셋)
> **추가 비교:** Gold-only 라벨 기준 (20개 법안 서브셋)
> **Component Normalization:** Weighted Field (True), Final Ensemble (True) (Method: row_minmax)

## 1. 최적 가중치 요약

### A. Weighted Field 최적 가중치
| Objective | w_title | w_full | w_current | w_problem | w_proposal | w_article | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **best_by_objective_score** | 0.20 | 0.20 | 0.10 | 0.20 | 0.20 | 0.10 | 0.5399 |
| **best_by_candidate_recall_objective** | 0.20 | 0.15 | 0.10 | 0.20 | 0.25 | 0.10 | 0.4529 |
| **best_by_ranking_objective** | 0.20 | 0.20 | 0.05 | 0.20 | 0.25 | 0.10 | 0.6570 |

### B. Final Ensemble 최적 가중치
| Objective | w_raw | w_structured | w_problem_proposal | w_weighted_field | w_cleaned | w_hybrid | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **best_by_objective_score** | 0.10 | 0.10 | 0.10 | 0.20 | 0.05 | 0.45 | 0.5640 |
| **best_by_candidate_recall_objective** | 0.10 | 0.10 | 0.10 | 0.20 | 0.05 | 0.45 | 0.4615 |
| **best_by_ranking_objective** | 0.10 | 0.10 | 0.10 | 0.20 | 0.05 | 0.45 | 0.6704 |

## 2. 전체 성능 비교표 (75개 법안 혼합 라벨 전체셋)

| Method | P@5 | P@10 | nDCG@10 | MRR | Avg Relevance | Avg LMS | Objective Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.2933 | 0.2040 | 0.8039 | 0.6298 | 1.3320 | 33.49 | 0.5138 |
| structured | 0.3093 | 0.2040 | 0.8111 | 0.6657 | 1.3053 | 32.6 | 0.5311 |
| problem_proposal | 0.3307 | 0.2027 | 0.7800 | 0.6113 | 1.2587 | 30.49 | 0.5216 |
| cleaned_problem_proposal | 0.3280 | 0.2240 | 0.7729 | 0.5979 | 1.3320 | 32.86 | 0.5154 |
| hybrid_cleaned_original | 0.3333 | 0.2218 | 0.7721 | 0.5848 | 1.3324 | 33.02 | 0.5152 |
| hybrid_cleaned_best | 0.3413 | 0.2255 | 0.7951 | 0.6140 | 1.3592 | 33.67 | 0.5320 |
| weighted_field_original | 0.2907 | 0.2077 | 0.7544 | 0.5266 | 1.2955 | 31.75 | 0.4770 |
| **weighted_field_best** | 0.3360 | 0.2394 | 0.8183 | 0.6318 | 1.4294 | 35.87 | 0.5399 |
| **final_ensemble_best** | 0.3787 | 0.2496 | 0.8121 | 0.6549 | 1.4469 | 36.19 | 0.5640 |

## 3. Gold-only 성능 비교표 (20개 법안 골드 라벨)

| Method | P@5 | P@10 | nDCG@10 | MRR | Avg Relevance | Avg LMS | Objective Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.4500 | 0.2750 | 0.8871 | 0.8921 | 1.6450 | 41.52 | 0.6695 |
| structured | 0.4300 | 0.2650 | 0.9201 | 0.9583 | 1.5950 | 39.38 | 0.6827 |
| problem_proposal | 0.4800 | 0.3100 | 0.8771 | 0.9500 | 1.6150 | 40.15 | 0.6931 |
| cleaned_problem_proposal | 0.4600 | 0.3150 | 0.8573 | 0.8988 | 1.6800 | 42.4 | 0.6670 |
| hybrid_cleaned_original | 0.5000 | 0.3300 | 0.8039 | 0.7030 | 1.6800 | 42.6 | 0.6318 |
| hybrid_cleaned_best | 0.5000 | 0.3200 | 0.8392 | 0.8167 | 1.7150 | 43.61 | 0.6651 |
| weighted_field_original | 0.4600 | 0.3101 | 0.8898 | 0.8806 | 1.7041 | 42.39 | 0.6731 |
| weighted_field_best | 0.5100 | 0.3451 | 0.9088 | 0.9167 | 1.8901 | 47.6 | 0.7110 |
| final_ensemble_best | 0.5300 | 0.3567 | 0.9280 | 0.9600 | 1.8680 | 47.18 | 0.7354 |

## 4. Train / Validation 검증 결과

### A. Fold-0 80/20 Split 검증
- **Weighted Field**:
  - Train 최적 가중치: {'w_title': 0.2, 'w_full': 0.25, 'w_current': 0.1, 'w_problem': 0.2, 'w_proposal': 0.2, 'w_article': 0.05}
  - Validation P@5: 0.3467, nDCG@10: 0.7950, MRR: 0.6206, Objective: 0.5360
- **Final Ensemble**:
  - Train 최적 가중치: {'w_raw': 0.0, 'w_structured': 0.1, 'w_problem_proposal': 0.1, 'w_weighted_field': 0.2, 'w_cleaned': 0.15, 'w_hybrid': 0.45}
  - Validation P@5: 0.3600, nDCG@10: 0.7726, MRR: 0.6467, Objective: 0.5411

### B. 5-Fold Cross Validation 평균 성능
- Weighted Field 평균 Validation Objective Score: **0.5318**
- Final Ensemble 평균 Validation Objective Score: **0.5536**

## 5. Component Normalization 상세 통계

| Component | Pre-Min | Pre-Max | Pre-Mean | Post-Min | Post-Max | Post-Mean |
| --- | --- | --- | --- | --- | --- | --- |
| WF Title Law Name | -0.0767 | 1.0000 | 0.2291 | 0.0000 | 1.0000 | 0.3002 |
| WF Full Text | -0.0178 | 0.9983 | 0.6125 | 0.0000 | 1.0000 | 0.6408 |
| WF Current Law | -0.0054 | 0.9963 | 0.5114 | 0.0000 | 1.0000 | 0.4983 |
| WF Problem | -0.0193 | 1.0000 | 0.5292 | 0.0000 | 1.0000 | 0.5056 |
| WF Proposal | -0.0319 | 1.0000 | 0.5206 | 0.0000 | 1.0000 | 0.5218 |
| WF Article Jaccard | 0.0000 | 1.0000 | 0.0075 | 0.0000 | 1.0000 | 0.0110 |
| FE Raw Score | -0.0363 | 0.9985 | 0.6024 | 0.0000 | 1.0000 | 0.6419 |
| FE Structured Score | -0.0809 | 0.9976 | 0.5132 | 0.0000 | 1.0000 | 0.5677 |
| FE Problem Proposal Score | -0.0373 | 1.0000 | 0.6478 | 0.0000 | 1.0000 | 0.6610 |
| FE Weighted Field Best Score | 0.0936 | 1.0000 | 0.4446 | 0.0000 | 1.0000 | 0.4210 |
| FE Cleaned Problem Proposal Score | 0.0184 | 1.0000 | 0.6952 | 0.0000 | 1.0000 | 0.6776 |
| FE Hybrid Cleaned Best Score | 0.0371 | 1.0000 | 0.6417 | 0.0000 | 1.0000 | 0.6364 |
