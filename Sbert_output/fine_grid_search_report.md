# 2차 Fine Grid Search 보고서

## 1. 실험 목적

1차 최적 가중치 주변을 0.025 간격으로 더 촘촘하게 탐색했다. 특히 최적 가중치가 탐색 범위 경계값에 붙어 있었기 때문에 범위를 확장한 2차 fine search를 수행했다.

## 2. 1차 Grid Search 결과 요약

- hybrid_cleaned: cleaned=0.90, TF-IDF=0.10, article=0.00
- weighted_field: title=0.20, full=0.20, current=0.10, problem=0.20, proposal=0.20, article=0.10
- final_ensemble: raw=0.10, structured=0.10, problem_proposal=0.10, weighted_field=0.20, cleaned=0.05, hybrid=0.45

## 3. 2차 탐색 범위 확장 이유

hybrid의 cleaned 상한을 1.00, final ensemble의 hybrid 상한을 0.60으로 확장했다. weighted field의 title/current/article도 1차 경계 바깥을 확인하도록 넓혔고 article=0.00은 유지했다.

## 4. 평가 데이터 설명

- 전체 데이터: 75개 법안, full-label 2,009개 라벨 쌍
- gold-only: 20개 source, 516개 라벨 쌍
- gold 판별 방식: label_source 컬럼 없음; 기존 evaluation_pooled_label_template_v2.xlsx의 20개 source subset
- full-label은 gold/LLM 혼합 라벨 기준이므로 보조 평가로 해석해야 하며, 최종 판단에서는 gold-only가 더 중요하다.
- 평가되지 않은 추천 쌍은 similarity를 0으로 채우지 않았으며, 지표 집계에서 unjudged로 제외했다.

## 5. Component Normalization 정책

모든 family에 `row_minmax`를 적용했다. 대각은 min/max 계산에서 제외했고 row 내 min=max인 경우 해당 row를 0으로 처리했다. 원본 및 정규화 통계는 결과 JSON에 저장했다.

## 6. hybrid_cleaned 2차 탐색 결과

- 후보 수: 25
- 최종 가중치: w_cleaned=0.900, w_tfidf=0.100, w_article=0.000
- full-label: P@5=0.3333, P@10=0.2233, nDCG@10=0.7965, MRR=0.6245, AvgRel=1.3454, Legal=33.35
- gold-only: P@5=0.4800, P@10=0.3150, nDCG@10=0.8522, MRR=0.8667, AvgRel=1.6750, Legal=42.49

## 7. weighted_field 2차 탐색 결과

- 후보 수: 5,253
- 최종 가중치: w_title=0.300, w_full=0.200, w_current=0.100, w_problem=0.150, w_proposal=0.150, w_article=0.100
- full-label: P@5=0.3753, P@10=0.2585, nDCG@10=0.8034, MRR=0.6578, AvgRel=1.5094, Legal=38.0
- gold-only: P@5=0.5325, P@10=0.3607, nDCG@10=0.8569, MRR=0.8875, AvgRel=1.9435, Legal=48.77

## 8. final_ensemble 2차 탐색 결과

- 후보 수: 12,959
- 최종 가중치: w_raw=0.075, w_structured=0.100, w_problem_proposal=0.075, w_weighted_field=0.300, w_cleaned=0.000, w_hybrid=0.450
- full-label: P@5=0.3840, P@10=0.2544, nDCG@10=0.8229, MRR=0.6744, AvgRel=1.4516, Legal=36.36
- gold-only: P@5=0.5400, P@10=0.3507, nDCG@10=0.9220, MRR=0.9625, AvgRel=1.8429, Legal=46.62

## 9. 1차 best 대비 개선 여부

아래 v1 수치는 과거 리포트 값을 그대로 복사한 것이 아니라, 공정한 비교를 위해 v1 가중치를 2차와 동일한 전체 75×75 행렬, row-minmax 및 평가 규칙으로 다시 계산한 값이다.

| Method | v1 objective | v2 objective | full-label 변화 | gold-only 변화 |
|---|---:|---:|---|---|
| hybrid_cleaned | 0.5305 | 0.5305 | 동률 (+0.0000) | 동률 (+0.0000) |
| weighted_field | 0.5421 | 0.5603 | 개선 (+0.0182) | 하락 (-0.0167) |
| final_ensemble | 0.5554 | 0.5737 | 개선 (+0.0183) | 개선 (+0.0046) |

final_ensemble이 단일 메소드보다 실제로 개선되는지 별도로 확인해야 한다. 이번 결과에서 full-label 기준 변화는 +0.0134, gold-only 기준 변화는 +0.0383이다.

## 10. full-label 결과와 gold-only 결과 비교

full-label은 75개 법안의 gold/LLM 혼합 라벨 전체를 사용한다. gold-only는 기존 20개 human 평가 source subset만 사용하며, 모델 선택의 최종 판단에서는 gold-only 결과를 더 중요하게 본다.

## 11. train/validation 검증 결과

- hybrid_cleaned_fine: train best `w_cleaned=0.925, w_tfidf=0.075, w_article=0.000` / validation P@5=0.3467, P@10=0.2933, nDCG@10=0.7956, MRR=0.5667, AvgRel=1.5267, Legal=37.52
- weighted_field_fine: train best `w_title=0.300, w_full=0.200, w_current=0.100, w_problem=0.150, w_proposal=0.150, w_article=0.100` / validation P@5=0.4500, P@10=0.3074, nDCG@10=0.7822, MRR=0.6167, AvgRel=1.7120, Legal=40.58
- final_ensemble_fine: train best `w_raw=0.200, w_structured=0.100, w_problem_proposal=0.050, w_weighted_field=0.300, w_cleaned=0.000, w_hybrid=0.350` / validation P@5=0.4133, P@10=0.3039, nDCG@10=0.8152, MRR=0.6667, AvgRel=1.6268, Legal=40.33

## 12. 5-fold 검증 결과

- hybrid_cleaned_fine: P@5=0.3147, P@10=0.2222, nDCG@10=0.7818, MRR=0.5847, AvgRel=1.3416, Legal=33.18
- weighted_field_fine: P@5=0.3724, P@10=0.2602, nDCG@10=0.7820, MRR=0.6267, AvgRel=1.5162, Legal=38.11
- final_ensemble_fine: P@5=0.3733, P@10=0.2483, nDCG@10=0.8123, MRR=0.6466, AvgRel=1.4292, Legal=35.7

## 13. 최종 추천 가중치

- hybrid_cleaned: w_cleaned=0.900, w_tfidf=0.100, w_article=0.000
- weighted_field: w_title=0.300, w_full=0.200, w_current=0.100, w_problem=0.150, w_proposal=0.150, w_article=0.100
- final_ensemble: w_raw=0.075, w_structured=0.100, w_problem_proposal=0.075, w_weighted_field=0.300, w_cleaned=0.000, w_hybrid=0.450

full-label 최적 후보를 저장했지만 실제 채택 시에는 gold-only 성능과 CV 안정성을 함께 확인한다. hybrid_cleaned의 article_similarity가 계속 0 또는 매우 낮게 선택된다면 현재 article_similarity는 점수 계산보다 추천 설명 근거로 사용하는 편이 적절하다.

## 14. row-minmax 점수 해석 주의

row-minmax normalized score는 절대적 동일도를 뜻하지 않고 source별 상대 순위를 위한 점수다. 사용자에게 추천 이유를 보여줄 때 normalized score를 그대로 ‘유사도 1.0’ 또는 ‘완전 일치’로 표현하지 말고, 원본 점수나 높음/중간/낮음 등급으로 변환해야 한다. top-k JSON에는 두 점수를 명확히 분리해 저장했다.

## 15. 향후 개선 방향

- gold source와 완전 판단 qrels를 확대해 pooled-label 편향을 줄인다.
- article component는 조문 번호 외에 조문 역할·개정 효과를 반영하도록 재설계한다.
- 후보 생성 성능과 최종 ranking 성능을 분리해 온라인 평가 또는 사용자 피드백으로 검증한다.
- final_ensemble의 단일 메소드 대비 개선 폭이 작거나 gold-only에서 하락하면 더 단순한 모델을 우선한다.

### Normalization 통계 요약

총 21개 component/파생 component의 정규화 전후 통계를 JSON에 저장했다.
