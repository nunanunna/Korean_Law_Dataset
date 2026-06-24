# 3차 Title-Heavy Weighted Field 탐색 보고서

## 1. 실험 목적

법률발의안 유사도에서 ‘어떤 법률을 개정하는가’가 어느 정도까지 중요한지 검증했다. 전체 75×75 행렬을 재계산하고 title SBERT와 exact same-law 두 방식을 비교했다.

## 2. 2차 Fine Search 결과 요약

- hybrid_cleaned: cleaned=0.90, TF-IDF=0.10, article=0.00
- weighted_field: title=0.30, full=0.20, current=0.10, problem=0.15, proposal=0.15, article=0.10
- final_ensemble: raw=0.075, structured=0.10, problem_proposal=0.075, weighted_field=0.30, cleaned=0.00, hybrid=0.45

## 3. 왜 title 비중을 확장했는가

2차 최적 w_title=0.30이 탐색 상한에 걸렸기 때문에 0.70까지 확장했다. title 비중을 높이면 같은 법률명 또는 같은 법체계의 법안이 상위에 더 많이 노출될 수 있으므로 성능과 편향을 함께 측정했다.

## 4. 평가 데이터 설명

- full-label: 75 sources, 2,009 judged pairs (gold/LLM 혼합)
- gold-only: 20 sources, 516 judged pairs
- gold 판별: label_source 컬럼 없음; 기존 evaluation_pooled_label_template_v2.xlsx의 20개 source subset
- full-label 결과는 혼합 라벨 기준이므로 보조 평가로 해석하며, gold-only와 사례 분석을 최종 판단에서 더 중요하게 본다.

## 5. title_law_name_similarity 생성 방식

bill_name에서 일부/전부개정법률안, 제정법률안, 개정안, 법률안 등의 suffix를 제거했다. 정규화 법률명을 SBERT cosine으로 비교한 방식과 완전 일치 시 1인 exact 방식을 모두 계산했다.

## 6. title-heavy weighted_field 탐색 범위

0.025 step으로 SBERT 28,205개, exact 28,205개를 평가했다. w_title 범위는 0.30~0.70이며 w_title=0.00과 0.85 수동 ablation도 포함했다.

## 7. weighted_field_title_sbert 결과

### 선택 기준별 결과

- best_by_full_label_objective: `w_title=0.300, w_full=0.225, w_current=0.100, w_problem=0.125, w_proposal=0.175, w_article=0.075` / balanced=0.6020 / full objective=0.5643 / gold objective=0.7055
- best_by_gold_only_objective: `w_title=0.300, w_full=0.100, w_current=0.150, w_problem=0.150, w_proposal=0.150, w_article=0.150` / balanced=0.6126 / full objective=0.5560 / gold objective=0.7306
- best_by_full_label_candidate_recall: `w_title=0.850, w_full=0.050, w_current=0.000, w_problem=0.025, w_proposal=0.025, w_article=0.050` / balanced=0.5843 / full objective=0.5506 / gold objective=0.6527
- best_by_gold_only_candidate_recall: `w_title=0.850, w_full=0.050, w_current=0.000, w_problem=0.025, w_proposal=0.025, w_article=0.050` / balanced=0.5843 / full objective=0.5506 / gold objective=0.6527
- best_by_ranking_objective: `w_title=0.300, w_full=0.225, w_current=0.100, w_problem=0.125, w_proposal=0.175, w_article=0.075` / balanced=0.6020 / full objective=0.5643 / gold objective=0.7055
- best_by_balanced_score: `w_title=0.300, w_full=0.100, w_current=0.150, w_problem=0.150, w_proposal=0.150, w_article=0.150` / balanced=0.6126 / full objective=0.5560 / gold objective=0.7306

Balanced 추천 full: P@5=0.3876, P@10=0.2674, nDCG@10=0.7840, MRR=0.6351, AvgRel=1.5342, Legal=38.6

Balanced 추천 gold: P@5=0.5692, P@10=0.3784, nDCG@10=0.8701, MRR=0.9250, AvgRel=1.9942, Legal=50.59

## 8. weighted_field_title_exact 결과

### 선택 기준별 결과

- best_by_full_label_objective: `w_title=0.650, w_full=0.050, w_current=0.075, w_problem=0.075, w_proposal=0.150, w_article=0.000` / balanced=0.5555 / full objective=0.5348 / gold objective=0.6961
- best_by_gold_only_objective: `w_title=0.300, w_full=0.250, w_current=0.150, w_problem=0.150, w_proposal=0.150, w_article=0.000` / balanced=0.5577 / full objective=0.5205 / gold objective=0.7129
- best_by_full_label_candidate_recall: `w_title=0.575, w_full=0.050, w_current=0.150, w_problem=0.075, w_proposal=0.050, w_article=0.100` / balanced=0.5327 / full objective=0.5123 / gold objective=0.6541
- best_by_gold_only_candidate_recall: `w_title=0.625, w_full=0.100, w_current=0.150, w_problem=0.050, w_proposal=0.050, w_article=0.025` / balanced=0.5513 / full objective=0.5176 / gold objective=0.6879
- best_by_ranking_objective: `w_title=0.450, w_full=0.250, w_current=0.075, w_problem=0.050, w_proposal=0.175, w_article=0.000` / balanced=0.5528 / full objective=0.5321 / gold objective=0.6922
- best_by_balanced_score: `w_title=0.450, w_full=0.100, w_current=0.125, w_problem=0.075, w_proposal=0.175, w_article=0.075` / balanced=0.5622 / full objective=0.5308 / gold objective=0.7093

Balanced 추천 full: P@5=0.3313, P@10=0.2249, nDCG@10=0.7983, MRR=0.6284, AvgRel=1.3640, Legal=33.78

Balanced 추천 gold: P@5=0.5100, P@10=0.3125, nDCG@10=0.9076, MRR=0.9100, AvgRel=1.7202, Legal=43.42

## 9. title ablation 분석

| Family | Group | Candidates | w_title | Full P@5 | Full nDCG | Gold P@5 | Gold nDCG | Gold Legal | Balanced |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| title_sbert | no_title | 1 | 0.000 | 0.2907 | 0.7568 | 0.4600 | 0.9053 | 41.76 | 0.5267 |
| title_sbert | moderate_title | 16,905 | 0.300 | 0.3876 | 0.7840 | 0.5692 | 0.8701 | 50.59 | 0.6126 |
| title_sbert | heavy_title | 11,298 | 0.475 | 0.3796 | 0.7752 | 0.5367 | 0.8599 | 52.38 | 0.5979 |
| title_sbert | title_dominant_manual | 1 | 0.850 | 0.4138 | 0.7490 | 0.5421 | 0.7968 | 55.74 | 0.5843 |
| title_exact | no_title | 1 | 0.000 | 0.2907 | 0.7568 | 0.4600 | 0.9053 | 41.76 | 0.5267 |
| title_exact | moderate_title | 16,905 | 0.450 | 0.3313 | 0.7983 | 0.5100 | 0.9076 | 43.42 | 0.5622 |
| title_exact | heavy_title | 11,298 | 0.525 | 0.3320 | 0.7937 | 0.5100 | 0.9084 | 43.35 | 0.5620 |
| title_exact | title_dominant_manual | 1 | 0.850 | 0.3287 | 0.7392 | 0.4925 | 0.8089 | 41.01 | 0.5111 |

## 10. same-law/title 편향 진단

- 선택 family: `weighted_field_title_sbert`
- v2 → title-heavy same_law_ratio@10 변화: +0.0000
- gold P@5 변화: +0.0367
- gold nDCG@10 변화: +0.0132
- gold Legal Meaning Score 변화: +1.82
- SBERT w_title와 same_law_ratio@10 상관: 0.2333
- exact w_title와 same_law_ratio@10 상관: 0.0272

same_law_ratio 증가가 P@5 또는 nDCG 개선으로 이어지는지는 위 변화량과 ablation 표를 함께 봐야 한다. same_law 추천이 늘면서 gold-only 또는 Legal Meaning Score가 하락하는 구간은 title 과적합, 즉 실제 개정 취지 다양성을 놓치는 신호로 해석한다.

## 11. final_ensemble 재탐색 결과

### 선택 기준별 결과

- best_by_full_label_objective: `w_raw=0.000, w_structured=0.200, w_problem_proposal=0.000, w_weighted_field=0.450, w_cleaned=0.000, w_hybrid=0.350` / balanced=0.5953 / full objective=0.5755 / gold objective=0.7290
- best_by_gold_only_objective: `w_raw=0.050, w_structured=0.050, w_problem_proposal=0.150, w_weighted_field=0.250, w_cleaned=0.050, w_hybrid=0.450` / balanced=0.6224 / full objective=0.5713 / gold objective=0.7539
- best_by_full_label_candidate_recall: `w_raw=0.075, w_structured=0.050, w_problem_proposal=0.000, w_weighted_field=0.450, w_cleaned=0.000, w_hybrid=0.425` / balanced=0.5700 / full objective=0.5621 / gold objective=0.7253
- best_by_gold_only_candidate_recall: `w_raw=0.075, w_structured=0.050, w_problem_proposal=0.000, w_weighted_field=0.450, w_cleaned=0.000, w_hybrid=0.425` / balanced=0.5700 / full objective=0.5621 / gold objective=0.7253
- best_by_ranking_objective: `w_raw=0.025, w_structured=0.200, w_problem_proposal=0.000, w_weighted_field=0.350, w_cleaned=0.000, w_hybrid=0.425` / balanced=0.6135 / full objective=0.5746 / gold objective=0.7321
- best_by_balanced_score: `w_raw=0.050, w_structured=0.050, w_problem_proposal=0.150, w_weighted_field=0.250, w_cleaned=0.050, w_hybrid=0.450` / balanced=0.6224 / full objective=0.5713 / gold objective=0.7539

최종 balanced 가중치: `w_raw=0.050, w_structured=0.050, w_problem_proposal=0.150, w_weighted_field=0.250, w_cleaned=0.050, w_hybrid=0.450`

v2 대비 full objective 변화: -0.0024

v2 대비 gold objective 변화: +0.0148

## 12. full-label vs gold-only 결과 비교

full-label objective best와 gold-only objective best가 다를 수 있으므로 두 결과를 모두 저장했다. 최종 추천은 gold 비중이 더 큰 balanced score를 사용한다.

## 13. train/validation 및 5-fold 검증

- weighted_field_title_sbert: train best `w_title=0.300, w_full=0.250, w_current=0.100, w_problem=0.100, w_proposal=0.175, w_article=0.075` / validation P@5=0.4067, P@10=0.3163, nDCG@10=0.7759, MRR=0.5833, AvgRel=1.7280, Legal=40.28 / 5-fold avg P@5=0.3585, P@10=0.2614, nDCG@10=0.7946, MRR=0.6339, AvgRel=1.5298, Legal=38.22
- weighted_field_title_exact: train best `w_title=0.450, w_full=0.250, w_current=0.075, w_problem=0.050, w_proposal=0.175, w_article=0.000` / validation P@5=0.2933, P@10=0.2541, nDCG@10=0.7814, MRR=0.6333, AvgRel=1.5211, Legal=36.65 / 5-fold avg P@5=0.3093, P@10=0.2133, nDCG@10=0.7808, MRR=0.5938, AvgRel=1.3183, Legal=32.54
- final_ensemble_title_heavy: train best `w_raw=0.025, w_structured=0.200, w_problem_proposal=0.000, w_weighted_field=0.350, w_cleaned=0.000, w_hybrid=0.425` / validation P@5=0.4133, P@10=0.3215, nDCG@10=0.8406, MRR=0.6667, AvgRel=1.6454, Legal=40.51 / 5-fold avg P@5=0.3773, P@10=0.2551, nDCG@10=0.8159, MRR=0.6665, AvgRel=1.4506, Legal=36.41

## 14. 최종 추천 가중치

- weighted_field family: `weighted_field_title_sbert`
- weighted_field balanced: `w_title=0.300, w_full=0.100, w_current=0.150, w_problem=0.150, w_proposal=0.150, w_article=0.150`
- final_ensemble balanced: `w_raw=0.050, w_structured=0.050, w_problem_proposal=0.150, w_weighted_field=0.250, w_cleaned=0.050, w_hybrid=0.450`

full-label best와 gold-only best는 각각 별도 항목으로 best JSON에 기록했다. 실제 채택은 balanced 추천, gold-only 지표, CV 안정성, 사례 검토를 함께 판단해야 한다.

## 15. 사례 분석 샘플 요약

- 순위 상승 후보: 30개
- 순위 하락 후보: 30개
- same-law 저관련 후보: 0개
- different-law 고관련 후보: 30개
- 중복 제거 후 90개, gold source 90개 row를 저장했다.

## 16. 한계와 다음 단계

- pooled qrels에 없는 추천은 0점 similarity로 만들지 않고 unjudged로 제외했다. 완전 판단 qrels 확대가 필요하다.
- exact same-law는 법률명이 다른 연관 법체계를 잡지 못하고, SBERT title은 유사 법률명을 과대평가할 수 있다.
- title-only SBERT와 exact baseline도 결과 CSV/JSON에 포함했으며, 문맥 component가 없는 순수 title 성능과 비교해야 한다.
- title 비중을 높이면 같은 법률명 또는 같은 법체계 추천이 증가할 수 있으므로 실제 사용자 사례 검증이 필요하다.
- row-minmax normalized score는 절대적 동일도가 아니라 source별 상대 점수다.
- 추천 이유에서 normalized score를 ‘유사도 1.0’이라고 표현하지 말고 원본 점수 또는 높음/중간/낮음 등급을 사용해야 한다.
