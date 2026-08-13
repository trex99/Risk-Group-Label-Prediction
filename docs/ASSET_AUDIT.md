# 최종 논문 자원 정리·검증 기록

검증 기준일: 2026-08-13

## 1. 검증 기준 원고

- 공개 자원 대조에 사용한 최종 v20 원문 PDF 스냅샷(공개 패키지에 미포함)
  - 28쪽
  - SHA-256: `D86607E0DC790B0C85B9D35A8F044B90AB1FCE48F72C5A4D9396C8A5761248A8`

PDF는 읽기 전용으로 대조했으며 수정하지 않았다. 공개 자원은 이 스냅샷의 표 수치와 그림 내용을 따르되, 이후 편집부 요청에 따른 영문 대·소문자와 축 제목 정리는 생성 코드와 공개 그림에 반영하였다.

## 2. 논문 그림

`figures`에는 최종 논문 그림 PNG 4개와 그림 2 편집용 SVG를 둔다. PNG는 모두 약 300 DPI이다.

| 번호 | 최종 파일 | 픽셀 크기 | 생성·편집 자원 |
|---|---|---:|---|
| 그림 1 | `figures/Figure_1_Exploratory_Data_Analysis_v18_corrected.png` | 4662×3516 | `create_figure1.py` |
| 그림 2 | `figures/Figure_2_Data_Analysis_and_Interpretation_Process_EN.png` | 3210×3480 | 편집용 SVG |
| 그림 3 | `figures/Figure_3_SHAP_VotingClassifier_300dpi.png` | 2730×4480 | `revision/run_shap_fold_stability.py` |
| 그림 4 | `figures/Figure_4_Reliability_Adaptive10Bins_300dpi.png` | 3119×1320 | `make_reliability_figure.py` |

그림 3은 5개 외부 fold의 OOF permutation SHAP 요약과 fold별 평균 절대 SHAP의 단순평균을 결합한 최종 그림이다. 개별 패널과 순위 안정성 보조 그림은 `revision/figures`에 보존한다.

## 3. 논문 표

- `tables/table1_*.csv`부터 `tables/table11_*.csv`까지 표 1–11이 모두 있다.
- 최종 v20 PDF의 표 1–11 제목, 행, 열과 표시 수치를 대조하였다.
- [표 3]은 DummyClassifier, 튜닝된 LogisticRegression, 4개 GBDT 모델과 최종 VotingClassifier를 포함한다.
- [표 7]은 주 중첩 교차검증 이력 포함, 주 외부 fold 이력 제외, PrimaryKey 완전분리 이력 제외의 세 조건을 포함한다.
- 전체 정밀 산출값은 재현 실행 시 `runtime/results`와 `revision/tables`에 생성된다.
- 논문 표시용 표 생성 코드는 루트의 `build_manuscript_artifacts.py`이다.

표 10의 상대시점 조건 Top 5% 라벨률 원 산출값은 `0.1106498565`이고 최종 v20 PDF에는 `11.07%`로 표시되어 있다. `tables/table10_temporal_holdout_topk.csv`는 PDF 표시값을 재현하며, 원 정밀값은 재현 결과에 보존한다.

## 4. 재현 자원

| 구분 | 위치 |
|---|---|
| 전체 재현 실행 진입점 | `run_full_reproduction.py` |
| 기본 누수 차단 분석 코드 | 저장소 루트의 Python 파일 |
| 최종 확장·보조 분석 코드 | `revision` |
| 엄격 동일월 제외 데이터 | `runtime/strict_month_reanalysis/data` |
| 최종 정밀 결과 | `runtime/results` |
| Optuna·평가 로그 | `runtime/logs` |
| fold별 이력 캐시 | `runtime/cache` |
| 원자료 | 사용자가 `--data-path`로 지정한 `open_v2` 디렉터리 |

`python run_full_reproduction.py --dry-run`으로 전체 실행 순서를 확인할 수 있다. 기본값은 각 모델·fold당 Optuna 100회, 난수 시드 42이며, 최종 확률은 `VotingClassifier.predict_proba()`에서 직접 산출한다.

## 5. 검증 상태

- 최종 기준 누수 감사: `PASS`
- 최종 기준 2회 독립 수치 검산: `PASS`
  - 지표 벡터 166개
  - 외부 예측 파일 11개
  - 실패 0개
- 앙상블 확률 출처: `VotingClassifier.predict_proba`
- 수동 평균 확률을 논문 결과에 사용: `false`
- 5-fold permutation SHAP 산출 상태: 외부 fold 5개 모두 `COMPLETE`
- 최종 Python 코드 구문과 공개 자원 체크섬: `verify_release.py`로 검증

## 6. 공개 범위

- 최종 재현 Python 코드: 저장소 루트
- 최종 논문 그림과 편집용 SVG: `figures`
- 최종 논문 표와 검증 기록: `tables`
- 최종 확장·보조 분석: `revision`
- 재현 중간자료: `runtime` (`.gitignore` 적용)

원자료, 캐시, 로그와 검사건별 예측파일은 공개 패키지에 포함하지 않는다.
