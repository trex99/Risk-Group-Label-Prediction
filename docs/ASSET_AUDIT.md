# 논문 자원 정리·검증 기록

검증 기준일: 2026-07-18

## 1. 검증 기준 원고

- 심사용 v19 PDF(공개 패키지에 미포함)
  - 22쪽
  - SHA-256: `89ADBB99826A0F4D4676CE799EDB4D9418BD9FD41876A3D1EBD96A00EB5C26EE`
- 원문 v19 PDF(공개 패키지에 미포함)
  - 23쪽
  - SHA-256: `15E397C0CE48B10281624B71740B5D73DAC611F0302CCBF60D8519F1D7EBDC3D`

두 PDF는 이번 정리에서 읽기 전용으로만 대조했으며 수정하지 않았다.

## 2. 논문 그림

`figures`에는 두 v19 PDF에 실제 삽입된 그림 PNG 4개와 그림 2 편집용 SVG를 둔다. PNG는 모두 약 300 DPI이다.

| 번호 | 최종 파일 | 픽셀 크기 | 생성·편집 자원 |
|---|---|---:|---|
| 그림 1 | `figures/Figure_1_Exploratory_Data_Analysis_v18_corrected.png` | 4662×3516 | `create_figure1.py` |
| 그림 2 | `figures/Figure_2_Data_Analysis_and_Interpretation_Process_EN.png` | 3210×3480 | `figures/Figure_2_Data_Analysis_and_Interpretation_Process_EN.svg` |
| 그림 3 | `figures/Figure_3_SHAP_VotingClassifier_300dpi.png` | 2370×2820 | `run_votingclassifier_shap.py` |
| 그림 4 | `figures/Figure_4_Reliability_Adaptive10Bins_300dpi.png` | 3119×1320 | `make_reliability_figure.py` |

PDF 내장 이미지와 최종 파일의 축소·압축 차이를 보정한 영상 상관계수는 심사용/원문에서 각각 그림 1 `0.999581`, 그림 2 `0.990627`, 그림 3 `0.999747`, 그림 4 `0.999399`였다. 두 PDF 모두 네 그림이 확인되었다.

논문에 삽입하지 않은 국문판·PDF 변형본은 공개 패키지에서 제외하였다.

## 3. 논문 표

- `tables/table1_*.csv`부터 `tables/table11_*.csv`까지 표 1~11이 모두 있다.
- 행·열은 v19 PDF에 실제 표시된 표 형태로 정리하였다.
- 두 PDF에서 표 1~11의 데이터 셀 252개를 직접 대조했으며 누락·불일치 셀은 0개였다.
- 전체 정밀 산출값은 재현 실행 시 `runtime/results`에 생성된다.
- 논문 표시용 표 생성 코드는 루트의 `build_manuscript_artifacts.py`이다.

표 10의 상대시점 조건 Top 5% 라벨률은 원 산출값이 `0.1106498565`이고 v19 PDF에는 `11.07%`로 표시되어 있다. 논문이 고정된 상태이므로 `tables/table10_temporal_holdout_topk.csv`는 PDF 표시값을 재현하며, 원 정밀값은 결과 CSV에 그대로 보존한다.

## 4. 최종 실험 자원

| 구분 | 위치 |
|---|---|
| 전체 재현 실행 진입점 | `run_full_reproduction.py` |
| 최종 누수 차단 실험 코드 | 저장소 루트의 Python 파일 25개 |
| 엄격 동일월 제외 데이터 | `runtime/strict_month_reanalysis/data` |
| 최종 정밀 결과 | `runtime/results` |
| Optuna·평가 로그 | `runtime/logs` |
| fold별 이력 캐시 | `runtime/cache` |
| 원자료 | 사용자가 `--data-path`로 지정한 `open_v2` 디렉터리 |

`python run_full_reproduction.py --dry-run`으로 전체 실행 순서를 확인할 수 있다. 기본값은 각 모델·fold당 Optuna 100회, 난수 시드 42이며, 최종 확률은 `VotingClassifier.predict_proba()`에서 직접 산출한다. 원자료 경로, 재현 작업경로, 표·그림 출력경로를 현재 구조에서 확인하였다.

## 5. 검증 상태

- 최종 기준 누수 감사: `PASS`
- 최종 기준 2회 독립 수치 검산: `PASS`
  - 지표 벡터 166개
  - 외부 예측 파일 11개
  - 실패 0개
- 앙상블 확률 출처: `VotingClassifier.predict_proba`
- 수동 평균 확률을 논문 결과에 사용: `false`
- SHAP 산출 상태: `COMPLETE`, seed 42
- 최종 Python 코드 구문 검사: 오류 0개

## 6. 루트 구조

- 최종 재현 Python 코드: 저장소 루트
- 최종 논문 그림과 편집용 SVG: `figures`
- 최종 논문 표와 검증 기록: `tables`
- 재현 중간자료: `runtime` (`.gitignore` 적용)

원자료, 캐시, 로그와 검사건별 예측파일은 공개 패키지에 포함하지 않는다.
