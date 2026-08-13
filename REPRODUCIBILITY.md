# 재현성 안내

이 문서는 논문의 분석환경, 입력자료 배치, 전체·부분 실행, 누수 차단 규칙과 산출물 검증 절차를 설명한다. 연구내용과 핵심 결과는 [README](README.md)를 참고한다.

## 1. 재현 수준

공개 저장소는 세 수준의 확인을 지원한다.

1. **자원 검증**: 코드 구문, 필수파일, 논문 표·그림의 SHA-256을 확인한다.
2. **전체 재현**: 원자료에서 피처 생성, 튜닝, 외부평가, 민감도 분석, SHAP과 논문 자원 생성까지 수행한다.
3. **선택 분석 재현**: 저장된 앞 단계 산출물을 이용하여 특정 단계나 최종 확장 분석만 수행한다.

## 2. 실행환경

- Python 3.12
- NumPy, pandas, SciPy
- scikit-learn
- LightGBM, XGBoost, CatBoost
- Optuna, SHAP
- Matplotlib, joblib, psutil
- 기본 난수 시드: 42

저장소 루트에서 의존 패키지를 설치한다.

```powershell
python -m pip install -r requirements.txt
```

## 3. 입력자료

분석은 DACON [운수종사자 운전적성정밀검사 데이터](https://dacon.io/competitions/official/236607/data)의 다음 파일구조를 전제로 한다.

```text
open_v2/
├─ train.csv
└─ train/
   ├─ A.csv
   └─ B.csv
```

원자료는 저장소에 포함하지 않는다. 데이터 이용조건과 공개 범위는 [DATA.md](DATA.md)를 참고한다.

기본 위치가 아닌 곳에 원자료가 있으면 `--data-path`를 지정한다.

```powershell
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

## 4. 빠른 공개 자원 검증

원자료나 모델 실행 없이 공개 패키지의 구조, Python 구문, 논문 표 행 수와 체크섬을 확인한다.

```powershell
python verify_release.py
```

정상 공개본은 다음 항목이 모두 `PASS`여야 한다.

- Python 코드 구문
- 필수파일과 공개정책
- 표 1–11의 행 수
- `tables/manuscript_artifacts_manifest.json` 상태와 버전
- `checksums.sha256`의 표·그림·보조 산출물

실험을 수행하지 않고 전체 호출 순서만 확인하려면 다음 명령을 사용한다.

```powershell
python run_full_reproduction.py --dry-run
```

## 5. 전체 재현

```powershell
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

전체 실행 순서는 다음과 같다.

```text
data → protocol → tuning → outer → sensitivity
→ post → verification → expanded → shap → artifacts
```

| 단계 | 주요 작업 |
|---|---|
| `data` | 원자료 결합, 37개 기본 피처와 그림 1 생성 |
| `protocol` | 분할·이력 원천·동일월 제외 및 누수 감사 |
| `tuning` | 외부 fold별 3-fold 내부 Optuna 튜닝 |
| `outer` | 단일모델과 VotingClassifier 외부평가 |
| `sensitivity` | 단계적 피처셋, 피처군 제거와 시점표현 비교 |
| `post` | PrimaryKey 군집 Bootstrap, 검사유형, Top-k와 보정도 |
| `verification` | 저장 예측값을 이용한 2회 독립 수치 검산 |
| `expanded` | 기준모델과 이력정보 제외·개인분리 분석 |
| `shap` | 5개 외부 fold permutation SHAP 및 그림 3 생성 |
| `artifacts` | 최종 표 1–11, ledger와 manifest 생성 |

주요 평가설정은 다음과 같다.

- 무작위 계층 외부 5-fold
- PrimaryKey 완전분리 외부 5-fold
- 시간분할: 2016–2020년 학습, 2021–2022년 검증
- 내부 튜닝: 외부 학습 파티션 내부 3-fold
- 모델: HistGradientBoosting, LightGBM, XGBoost, CatBoost
- Optuna: 11개 외부 fold × 4개 모델 × 100 trial
- 앙상블: `VotingClassifier(voting="soft")`
- 최종 확률: `VotingClassifier.predict_proba()` 직접 출력
- 군집 Bootstrap: PrimaryKey 단위 500회
- SHAP: 외부 5개 fold, fold별 검증 2,000건과 배경 100건

## 6. 부분 실행

단계 범위를 지정할 수 있다.

```powershell
python run_full_reproduction.py --start-at tuning --stop-after outer
```

`--start-at`으로 중간 단계부터 시작하려면 앞 단계의 산출물이 재현 작업경로에 존재해야 한다.

기본 데이터만 강제로 다시 생성하려면 다음과 같이 실행한다.

```powershell
python run_full_reproduction.py --force-data --stop-after data
```

기본 중간자료 경로는 `runtime`이다. 별도 경로를 사용하려면 환경변수를 지정한다.

```powershell
$env:TRANSPORT_PAPER_RUNTIME="D:\transport_paper_runtime"
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

## 7. 최종 확장 분석의 개별 실행

기본 재현의 `outer`, `sensitivity` 및 `verification` 산출물이 준비된 상태에서 다음 순서로 실행한다.

```powershell
python revision/run_revision_analyses.py
python revision/tune_logistic_c.py
PowerShell -ExecutionPolicy Bypass -File revision/run_primarykey_history_factorial_queue.ps1
python revision/run_primarykey_history_factorial.py --condition summarize
python revision/run_shap_fold_stability.py
python build_manuscript_artifacts.py
```

PrimaryKey 완전분리에서 검증 개인의 엄격히 이전 자기이력을 별도 원천으로 사용하는 보조조건은 최종 [표 7]에 포함되지 않는다. 이 조건까지 실행하려면 다음 명령을 사용한다.

```powershell
PowerShell -ExecutionPolicy Bypass -File revision/run_primarykey_history_factorial_queue.ps1 -IncludeSupplementaryHistory
python revision/run_primarykey_history_factorial.py --condition summarize
```

개별 SHAP 배열과 중간 예측값은 `revision/tmp/`에 저장되며 Git 추적 대상에서 제외한다. 5-fold SHAP을 처음부터 다시 계산하려면 `--force`를 지정한다.

```powershell
python revision/run_shap_fold_stability.py --force
```

세부 결과 파일은 [revision/README.md](revision/README.md)에 정리되어 있다.

## 8. 누수 차단 규칙

- 이력은 `source.TestDate < target.TestDate`를 만족하는 기록만 사용한다.
- 동일 PrimaryKey의 동일 연월 기록은 선후관계를 확인할 수 없어 서로의 과거이력에서 제외한다.
- 외부 검증행은 학습·검증 이력의 원천으로 사용하지 않는다.
- 내부 검증행은 해당 내부 fold의 학습 피처 원천으로 사용하지 않는다.
- 검증 라벨은 동일 검증 파티션의 다른 검사건 피처에 사용하지 않는다.
- 분할 인덱스와 원천자료 해시가 일치하는 캐시만 재사용한다.
- 이력 피처를 교체한 뒤 `isna_sum`을 해당 fold의 최종 피처 기준으로 다시 계산한다.

## 9. 계산자원

기준 장비는 Intel Core Ultra 7 265K 20코어와 32GB 메모리였다.

- 기본 파이프라인 경과시간: 약 60–70시간
- 최종 확장 분석과 5-fold SHAP 실행시간: 별도
- 누적 CPU 사용량 근사치: 약 430–500 코어시간
- 원자료 제외 기본 중간 산출물: 약 6.7GiB
- 원자료 포함 기본 사용량: 약 8.5GiB
- 권장 여유공간: 12–15GiB 이상

실행시간은 CPU, 메모리, 저장장치와 패키지 빌드에 따라 달라진다.

## 10. 산출물

```text
figures/   최종 논문 그림 1–4와 그림 2 편집용 SVG
tables/    최종 표 1–11, numeric ledger와 manifest
revision/  최종 확장·보조 분석 표와 그림
runtime/   원자료 파생자료, 캐시, 로그와 정밀 예측값(공개 제외)
```

표 10 상대시점 조건의 Top 5% 라벨률 원 산출값은 `0.1106498565`이다. 최종 v20 논문 표의 표시값을 재현하기 위해 논문용 표에는 `11.07%`를 유지하고 원 정밀값은 재현 결과에 보존한다.

전체 실행 후 공개 기준 자원과 비교하려면 다시 다음 명령을 수행한다.

```powershell
python verify_release.py
```
