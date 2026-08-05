# 검사이력과 인지·반응 통계를 활용한 운수종사자 운전적성정밀검사 위험군 라벨 예측

이 저장소는 「검사이력과 인지·반응 통계를 활용한 운수종사자 운전적성정밀검사 위험군 라벨 예측」의 투명한 검증과 재현을 위한 연구 코드·최종 표·그림 패키지이다. 원자료와 개별 검사건 예측파일은 저장소에 포함하지 않는다.

## 1. 목적

본 디렉터리는 운전적성정밀검사 데이터셋을 이용한 위험군 라벨 예측 연구의 최종 결과를 재현하기 위한 코드로 구성하였다. 분석자료 생성부터 누수 차단 검증, 중첩 교차검증, Optuna 튜닝, 최종 VotingClassifier 평가, 민감도 분석, Bootstrap, Top-k, 보정도, SHAP, 논문용 표·그림 생성까지의 전체 절차를 포함한다.

25개 Python 파일을 사용자가 하나씩 직접 실행하는 구조가 아니다. 전체 재현의 진입점은 `run_full_reproduction.py`이며, 이 파일이 필요한 분석 파일을 정해진 순서대로 호출한다.

```text
원자료 open_v2
→ 37개 기본 피처 생성
→ 누수 차단 분할 및 이력 재생성
→ 외부 fold별 Optuna 재튜닝
→ VotingClassifier 외부평가
→ 피처군·시간·개인분리 민감도 분석
→ Bootstrap·Top-k·검사유형·보정도 분석
→ 2회 독립 수치 검증
→ permutation SHAP
→ 논문 표 1~11 및 그림 1·3·4 생성
```

본 코드는 데이터셋에서 제공된 위험군 라벨의 선별 성능을 재현한다. 실제 교통사고 발생확률을 직접 산출하거나 운전적성정밀검사의 공식 판정체계를 대체하는 코드는 아니다.

## 2. 입력자료

기본 원자료 위치는 다음과 같다.

```text
../../open_v2/
├─ train.csv
└─ train/
   ├─ A.csv
   └─ B.csv
```

분석자료는 DACON의 [데이터 페이지](https://dacon.io/competitions/official/236607/data)에서 제공된 파일구조를 전제로 한다. 데이터 이용조건과 접근절차는 원 배포처의 정책을 따르며, 본 저장소는 원자료를 재배포하지 않는다. 자세한 내용은 `DATA.md`를 참고한다.

다른 위치의 원자료를 사용하려면 전체 실행 시 `--data-path`를 지정한다.

```powershell
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

## 3. 실행환경

본 분석은 Python 3.12 환경에서 수행하였다. 주요 의존 패키지는 다음과 같다.

- NumPy
- pandas
- SciPy
- scikit-learn
- LightGBM
- XGBoost
- CatBoost
- Optuna
- SHAP
- Matplotlib
- joblib
- psutil

모든 분석의 기본 난수 시드는 42이다.

공개본과 동일한 패키지 버전은 다음 명령으로 설치한다.

```powershell
python -m pip install -r requirements.txt
```

## 4. 실행 방법

### 4.1 실행 순서만 확인

다음 명령은 실험을 수행하지 않고 호출될 명령만 출력한다.

```powershell
python run_full_reproduction.py --dry-run
```

### 4.2 전체 재현

```powershell
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

기본 실행은 각 모델·외부 fold마다 Optuna 100회를 수행한다. Intel Core Ultra 7 265K 20코어, 32GB급 메모리를 사용한 기준 실행에서 전체 경과시간은 약 60\~70시간이었다. 원자료를 제외한 중간 산출물은 약 6.7GiB였으며, 원자료를 포함하면 약 8.5GiB가 필요했다. 임시파일과 동기화 여유를 고려하여 12~15GiB 이상의 여유공간을 권장한다.

### 4.3 일부 단계만 실행

```powershell
python run_full_reproduction.py --start-at tuning --stop-after outer
```

사용 가능한 단계는 다음과 같다.

```text
data → protocol → tuning → outer → sensitivity
→ post → verification → shap → artifacts
```

`--start-at`으로 중간 단계부터 시작하려면 앞 단계의 산출물이 재현 작업경로에 이미 존재해야 한다.

### 4.4 기본 데이터 강제 재생성

```powershell
python run_full_reproduction.py --force-data --stop-after data
```

### 4.5 재현 작업경로 변경

기본 중간자료 저장 위치는 `runtime`이다. 다른 위치에 장기 보존하려면 실행 전에 환경변수를 설정한다.

```powershell
$env:TRANSPORT_PAPER_RUNTIME="D:\transport_paper_runtime"
python run_full_reproduction.py
```

## 5. 단계별 재현 절차

### 5.1 데이터 준비: `data`

관련 파일:

- `prepare_paper_data.py`
- `past_only_pipeline.py`
- `create_figure1.py`

`train.csv`, `train/A.csv`, `train/B.csv`를 결합하여 944,767건, 37개 피처의 기본 데이터셋을 생성한다. 과거이력은 예측 대상 검사월보다 엄격히 이전인 기록만 사용하며, 동일 PrimaryKey의 동일 연월 기록은 과거이력에서 제외한다.

이 단계에서 생성된 기본 이력열은 데이터 구조를 구성하기 위한 것이다. 실제 교차검증과 시간분할에서는 분할 이후 각 학습 파티션만 이력 원천으로 사용하여 이력 피처를 다시 생성한다.

주요 산출 위치:

```text
runtime/strict_month_reanalysis/data
```

그림 1도 이 단계에서 생성한다.

### 5.2 누수 차단 검증: `protocol`

관련 파일:

- `verify_fold_isolated_pipeline.py`
- `audit_leakage_protocol.py`
- `fold_isolated_pipeline.py`
- `nested_protocol.py`

다음 조건을 확인한다.

- 외부 학습행과 검증행의 중복이 없는지 확인
- 내부 fold가 외부 검증자료를 사용하지 않는지 확인
- PrimaryKey 완전분리 조건에서 개인 중복이 없는지 확인
- 시간분할과 시간기반 내부 fold가 과거에서 미래 방향인지 확인
- 검증 라벨을 변경해도 이력 피처가 변하지 않는지 확인
- 동일 월 기록이 과거이력에 포함되지 않는지 확인
- 캐시의 분할 및 원천자료 해시가 현재 실행과 일치하는지 확인

`fold_isolated_pipeline.py`는 데이터 로딩, 분할별 이력 생성, 모델 구성, 평가척도 및 VotingClassifier 생성을 담당하는 핵심 공통 모듈이다.

### 5.3 Optuna 튜닝: `tuning`

관련 파일:

- `prepare_nested_histories.py`
- `run_nested_optuna.py`
- `run_parallel_nested_models.py`
- `run_tuning_queue.py`

외부 평가구조는 다음과 같다.

- 무작위 계층 5-fold
- PrimaryKey 완전분리 5-fold
- 2016~2020년 학습, 2021~2022년 검증의 시간분할 1개

각 외부 학습 파티션 내부에서 3-fold 검증을 수행하며, HistGradientBoosting, LightGBM, XGBoost, CatBoost를 독립적으로 튜닝한다.

```text
11개 외부 fold × 4개 모델 = 44개 Optuna 연구
```

기본 탐색 횟수는 연구당 100회이다. 동일 외부 fold의 네 모델은 가용 자원을 고려하여 병렬 처리하고, 외부 fold는 큐 순서대로 진행한다. 중단 후 재실행할 경우 완료된 trial 수를 확인하여 부족한 횟수만 추가한다.

### 5.4 최종 VotingClassifier 외부평가: `outer`

관련 파일:

- `run_outer_evaluation.py`
- `summarize_outer_evaluation.py`
- `run_outer_queue.py`

각 외부 fold에서 독립적으로 선택한 네 모델을 다음과 같이 결합한다.

```python
VotingClassifier(
    estimators=[
        HistGradientBoosting,
        LightGBM,
        XGBoost,
        CatBoost,
    ],
    voting="soft",
)
```

최종 앙상블 확률은 반드시 다음 직접 출력으로 산출한다.

```python
VotingClassifier.predict_proba()
```

논문 결과를 위해 네 모델의 확률을 별도로 수동 평균하지 않는다. 단일모델 지표도 동일하게 적합된 VotingClassifier 내부 estimator의 예측확률에서 산출한다.

외부평가 이후 다음 자료를 생성한다.

- fold별 단일모델 및 VotingClassifier 성능
- OOF 예측확률
- 시간분할 예측확률
- PrimaryKey 완전분리 예측확률
- AUC, PR-AUC, Score, Brier, ECE
- Top-k 및 reliability 자료

### 5.5 민감도 분석: `sensitivity`

관련 파일:

- `run_sensitivity_evaluation.py`
- `summarize_sensitivity.py`
- `run_sensitivity_queue.py`

다음 조건을 평가한다.

- Model A: 연령·검사시점
- Model B: 연령·검사시점과 과거·교차 검사이력
- Model C: 현재 인지·반응 요약통계
- Model D: 전체 37개 피처
- 정확도·오류 통계 제거
- 연령·검사시점 제거
- 과거·교차 검사이력 제거
- 반응시간 통계 제거
- 결측수 피처 제거
- 현재 절대시점 변수 제거
- 과거 시점의 상대시점 변환
- 시간분할 조건에서의 시점표현 비교

이 단계는 전체모형에서 각 외부 fold별로 선택된 하이퍼파라미터를 고정하여 입력정보 변화에 따른 조건부 성능을 비교하는 민감도 분석이다.

### 5.6 후속 분석: `post`

관련 파일:

- `bootstrap_incremental_value.py`
- `summarize_test_type_subgroups.py`
- `make_reliability_figure.py`

다음 결과를 생성한다.

- Model D−Model B의 PrimaryKey 군집 Bootstrap 500회
- 신규검사와 자격유지검사별 성능
- 중첩 교차검증 OOF Top-k
- 시간분할 Top-k
- uniform 10-bin ECE
- adaptive 10-bin ECE
- reliability diagram인 그림 4

### 5.7 독립 수치 검산: `verification`

관련 파일:

- `verify_final_outputs.py`

저장된 예측값을 이용하여 두 가지 독립적인 계산 방식으로 다음 항목을 다시 계산한다.

- AUC
- PR-AUC
- Score
- Brier
- uniform/adaptive ECE
- Top-k 라벨률, lift, recall
- fold 요약값
- 군집 Bootstrap 요약값
- 검사유형별 지표

수동 확률 평균은 VotingClassifier 출력과의 동일성 감사에만 사용하며 논문 결과의 확률원으로 사용하지 않는다. 검산 결과가 `PASS`가 아니면 최종 논문 표 생성이 중단된다.

### 5.8 SHAP: `shap`

관련 파일:

- `run_votingclassifier_shap.py`

무작위 계층 외부 1번 fold에서 최종 VotingClassifier에 permutation SHAP을 적용한다.

- 설명자료: 외부 검증 파티션의 2,000건 표본
- 배경자료: 외부 학습 파티션의 100건 표본
- 난수 시드: 42
- 예측함수: `VotingClassifier.predict_proba()`

이 단계에서 그림 3과 SHAP 중요도 자료를 생성한다.

### 5.9 논문 자원 생성: `artifacts`

관련 파일:

- `build_manuscript_artifacts.py`

누수 감사, 2회 수치 검산 및 SHAP 완료 상태를 확인한 뒤 다음 자료를 생성한다.

- 표 1~11 CSV
- `manuscript_ready_tables.md`
- 수치 ledger
- 논문 자원 manifest

최종 표는 `tables`, 최종 그림은 `figures`에 저장한다. 논문 PDF 자체는 어떤 단계에서도 수정하지 않는다.

## 6. 25개 Python 파일의 역할

| 구분 | 파일 | 역할 |
|---|---|---|
| 전체 실행 | `run_full_reproduction.py` | 전체 단계를 순서대로 호출 |
| 데이터 준비 | `prepare_paper_data.py` | 원자료에서 37개 기본 피처 생성 |
| 데이터 준비 | `past_only_pipeline.py` | A/B 전처리와 엄격 과거이력 구성 |
| 그림 | `create_figure1.py` | 그림 1 생성 |
| 핵심 공통 | `fold_isolated_pipeline.py` | fold별 이력, 모델, 지표, 경로 관리 |
| 분할 | `nested_protocol.py` | 외부·내부 분할과 분할 검증 |
| 누수 검증 | `verify_fold_isolated_pipeline.py` | 이력 생성 규칙의 기본 검증 |
| 누수 검증 | `audit_leakage_protocol.py` | 전체 누수 감사 및 manifest 생성 |
| 캐시 | `build_fold_history_caches.py` | 선택적으로 외부 fold 이력 캐시 사전 생성 |
| 튜닝 준비 | `prepare_nested_histories.py` | 내부 fold용 이력자료 사전 구성 |
| 튜닝 | `run_nested_optuna.py` | 모델별 Optuna 연구 수행 |
| 튜닝 | `run_parallel_nested_models.py` | 동일 외부 fold의 모델 병렬 실행 |
| 튜닝 | `run_tuning_queue.py` | 전체 튜닝 큐 관리 |
| 외부평가 | `run_outer_evaluation.py` | fold별 VotingClassifier 적합·예측 |
| 외부평가 | `summarize_outer_evaluation.py` | 외부평가·Top-k·보정도 요약 |
| 외부평가 | `run_outer_queue.py` | 전체 외부평가 큐 관리 |
| 민감도 | `run_sensitivity_evaluation.py` | 개별 피처셋·시점조건 평가 |
| 민감도 | `summarize_sensitivity.py` | 민감도 결과 통합 |
| 민감도 | `run_sensitivity_queue.py` | 전체 민감도 분석 큐 관리 |
| Bootstrap | `bootstrap_incremental_value.py` | Model D−B 군집 Bootstrap |
| 하위집단 | `summarize_test_type_subgroups.py` | 신규검사·자격유지검사별 지표 |
| 보정도 | `make_reliability_figure.py` | 그림 4 생성 |
| 수치검증 | `verify_final_outputs.py` | 결과 전체의 2회 독립 검산 |
| SHAP | `run_votingclassifier_shap.py` | 그림 3과 SHAP 중요도 생성 |
| 논문 표 | `build_manuscript_artifacts.py` | 표 1~11과 manifest 생성 |

`build_fold_history_caches.py`는 전체 실행기가 직접 호출하지 않는 선택적 보조 코드이다. 무작위·개인분리 외부 fold의 이력행렬을 미리 만들거나 변경 행을 별도로 감사할 때만 실행한다. 일반적인 전체 재현에서는 필요한 이력 캐시를 각 단계가 생성하거나 재사용한다.

## 7. 산출물 위치

### 최종 논문 자원

```text
figures/   # 그림 1~4 및 그림 2 편집용 SVG
tables/    # 표 1~11, ledger, manifest, 검증 기록
```

그림 2는 분석결과 그래프가 아니라 연구절차를 정리한 편집 도식이다. 따라서 분석 코드로 자동 생성하지 않고 기존 PNG와 편집용 SVG를 최종 자원으로 유지한다.

### 재현 중간자료

```text
runtime/
├─ strict_month_reanalysis/
│  ├─ data/
│  └─ outputs/
├─ cache/
├─ data/
├─ logs/
└─ results/
   ├─ optuna/
   ├─ nested_evaluation/
   ├─ sensitivity/
   └─ subgroup/
```

중간자료와 정밀 예측파일을 보존하려면 `TRANSPORT_PAPER_RUNTIME` 환경변수로 별도의 장기보존 경로를 지정한다.

## 8. 재현 시 주의사항

1. 전체 실행은 44개 Optuna 연구와 대규모 OOF 예측을 포함하므로 장시간이 소요된다.
2. `--start-at`을 사용할 때는 앞 단계의 결과가 반드시 존재해야 한다.
3. 이력 피처는 캐시된 값을 무조건 신뢰하지 않고 분할 인덱스와 원천자료 해시를 확인한다.
4. 최종 앙상블 결과에는 `VotingClassifier.predict_proba()`만 사용한다.
5. 그림 2는 정적 편집 자원이므로 자동 실행 대상이 아니다.
6. 표 10 상대시점 조건의 Top 5% 라벨률 원 산출값은 `0.1106498565`이다. 고정된 v19 PDF의 표시값을 재현하기 위해 논문용 표에는 `11.07%`를 유지하며 원 정밀값은 재현 결과 CSV에 보존한다.
7. 재현 코드는 논문용 표와 그림을 생성한다.

## 9. 공개 저장소의 범위

이 저장소에는 다음 자료만 포함한다.

- 최종 재현 Python 코드
- 논문 표 1~11의 CSV
- 논문 그림 1~4와 그림 2 편집용 SVG
- 수치 ledger와 논문 자원 manifest
- 원고 대조 기록(`docs/ASSET_AUDIT.md`)
- 데이터 접근 및 전체 재현 안내
- 최종 자원 SHA-256 체크섬

다음 자료는 공개하지 않는다.

- 원자료와 PrimaryKey
- 검사건별 OOF·시간분할 예측확률
- fold별 이력 캐시
- Optuna SQLite DB 및 전체 trial 로그
- 로컬 실행 로그, PID, 임시파일

공개 패키지의 구문, 필수파일, 체크섬은 다음 명령으로 확인한다.

```powershell
python verify_release.py
```

## 10. 인용과 라이선스

이 저장소를 연구에 활용하는 경우 `CITATION.cff`의 저자 및 논문 정보를 인용한다. 대상 논문은 현재 「교통안전연구」 심사 중이며, 확정된 권·호·쪽수와 DOI는 게재 승인 후 갱신한다.

코드는 `LICENSE`에 명시된 MIT License로 배포한다. 데이터와 논문 PDF는 코드 라이선스의 적용대상이 아니며, 원자료는 원 배포처의 이용조건을 따른다. 심사 중인 논문 PDF는 이 저장소에 포함하지 않는다.
