# 재현성 안내

## 재현 수준

본 공개본은 두 수준의 확인을 지원한다.

1. 자원 검증: 공개된 코드의 구문, 필수파일과 최종 표·그림의 SHA-256을 확인한다.
2. 전체 재현: 원자료에서 시작하여 Optuna 튜닝, 외부평가, 민감도 분석, Bootstrap, SHAP, 표·그림 생성을 다시 수행한다.

## 빠른 자원 검증

원자료와 외부 패키지 설치 없이 공개 패키지의 구조와 체크섬을 확인한다.

```powershell
python verify_release.py
```

전체 실행명령을 실제 수행하지 않고 확인한다.

```powershell
python run_full_reproduction.py --dry-run
```

## 전체 재현

```powershell
python -m pip install -r requirements.txt
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

전체 절차는 다음 순서로 진행된다.

```text
data → protocol → tuning → outer → sensitivity
→ post → verification → shap → artifacts
```

주요 설정은 다음과 같다.

- 난수 시드: 42
- 무작위 계층 외부 5-fold
- PrimaryKey 완전분리 외부 5-fold
- 시간분할: 2016\~2020년 학습, 2021~2022년 검증
- 내부 튜닝: 외부 학습 파티션 내부 3-fold
- 모델: HGB, LightGBM, XGBoost, CatBoost
- Optuna: 11개 외부 fold × 4개 모델 × 100 trial
- 앙상블: `VotingClassifier(voting="soft")`
- 최종 확률: `VotingClassifier.predict_proba()` 직접 출력
- 군집 Bootstrap: PrimaryKey 단위 500회
- SHAP: 외부 1번 fold, 검증 2,000건, 배경 100건

## 누수 차단 규칙

- 이력은 `source.TestDate < target.TestDate`를 만족하는 기록만 사용한다.
- 동일 PrimaryKey의 동일 연월 기록은 서로의 과거이력에서 제외한다.
- 외부 검증행은 학습·검증 이력의 원천으로 사용하지 않는다.
- 내부 검증행은 해당 내부 fold의 학습 피처 원천으로 사용하지 않는다.
- 검증 라벨은 동일 검증 파티션의 다른 검사건 피처에 사용하지 않는다.
- 분할 인덱스와 원천자료 해시가 일치하는 캐시만 재사용한다.

## 계산자원

기준 장비는 Intel Core Ultra 7 265K 20코어와 32GB 메모리였다.

- 전체 경과시간: 약 60~70시간
- 누적 CPU 사용량 근사치: 약 430~500 코어시간
- 원자료 제외 중간 산출물: 약 6.7GiB
- 원자료 포함 총 사용량: 약 8.5GiB
- 권장 여유공간: 12~15GiB 이상

CPU, 메모리, 저장장치 및 패키지 빌드에 따라 실행시간은 달라질 수 있다.

## 최종 산출물 비교

`checksums.sha256`은 공개된 `figures`와 `tables`의 최종 자원을 대상으로 한다. 전체 재현이 끝난 뒤 다음 명령으로 공개 기준값과 비교할 수 있다.

```powershell
python verify_release.py
```

표 10 상대시점 조건의 Top 5% 라벨률 원 산출값은 `0.1106498565`이다. 고정된 v19 논문 표의 표시값을 재현하기 위해 논문용 표에는 `11.07%`를 유지하고 원 정밀값은 재현 결과에 보존한다.
