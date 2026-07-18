# 데이터 이용 안내

## 데이터 출처

본 연구는 DACON이 제공한 운수종사자 위험군 예측 자료를 사용하였다.

- 데이터 페이지: <https://dacon.io/competitions/official/236607/data>
- 관련 게시글: <https://dacon.io/competitions/official/236607/talkboard/415363>

데이터 이용조건, 접근권한과 재배포 가능 여부는 DACON 및 원 제공기관의 정책을 따른다. 본 GitHub 공개본에는 원자료를 포함하지 않는다.

## 필요한 파일구조

데이터를 내려받아 압축을 해제하면 분석 입력파일은 다음 구조에 있다.

```text
open_v2/
└─ data/
   ├─ train.csv
   └─ train/
      ├─ A.csv
      └─ B.csv
```

전체 재현 시 `train.csv`와 `train/`을 직접 포함하는 `data` 디렉터리를 명시한다.

```powershell
python run_full_reproduction.py --data-path "D:\data\open_v2\data"
```

Linux 또는 macOS의 예시는 다음과 같다.

```bash
python run_full_reproduction.py --data-path /path/to/open_v2/data
```

## 분석 단위와 공개 범위

- 분석 단위: 검사건 `Test_id`
- 종속변수: 데이터셋에서 제공된 위험군 `Label`
- 검사유형: 신규검사 A, 자격유지검사 B
- 논문 분석건수: 944,767건
- 논문 피처 수: 37개

본 저장소는 집계된 논문 표와 그림만 공개한다. 다음 자료는 포함하지 않는다.

- `PrimaryKey`
- 원자료 행
- 검사건별 OOF·시간분할 예측확률
- fold별 이력 캐시
- Optuna DB와 전체 trial 로그

## 무결성 확인

원자료 배포본의 공식 체크섬이 제공되는 경우 이 문서에 추가할 수 있다. 현재 공개 패키지는 원자료 파일명과 구조만 검증하며 원자료 자체의 체크섬은 포함하지 않는다.
