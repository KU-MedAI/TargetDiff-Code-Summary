# TargetDiff 재현 코드

TargetDiff 원본 코드를 참고해 학습과 샘플링 흐름을 간단히 재작성한 코드임. 이 저장소의 핵심 목적은 `62000.pt` checkpoint를 사용해 test set pocket 100개(`data_id=0..99`)에 대해 ligand를 샘플링하는 것임.

전체 흐름:

```text
train_diffusion.py -> checkpoint 생성
sampling.py        -> test set pocket별 ligand 샘플링
```

## 주요 파일

- `config.yml`: 학습 설정 파일
- `sampling.yml`: 샘플링 설정 파일. 기본 checkpoint는 `62000.pt`
- `train_diffusion.py`: diffusion model 학습 코드
- `sampling.py`: 특정 pocket에 대해 ligand를 샘플링하는 코드
- `dataset.py`, `diffusion.py`, `network.py`, `reconstruct.py`: 데이터 처리, 모델, diffusion, 분자 재구성 핵심 코드
- `run_sample_test100.sh`: `data_id=0..99` 전체에 대해 샘플링 실행
- `run_full_train_and_sample.sh`: 학습 수행 후 최신 checkpoint로 test100 샘플링 실행
- `build_lmdb.py`: pocket/ligand 파일에서 LMDB를 만드는 선택용 유틸리티
- `evaluate_diffusion.py`: 보조 평가 스크립트. 원본 TargetDiff의 `utils/` 계열 코드와 docking 도구 필요. 최소 샘플링 재현 경로에는 포함되지 않음

## 필요한 파일

데이터 파일은 용량이 커서 Git에 포함하지 않음. 실행 전에 아래 경로에 직접 준비 필요.

```text
data/crossdocked_pocket10_pose_split.pt
data/crossdocked_v1.1_rmsd1.0_pocket10_processed_final.lmdb
```

샘플링에 사용하는 checkpoint는 저장소에 포함.

```text
logs_diffusion_full/targetdiff_cjkim_full_gpu/checkpoints/62000.pt
```

위 경로는 `sampling.yml`과 `run_sample_test100.sh`의 기본값과 일치.

## 환경

필요 패키지:

```text
torch
torch_geometric
torch_scatter
rdkit
openbabel
lmdb
scipy
numpy
pyyaml
tqdm
tensorboard
```

특히 `torch`, `torch_geometric`, `torch_scatter`는 서로 호환되는 버전으로 설치 필요.

## Test100 샘플링

기본 실행:

```bash
bash run_sample_test100.sh
```

샘플 수, batch size, diffusion step 수 변경은 환경변수 사용.

```bash
NUM_SAMPLES=100 BATCH_SIZE=16 NUM_STEPS=1000 bash run_sample_test100.sh
```

결과 저장 위치:

```text
sampling_results_full_test100/
```

## 학습부터 샘플링까지 실행

학습을 다시 수행한 뒤, 생성된 최신 checkpoint로 test100 샘플링까지 실행하는 스크립트:

```bash
bash run_full_train_and_sample.sh
```

예시:

```bash
TRAIN_MAX_ITERS=71000 TRAIN_TAG=cjkim_full_gpu bash run_full_train_and_sample.sh
```

## Git에 포함하지 않는 파일

아래 파일과 폴더는 로컬 데이터 또는 실행 산출물이므로 `.gitignore`에 포함.

```text
data/
logs_diffusion*/              # 단, 62000.pt는 예외로 포함
sampling_results*/
targetdiff_eval_meta_full_test100/
sampling_runtime*.yml
*.lmdb
```
