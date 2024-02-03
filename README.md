## CV 5팀 소개
> ### 멤버
<table>
    <tr height="160px">
        <td align="center" width="150px">
            <a href="https://github.com/woohee-yang"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/a1e74529-0abf-4d80-9716-4e8ae5ec8e72"/></a>
            <br/>
            <a href="https://github.com/woohee-yang"><strong>양우희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/jinida"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/28955c1d-fa4e-46b1-9d70-f98eb54109b2"/></a>
            <br />
            <a href="https://github.com/jinida"><strong>이영진</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/cmj5064"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/6388976d-d0bd-4ba6-bae8-6c7e6c5b3352"></a>
            <br/>
            <a href="https://github.com/cmj5064"><strong>조민지</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/ccsum19"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/9ad5ecc3-e5be-4738-99c2-cc6e7f3931cb"/></a>
            <br/>
            <a href="https://github.com/ccsum19"><strong>조수민</strong></a>
            <br />
        </td>
        <리|
|이영진|데이터 증강 및 데이터 전처리, Ensemble|
|조민지|데이터 전처리, Ensemble, 변경 가능한 하이퍼 파라미터 실험|
|조수민|데이터 증강 및 Enseble, Logging 기능 구현|
|조창희|외부 데이터 셋 수집 및 FP16 기능 구현|
|한상범|GitHub 전략 셋팅, 모델 실험 프로세스 개선선|
<br/>


> ### 개발환경
```bash
- Language : Python
- Environment
  - CPU : Intel(R) Xeon(R) Gold 5120
  - GPU : Tesla V100-SXM2 32GB × 1
- Framework : PyTorch
- Collaborative Tool : Git, Notion
```
<br/>

> ### 🔥 필수 라이브러리 설치
``` bash
pip install -r requirements.txt
```
<br/>

> ### Dataset
- 전체 이미지 개수 : 200장 (train 100장, test 100장)
- 이미지 크기 : 최대 가로크기 4032, 세로크기 4160, 최소 가로크기 645, 세로크기 803
- Input : 박스 좌표 및 angle, 글자가 포함된 이미지
- Output : 모델은 bbox 좌표, angle, score 값을 리턴
<br/>

> ### EDA
<img width="500" alt="image" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/e5bcdb15-4203-4d7e-aaab-238108903667">
<img width="500" alt="image" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/73f4d49e-d2b8-4411-824a-1e6eab004f44">
<br/>
<br/>

- 최초 EDA 결과, 클래스 불균형과 클래스별 각 annotation box 크기 편차가 심함을 알 수 있었다.
- 일반쓰레기는 데이터가 많은데 비해서 박스 크기의 중간값이 다른 클래스들에 비해 가장 작은 것을 볼수 있다.
- 또한, 배터리 클래스는 가장 적은데이터를 가지고 있는데 반해 비교적 일관성있는 크기를 가지고 있어 어떤 영향을 미칠지 훈련 결과를 통해 알아보기로 하였다.
- 각 클래스별 box 크기가 일관되지 않는 부분이 크고, 일반쓰레기의 경우 크기도 작아 잘못된 라벨링 결과가 있을 가능성이 높으므로 데이터 클랜징을 진행하기로 하였다.
<br/>

> ### Data Cleansing
- 데이터를 직접 관찰하여 라벨링 된 방식에 대해 파악하였다. 데이터 클렌징을 모든 팀원이 나누어 진행을 하였기 때문에, 통합된 규칙이 필요하였다.
<img width="500" alt="image" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/c3a66233-2d70-48db-abca-69fb76cafaec">
<br/>
<br/>

- 해당 데이터 셋을 리뷰한 뒤 클렌징 기법에 대한 논의를 진행하였다.
- 논의의 진행 결과로서 일관적인 레이블링이 필요할 것으로 판단되어 각 객체에 대해 약 20px이상 띄워져 있는 경우 다른 객체라 판단하였다.
- 뿐만 아니라 일정 크기 이하를 가지는 예를들어 쉼표와 마침표의 경우 학습에 어려움이 있을 것으로 판단되고 평가 진행시에도 큰 점수를 가져가지 않으므로 레이블링에서 제외하였다.
- 잘못 레이블링 된 경우(배경 레이블링, 박스의 크기를 작게 설정함)도 클렌징 대상에 포함하였다.
<br/>


> ### Training
```bash
python train.py --config "config path"
```
<br/>

> ### Inference
```bash
python inference.py --model-dir "model path"
```
<br/>

> ### 📂 File Tree
```
📦 level2-objectdetection-cv-05
├─ 📂EDA
│  └─ eda.ipynb
├─ 📂config
│  └─ base.yaml
├─ 📂data
│  ├─ dataset.py
│  ├─ east_dataset.py
│  ├─ preprocess.py
│  └─ augmentation.py
├─ 📂model
│  └─ model.py
├─ 📂utils
│  ├─ argparsers.py
│  ├─ create.py
│  ├─ detect.py
│  ├─ deteval.py
│  ├─ logger.py
│  ├─ loss.py
│  ├─ lr_scheduler.py
│  ├─ plot.py
│  └─ util.py
├─ train.py
├─ inference.py
├─ make_train.py
└─ visualize.py

```
