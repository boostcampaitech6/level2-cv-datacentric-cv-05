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
        <td align="center" width="150px">
            <a href="https://github.com/hee000"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/cde48fcd-8099-472b-9877-b2644954ec68"/></a>
            <br />
            <a href="https://github.com/hee000"><strong>조창희</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
              <a href="https://github.com/SangBeom-Hahn"><img height="120px" width="120px" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/1f7ed5a5-5e0f-46e4-85c6-31b9767dce41"/></a>
              <br />
              <a href="https://github.com/SangBeom-Hahn"><strong>한상범</strong></a>
              <br />
          </td>
    </tr>
</table>
<br/>

## 진료비 영수증 글자 검출 프로젝트
> ### 대회 개요
- 본 프로젝트는 진료비 영수증의 글자를 검출하는 것을 목표로한다.
- 영수증의 QR 코드 및 도장, 마스킹 부분은 검출 대상에서 제외된다.
- 본 대회는 데이터 중심적 사고를 통해 검출 성능을 높이므로 모델 및 손실 함수의의 개선을 지양한다.
<br/>

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

> ### Dataset
- 전체 이미지 개수 : 200장 (train 100장, test 100장)
- 이미지 크기 : 최대 가로크기 4032, 세로크기 4160, 최소 가로크기 645, 세로크기 803
- Input : 박스 좌표 및 angle, 글자가 포함된 이미지
- Output : 모델은 bbox 좌표, angle, score 값을 리턴
<br/>


> ### Data Cleansing
- 데이터를 직접 관찰하여 라벨링 된 방식에 대해 파악하였다. 데이터 클렌징을 모든 팀원이 나누어 진행을 하였기 때문에, 통합된 규칙이 필요하였다.
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
