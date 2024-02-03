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

## 재활용 쓰레기 검출 대회
> ### 대회 개요
- 본 프로젝트는 분리수거를 돕기 위한 모델을 제작하는 것이다.
- 주어진 사진에서 쓰레기를 Detection하여 환경 문제를 돕는다.
- 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기를 감지할 수 있도록 한다.
<br/>

> ### 팀 역할
|이름|역할|
|------|---|
|전체|EDA, 데이터 클렌징, 팀 분배 및 Wrap up report 작성, 데이터 증강 및 모델 성능 향상을 위한 각종 실험|
|양우희|하이퍼 파라미터 성능 실험|
|이영진|TTA 실험, Ensemble|
|조민지|Mosaic aug, CV (stratified group, multilabel stratified), Ensemble, CAM visualize|
|조수민|Detectron2를 이용한 모델 실험, 하이퍼 파라미터 성능 실험, Ensemble|
|조창희|Stage1 모델 실험, Ensemble|
|한상범|GitHub 전략 셋팅, 모델 실험 프로세스 개선, 모델 클래스별 튜닝|
<br/>

> ### WBS
<img width="1000" alt="image" src="https://github.com/boostcampaitech6/level2-objectdetection-cv-05/assets/78292486/d0acdc43-60c0-4992-90b4-6b3dcdc01447">
<br/>
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
- 전체 이미지 개수 : 9754장 (train 4883장, test 4871장)
- 10 class : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- 이미지 크기 : (1024, 1024)
- Input : 쓰레기 객체가 담긴 이미지와 bbox 정보(좌표, 카테고리)가 모델의 인풋으로 사용. bbox annotation은 COCO format으로 제공
- Output : 모델은 bbox 좌표, 카테고리, score 값을 리턴
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

- 가려짐이 심한 object에도 일일이 bounding box가 그려진 경우, 다른 물체에 부착된 스티커 등이 별개의 객체로 분류된 경우, 겹쳐진 모든 객체를 각각으로 판단하는 경우 등 범위의 일관성 논의가 필요했다.
- 또한, Class 표기가 이미지마다 다르게 적용된 것이 있었고, class의 정체를 파악하기 어려운 object가 있었다.
- 관찰 기록을 토대로 데이터의 일관성에서 어긋나는 데이터는 삭제할 수 있는 규칙을 정했다. 개인이 애매하다고 판단하는 데이터의 경우에는 우선 기록을 하고 추후 처리 방식을 의논하였다.
<br/>

> ### Model
추후에
<br/>

> ### Training
```bash
python train.py -c config.json
```
<br/>

> ### Inference
```bash
python live/inference.py -c <setting> -r <ckpt_dir> -d 0
```
<br/>

> ### 📂 File Tree
```
📦 level2-objectdetection-cv-05
├─ 📂EDA
│  └─ eda.ipynb
└─ 📂live
   ├─ 📂data_loader
   │  ├─ custom_data_loader.py
   │  └─ custom_test_data_loader.py
   ├─ 📂dataset
   │  └─ custom_dataset.py
   ├─ 📂model
   │  └─ custom_model.py
   ├─ 📂trainer
   │  └─ trainer.py
   ├─ 📂utils
   │  └─ util.py
   ├─ config.json
   ├─ inference.py
   ├─ parse_config.py
   ├─ test_config.py
   └─ train.py
```
