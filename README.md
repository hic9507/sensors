# sensors

#### 학습 전에 데이터의 경로를 알맞게 설정
#### 학습 후에 저장될 Accuracy, Loss, AUC 그래프 및 가중치들이 저장될 경로 확인

#### ● 아나콘다 설치
#### 예시: https://seo-security.tistory.com/41 

#### ● NVIDIA. 드라이버 설치
#### https://www.nvidia.co.kr/Download/index.aspx?lang=kr 에서 자신의 GPU에 맞게 검색 후 설치

#### ● CUDA 및 Cudnn 설치 (CUDA-11.7, Cudnn: 8.2.0)
#### GPU: NVIDIA GeForce RTX 4090
#### https://developer.nvidia.com/cuda-toolkit-archive 에서 8.9 이상의 CUDA Toolkit 설치
#### https://developer.nvidia.com/rdp/cudnn-archive 에서 CUDA 버전에 맞는 Cudnn 설치
#### 예시: https://foreverhappiness.tistory.com/123  

#### ● 가상 환경 설치
#### conda create –n resnet_cbam python=3.8 -y
##### -n 뒤는 가상환경 명으로, 다른 것으로 변경 가능. python 버전은 3.8을 의미. -y는 수락한다는 의미.

#### ● 가상 환경 활성화
##### conda activate environment
##### conda activate resnet_cbam

#### ● 코드 디렉터리로 이동
##### cd 코드/디렉토리
##### cd C:/Users/USER/resnet_cbam

#### ● 라이브러리 설치
##### pip install –r requirements.txt
##### pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

#### 실험하려는 데이터셋에 해당하는 코드로 이동
##### 예시) cd UBI-Fights, cd RWF2000, cd UCSD Ped1, cd UCSD Ped2

##### 입력 방식마다 ResNet10, 18, 34, 50을 모두 실험해야하므로 한 파일 당 4번 실행해야함.
##### 코드 내의 model_depth 변수를 10, 18, 34, 50으로 바꿔가며 실행

#### ● 다음 명령어를 입력하고 아래 코드를 실행하세요.
####- 원본 UBI-Fights 데이터셋
##### 3x3 입력: 3x3.py
##### 4x3 입력: 4x3.py
##### 4x4 입력: 4x4.py
##### 5x3 입력: 5x3.py

#### - 언더샘플링한 UBI-Fights 데이터셋
##### 3x3 입력: undersample_3x3.py
##### 4x3 입력: undersample_4x3.py
##### 4x4 입력: undersample_4x4.py
##### 5x3 입력: undersample_5x3.py

#### - RWF2000 데이터셋
##### 3x3 입력: 3x3.py
##### 4x3 입력: 4x3.py
##### 4x4 입력: 4x4.py
##### 5x3 입력: 5x3.py

#### - UCSD Ped1
##### 3x3 입력: 3x3.py
##### 4x3 입력: 4x3.py
##### 4x4 입력: 4x4.py
##### 5x3 입력: 5x3.py

#### - UCSD Ped2
##### 3x3 입력: 3x3.py
##### 4x3 입력: 4x3.py
##### 4x4 입력: 4x4.py
##### 5x3 입력: 5x3.py
