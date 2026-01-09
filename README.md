# 💫YOLO를 사용한 웹켐 실시간 물체 감지

- yolo를 사용해서 웹켐으로 실시간 물체를 감지하도록 함
  
- 미쯔비시 PLC 와 통신하도록 파이썬 코드를 작성함
  
- cmd 창에서 아래 명령어를 입력해서 필요한 패키지 설치 필요
```cmd
py -m pip install --upgrade pip 
py -m pip install ultralytics opencv-python 
py -m pip install pymcprotocol
```

- runs 폴더는 학습시킨 yolo 모델임. 

- 파이썬 코드는 runs\detect\train6\weights 경로의 best.pt 를 사용함
