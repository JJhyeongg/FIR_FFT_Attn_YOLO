from ultralytics import YOLO

# 이전 run의 마지막 체크포인트로 모델 생성
model = YOLO('../experiments/experiment_09/train/weights/last.pt')

# 그대로 이어서 학습 (run 폴더의 args.yaml/hyp.yaml을 불러옵니다)
model.train(resume=True, epochs=1500, device='1')