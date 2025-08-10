import torch
import os
import multiprocessing
from ultralytics import YOLO
import shutil
import os
if __name__ == '__main__':
    # GPU 설정 (여러 GPU 사용 가능)
    epochs = 1500
    batch = 60
    imgsz = 640
    device = '0'  # 0번과 1번 GPU를 사용하도록 명시적으로 지정
    data_name = "data01"
    experiment_name = "experiment_07"
    config_name = "experiment_07n"
    model_name = "yolo11n"
    #model = YOLO(f'../configs/{config_name}.yaml').load(f'{model_name}.pt')

    model = YOLO(f'../configs/{config_name}.yaml')
    try:
        # Linux 
        n_cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        # Windows
        n_cpus = os.cpu_count() or multiprocessing.cpu_count()

    worker = 6      # 최소 1개는 남겨두기
    print(f"Using {worker} dataloader workers")

    # manual setting for dataloader workers
    worker = worker

    # 데이터 증강 설정
    results = model.train(
        data=f'../Dataset/{data_name}/data.yaml',
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        resume= True,
        workers=worker,
        device=device,
        seed = 42,      
        deterministic=True,     # 여러 GPU 지정
        amp=True,                # Mixed Precision (AMP) 활성화
        save=True,
        save_period=5000,
        project=f'../experiments/{experiment_name}',    # TensorBoard 로그가 저장될 디렉터리 경로 지정
        name=f'train',          # 실험 이름 지정 (해당 디렉터리에 저장됨)
        patience=100,  # Early Stopping을 위한 patience 설정 (예: 50 에포크)
    )