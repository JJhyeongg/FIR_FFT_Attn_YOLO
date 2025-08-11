import os
import pandas as pd
from ultralytics import YOLO

data_name = 'data01'
experiment_name = "experiment_11"

# 모델 로드
model = YOLO(f"../experiments/{experiment_name}/train/weights/best.pt")

# 검증 실행
metrics = model.val(
    data    = f'../Dataset/{data_name}/data.yaml',
    imgsz = 640,
    device="0",
    project= f"../experiments/{experiment_name}",
    name= f'val'
)

# 클래스별 결과 수집
names = model.names
per_class_results = []

# 🔹 전체(all) 평균 성능 추가
per_class_results.append({
    'class_id': 'all',
    'class_name': 'all',
    'precision': round(metrics.box.p.mean(), 4),
    'recall': round(metrics.box.r.mean(), 4),
    'mAP50': round(metrics.box.ap50.mean(), 4),
    'mAP50-95': round(metrics.box.map.mean(), 4)
})

# 🔹 개별 클래스 성능 추가
for cls_id, name in names.items():
    precision = metrics.box.p[cls_id]
    recall = metrics.box.r[cls_id]
    map50 = metrics.box.ap50[cls_id]
    map5095 = metrics.box.maps[cls_id]

    per_class_results.append({
        'class_id': cls_id,
        'class_name': name,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'mAP50': round(map50, 4),
        'mAP50-95': round(map5095, 4)
    })

# DataFrame 생성
df = pd.DataFrame(per_class_results)

# 저장 경로 생성
output_dir = f'../experiments/{experiment_name}/csv'
os.makedirs(output_dir, exist_ok=True)  # 디렉터리 없으면 생성

# CSV 저장
save_path = os.path.join(output_dir, f'val_metrics_{experiment_name}.csv')
df.to_csv(save_path, index=False)
print(f'✔️ 저장 완료: {save_path}')
