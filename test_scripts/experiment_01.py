import torch
import torchvision.transforms as T
import cv2
import os
import numpy as np
from ultralytics import YOLO
import yaml

# 모델 로드
experiment_name = "experiment_13"
model_path = f'../experiments/{experiment_name}/train/weights/best.pt'
yaml_path = f"../configs/{experiment_name}.yaml"
model = YOLO(model_path)
model.eval()

def extract_layer_types(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    layer_types = []
    for section in ['backbone', 'head']:
        for layer in config.get(section, []):
            # layer[2]에 타입이 있음 (예: Conv, C3k2, Detect 등)
            if isinstance(layer[2], str):
                layer_types.append(layer[2])
            elif isinstance(layer[2], list):
                layer_types.append(layer[2][0])
    return layer_types

# YAML에서 레이어 타입 추출

layer_types = extract_layer_types(yaml_path)

# Hook 함수 정의
feature_maps = {}
def make_hook(idx, name):
    def hook_fn(module, input, output):
        def save_output(out, key):
            if isinstance(out, torch.Tensor):
                feature_maps[key] = out.detach().cpu()
            elif isinstance(out, (tuple, list)):
                for j, o in enumerate(out):
                    save_output(o, f"{key}_item{j}")
        save_output(output, f"{idx:02d}_{name}")
    return hook_fn

# 전체 레이어에 Hook 등록
handles = []
for idx, name in enumerate(layer_types):
    try:
        layer = model.model.model[idx]
        handles.append(layer.register_forward_hook(make_hook(idx, name)))
        # PhaseIFFTStack 파라미터 출력
        if name == "PhaseIFFTStack":
            print(f"PhaseIFFTStack params at layer {idx}:")
            print(f"  cut_low: {getattr(layer, 'cut_low', None)}")
            print(f"  cut_high: {getattr(layer, 'cut_high', None)}")
            print(f"  norm: {getattr(layer, 'norm', None)}")
            print(f"  learnable: {getattr(layer, 'learnable', None)}")
    except IndexError:
        print(f"Warning: Layer {idx} ({name}) not found in model")
        break

# 이미지 전처리 및 모델 통과
img_path = "./ex_v.jpg"
results = model.predict(source=img_path, imgsz=640, save=False, verbose=False)
# Hook된 feature map 저장 (레이어별로 폴더 분리)
base_dir = f"./backbone/{experiment_name}_backbone_output"
os.makedirs(base_dir, exist_ok=True)
for name, fmap in feature_maps.items():
    if "Detect" in name:
        print(f"{name}: Multiple outputs (detection layer)")
        continue

    # 첫 번째 레이어(raw) 값 출력
# ...existing code...
    # 첫 번째 레이어(raw) 값 출력
    if name.startswith("00_nn.Identity"):
        print(f"Raw feature map (Identity layer) min: {fmap.min()}, max: {fmap.max()}")
        print(f"Raw feature map (Identity layer) sample values:\n{fmap.flatten()[:10]}")
# ...existing code...

    fmap = fmap.squeeze(0)  # [C, H, W]
    print(f"{name}: {fmap.shape}")
    layer_dir = os.path.join(base_dir, name)
    os.makedirs(layer_dir, exist_ok=True)

    for i in range(fmap.shape[0]):
        channel = fmap[i]
        channel = (channel - channel.min()) / (channel.max() - channel.min() + 1e-8)
        channel_img = (channel.numpy() * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(layer_dir, f"channel_{i:04d}.png"), channel_img)

# Hook 제거
for h in handles:
    h.remove()