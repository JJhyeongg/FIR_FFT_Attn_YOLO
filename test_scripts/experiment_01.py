import torch
import torchvision.transforms as T
import cv2
import os
import numpy as np
from ultralytics import YOLO

# 모델 로드
model_path = '../model/experiment_01.pt'
model = YOLO(model_path)
model.eval()

# yaml 구조에 맞춘 레이어 이름 리스트 (0~27)
layer_names = [
    "Identity",           # 0: raw
    "PhaseIFFTStack",     # 1
    "ChSelect_low",       # 2
    "ChSelect_high",      # 3
    "Conv_P1",            # 4
    "Conv_P2",            # 5
    "C3k2_P2",            # 6
    "Conv_P3",            # 7
    "C3k2Gated_P3",       # 8
    "Conv_P4",            # 9
    "C3k2Gated_P4",       # 10
    "Conv_P5",            # 11
    "C3k2_P5",            # 12
    "SPPF",               # 13
    "C2PSA",              # 14
    "Upsample_P5",        # 15
    "Concat_P4",          # 16
    "C3k2_head_P4",       # 17
    "Upsample_P4",        # 18
    "Concat_P3",          # 19
    "C3k2_head_P3",       # 20
    "Conv_head_P4",       # 21
    "Concat_head_P4",     # 22
    "C3k2_head_P4_2",     # 23
    "Conv_head_P5",       # 24
    "Concat_head_P5",     # 25
    "C3k2_head_P5",       # 26
    "Detect",             # 27
]

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
# 전체 레이어 0~27에 Hook 등록
handles = []
for idx, name in enumerate(layer_names):
    try:
        layer = model.model.model[idx]
        handles.append(layer.register_forward_hook(make_hook(idx, name)))
    except IndexError:
        print(f"Warning: Layer {idx} ({name}) not found in model")
        break

# 이미지 전처리 및 모델 통과
image = cv2.imread("./example_img/ex_s.jpg")
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_tensor = T.Compose([
    T.ToPILImage(),
    T.Resize((640, 640)),
    T.ToTensor(),
])(image_rgb)
img_tensor = img_tensor.unsqueeze(0)  # [1, 3, 640, 640]

with torch.no_grad():
    _ = model(img_tensor)

# Hook된 feature map 저장 (레이어별로 폴더 분리)
base_dir = "experiment_01_backbone_output"
os.makedirs(base_dir, exist_ok=True)
for name, fmap in feature_maps.items():
    # Detect 레이어는 여러 출력이 있으므로 처리 방식이 다름
    if "Detect" in name:
        print(f"{name}: Multiple outputs (detection layer)")
        continue

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