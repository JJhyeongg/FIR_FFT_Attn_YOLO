import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# PhaseIFFTStack 정의 (앞서 주신 코드 사용)
# ─────────────────────────────────────────────────────────────────────────────
class PhaseIFFTStack(nn.Module):
    def __init__(self, c1, c2=3, cut_low=0.11, cut_high=1.0, norm=True, eps=1e-6):
        super().__init__()
        self.cut_low, self.cut_high = cut_low, cut_high
        self.norm, self.eps = norm, eps
        self.register_buffer("rgb2y",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1,3,1,1))
        self.c2 = 3

    def _masks(self, H, W, dev):
        fy = torch.fft.fftfreq(H, device=dev).view(-1,1).repeat(1,W)
        fx = torch.fft.fftfreq(W, device=dev).view(1,-1).repeat(H,1)
        r  = torch.fft.fftshift((fx**2 + fy**2).sqrt())
        low  = (r <= self.cut_low)
        mid  = (r >  self.cut_low)
        high = (r >  self.cut_high)
        return low, mid, high

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        if x.shape[1] == 3:
            x = (x * self.rgb2y.to(x.dtype)).sum(1, keepdim=True)
        F = torch.fft.fft2(x.float(), norm='ortho')

        outs = []
        for m in self._masks(x.shape[2], x.shape[3], x.device):
            comp = torch.fft.ifftshift(F * m)
            y = torch.fft.ifft2(comp, norm='ortho').abs()
            if self.norm:
                y = torch.clamp(y, 0, 255) 
            outs.append(y)
        return torch.cat(outs, 1).to(x.dtype)

# ─────────────────────────────────────────────────────────────────────────────
# 이미지 로딩 및 모델 적용
# ─────────────────────────────────────────────────────────────────────────────
def run_phase_fft_on_image(img_path):
    # 이미지 로드 및 전처리
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor_img = to_tensor(rgb).unsqueeze(0).cuda()  # [1,3,H,W]

    # 모델 초기화 및 추론
    model = PhaseIFFTStack(c1=3).cuda().eval()
    with torch.no_grad():
        out = model(tensor_img)  # [1,3,H,W]

    return out[0]  # [3,H,W]

# ─────────────────────────────────────────────────────────────────────────────
# 결과 시각화
# ─────────────────────────────────────────────────────────────────────────────
def visualize_phase_outputs(out_tensor):
    titles = ['Low Frequency', 'Mid Frequency', 'High Frequency']
    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(titles[i])
        plt.axis('off')
        plt.imshow(to_pil_image(out_tensor[i].cpu()))
    plt.tight_layout()
    plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 사용 예시
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    img_path = 'ex_s.jpg'  # 이미지 경로를 여기에 입력
    out_tensor = run_phase_fft_on_image(img_path)
    visualize_phase_outputs(out_tensor)
