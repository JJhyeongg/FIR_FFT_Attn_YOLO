import torch
import cv2
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch import nn

# ─────────────────────────────────────────────────────────────────────────────
# PhaseIFFTStack 정의 (앞서 주신 코드 사용)
# ─────────────────────────────────────────────────────────────────────────────
class PhaseIFFTStack(nn.Module):
    """
    FFT 한 번 → 대역(mask) 3개 → IFFT → [B,3,H,W] (low, mid, high)
    """
    def __init__(self, c1, c2=3, cut_low=0.1, cut_high=1.0, norm=True, eps=1e-6):
        super().__init__()
        self.cut_low = cut_low
        self.cut_high = cut_high
        self.norm = norm
        self.eps = eps
        self.register_buffer("rgb2y", torch.tensor([0.2989, 0.5870, 0.1140]).view(1,3,1,1))
        self.c2 = 3

    def _r_norm(self, H, W, device):
        """ 중심 정규화된 반지름 맵 (NumPy 방식과 동일하게) """
        y = torch.arange(H, device=device).view(-1, 1)
        x = torch.arange(W, device=device).view(1, -1)
        cy, cx = H // 2, W // 2
        r = ((x - cx)**2 + (y - cy)**2).float().sqrt()
        r_norm = r / r.max()
        return r_norm  # shape: [H, W]

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        # 1. RGB → Grayscale
        if x.shape[1] == 3:
            x = (x * self.rgb2y.to(x.dtype)).sum(1, keepdim=True)  # [B,1,H,W]

        B, _, H, W = x.shape
        F = torch.fft.fft2(x.float(), norm='ortho')               # [B,1,H,W]
        F_shifted = torch.fft.fftshift(F, dim=(-2, -1))           # 중심 이동

        r_norm = self._r_norm(H, W, x.device)                     # [H,W]

        # 마스크 정의 (NumPy 방식과 동일)
        low_mask = (r_norm <= self.cut_low).float()
        mid_mask = ((r_norm > self.cut_low) & (r_norm <= self.cut_high)).float()
        high_mask = (r_norm > self.cut_high).float()

        masks = [low_mask, mid_mask, high_mask]
        outs = []

        for m in masks:
            m = m.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
            m = m.expand(B, 1, H, W)         # 배치에 맞게 확장
            F_filtered = F_shifted * m       # 마스크 곱
            F_ishift = torch.fft.ifftshift(F_filtered, dim=(-2, -1))
            img_back = torch.fft.ifft2(F_ishift, norm='ortho').abs()  # [B,1,H,W]

            if self.norm:
                img_back = torch.clamp(img_back, 0, 255)
                print(img_back.max())

            outs.append(img_back)

        return torch.cat(outs, dim=1).to(x.dtype)  # [B,3,H,W]

img_path = 'ex_s.jpg'
tensor_img = to_tensor(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)).unsqueeze(0).cuda()
model = PhaseIFFTStack(c1=3).cuda().eval()

with torch.no_grad():
    out = model(tensor_img)  # [1,3,H,W]

# 시각화
for i, title in enumerate(['Low', 'Mid', 'High']):
    plt.subplot(1, 3, i + 1)
    plt.title(title)
    plt.imshow(to_pil_image(out[0, i].cpu()), vmin=0, vmax=255)
plt.show()
