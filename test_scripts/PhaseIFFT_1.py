    
import torch
import cv2
from torchvision.utils import save_image
import torch.nn as nn
class PhaseIFFTStack(nn.Module):
    """
    FFT → 주파수 대역 분리(low, high) → IFFT → [B,2,H,W]
      - 채널 0: low  (r <= cut_low)
      - 채널 1: high (r >= cut_high)

    Args:
        c1 (int): 입력 채널 수 (RGB=3 가정 시 자동 Y 변환)
        c2 (int): 항상 2로 고정 (low, high)
        cut_low (float): 저역 컷오프 (주파수 반경, 권장 범위 0~0.5)
        cut_high (float): 고역 컷오프 (권장 범위 0~0.5, cut_high >= cut_low)
        norm (bool): 대역 복원 결과를 각 샘플별 [0,1] 정규화 여부
        eps (float): 정규화 안정성용 epsilon
    Notes:
        - 주파수 반경은 torch.fft.fftfreq 기반(단위: cycles/pixel), 최대 0.5 부근.
        - AMP 혼합정밀 학습에서도 안전하도록 autocast 비활성 + float32 강제.
    """

    def __init__(self, c1: int, c2: int = 2,
                 cut_low: float = 0.11, cut_high: float = 0.40,
                 norm: bool = False, eps: float = 1e-6):
        super().__init__()
        assert c2 == 2, "PhaseIFFTStack은 2채널(low, high)만 지원합니다."
        self.c2 = 2
        self.cut_low = float(cut_low)
        self.cut_high = float(cut_high)
        self.norm = bool(norm)
        self.eps = float(eps)

        # RGB -> Y 변환 가중치
        self.register_buffer(
            "rgb2y",
            torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1),
            persistent=False
        )

        self._sanitize_cuts()

    def _sanitize_cuts(self):
        # 권장 범위 0~0.5로 클램프, cut_high >= cut_low 강제
        lo = max(0.0, min(0.5, self.cut_low))
        hi = max(0.0, min(0.5, self.cut_high))
        if hi < lo:
            hi = lo
        self.cut_low, self.cut_high = lo, hi

    def _make_masks(self, H: int, W: int, device, dtype):
        # 주파수 그리드 (cycles/pixel). 범위 대략 [-0.5, 0.5)
        fy = torch.fft.fftfreq(H, device=device, dtype=dtype).view(-1, 1).repeat(1, W)
        fx = torch.fft.fftfreq(W, device=device, dtype=dtype).view(1, -1).repeat(H, 1)
        r = torch.sqrt(fx * fx + fy * fy)  # 0 ~ ~0.5

        low_mask = (r <= self.cut_low)
        high_mask = (r >= self.cut_high)
        return low_mask, high_mask

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,C,H,W] (C=3 권장)
        return: [B,2,H,W]  (0=low, 1=high), 입력 dtype으로 반환
        """
        orig_dtype = x.dtype

        # RGB -> Y (C==3이면)
        if x.shape[1] == 3:
            # rgb2y는 buffer라 x dtype에 맞춰 cast
            x = (x * self.rgb2y.to(dtype=x.dtype, device=x.device)).sum(1, keepdim=True)

        # FFT 연산은 float32로 고정 (AMP off)
        x32 = x.float()

        B, C, H, W = x32.shape
        device = x32.device

        # 2D FFT
        F2 = torch.fft.fft2(x32, norm='ortho')              # [B,1,H,W]
        # 주파수 마스크
        low_m, high_m = self._make_masks(H, W, device, dtype=x32.dtype)  # bool
        low_m = low_m.view(1, 1, H, W)
        high_m = high_m.view(1, 1, H, W)

        outs = []
        for m in (low_m, high_m):
            comp = F2 * m                                    # 대역 필터링
            img = torch.fft.ifft2(comp, norm='ortho').abs()  # magnitude 사용
            if self.norm:
                mn = img.amin(dim=(-2, -1), keepdim=True)
                mx = img.amax(dim=(-2, -1), keepdim=True)
                img = (img - mn) / (mx - mn + self.eps)
            outs.append(img)

        y = torch.cat(outs, dim=1)  # [B,2,H,W]
        return y.to(dtype=orig_dtype)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 필터 모듈 생성
    m = PhaseIFFTStack(c1=3, c2=2, cut_low=0.11, cut_high=0.11, norm=False).to(device)
    m.eval()

    # ---- 이미지 읽기 ----
    img_path = "ex_s.jpg"  # 테스트할 이미지 경로
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {img_path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_tensor = torch.from_numpy(rgb).float() / 255.0  # [H,W,3], 0~1
    rgb_tensor = rgb_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # [1,3,H,W]

    # ---- 필터 적용 ----
    with torch.inference_mode():
        y = m(rgb_tensor)  # [1,2,H,W]

    print("입력:", rgb_tensor.shape, rgb_tensor.dtype)
    print("출력:", y.shape, y.dtype)  # [1,2,H,W]

    low, high = y[:, 0:1], y[:, 1:2]
    print(f"low  - min:{low.min().item():.4f} max:{low.max().item():.4f} mean:{low.mean().item():.4f}")
    print(f"high - min:{high.min().item():.4f} max:{high.max().item():.4f} mean:{high.mean().item():.4f}")

    # ---- 결과 저장 ----
    save_image(low,  "phase_low.png")
    save_image(high, "phase_high.png")
    print("저장 완료: phase_low.png, phase_high.png")

if __name__ == "__main__":
    main()
