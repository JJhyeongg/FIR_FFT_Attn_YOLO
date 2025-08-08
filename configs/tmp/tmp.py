import torch
import torch.nn as nn
import torch.nn.functional as F

def resize_to(x: torch.Tensor, hw):
    """x: [B,C,Hx,Wx] → [B,C,h,w]"""
    h, w = hw
    if x.shape[-2:] != (h, w):
        x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
    return x

class C3k2Gated(C2f):
    """
    C3k2(+gate) with multi-channel mask
      inputs: x  또는 [x, mask]
      mask  : [B, Cm, Hm, Wm]  (예: Cm=2 for low/high)
        → resize_to(..., (h,w)) → 1x1 conv(Cm→self.c) → sigmoid → [B,self.c,h,w]
        → 각 bottleneck 출력에 a * (1 + α·m) 곱 게이트
    """
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True,
                 mask_ch: int = 1, use_alpha: bool = False, alpha_init: float = 0.0):
        super().__init__(c1, c2, n, shortcut, g, e)  # cv1, cv2, self.c 생성

        # 내부 블록(C3k 또는 Bottleneck) 구성
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g)
            for _ in range(n)
        )

        # 멀티채널 마스크 → self.c 채널 게이트로 투영
        self.mask_proj = nn.Conv2d(mask_ch, self.c, kernel_size=1, bias=True)
        nn.init.zeros_(self.mask_proj.weight)  # 초기 영향 최소화
        nn.init.zeros_(self.mask_proj.bias)

        # α 스칼라(옵션). 기본은 끔(=굳이 없어도 됨)
        self.use_alpha = bool(use_alpha)
        if self.use_alpha:
            self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))  # 보통 0.0로 중립 시작 추천

    def _gate(self, feat: torch.Tensor, mask: torch.Tensor | None):
        if mask is None:
            return feat
        # mask: [B,Cm,h,w] → 1x1 conv → sigmoid → [B,self.c,h,w]
        m = torch.sigmoid(self.mask_proj(mask))
        if self.use_alpha:
            return feat * (1.0 + self.alpha * m)
        else:
            return feat * (1.0 + m)

    def forward(self, inputs):
        # inputs: x  또는 [x, mask]
        if isinstance(inputs, (list, tuple)):
            x, raw_mask = inputs[0], inputs[1]
        else:
            x, raw_mask = inputs, None

        y = list(self.cv1(x).chunk(2, 1))  # [a, b]

        # C2f의 extend 구간만 후킹해서 게이트 삽입
        out = y[-1]
        for blk in self.m:
            t = blk(out)
            if raw_mask is not None:
                m = resize_to(raw_mask, t.shape[-2:])  # 다운샘플/업샘플로 공간 해상도 맞춤
                t = self._gate(t, m)
            y.append(t)
            out = t

        return self.cv2(torch.cat(y, 1))
