import torch
from fvcore.nn import FlopCountAnalysis, flop_count_table
from model import DiQP

device = "cpu"
crop = 512
B, C = 1, 3
F_main = 3  # 메인 스트림용 프레임 (Downsample/OutputProj용)
F_guide = 2 # 가이드 스트림용 프레임 (내부 하드코딩된 2*channels용)

# DiQP 초기화 (test.py의 파라미터와 완벽 일치)
model = DiQP(
    crop_size=crop, 
    embed_dim=15, 
    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], 
    win_size=8, 
    mlp_ratio=3., 
    token_projection='linear', 
    token_mlp='steff', 
    shift_flag=True,
    inputFrames=F_main # 메인 프레임 설정
).to(device)

model.eval()

# Dummy Inputs 생성 (각기 다른 프레임 수 적용)
x = torch.randn(B, C, F_main, crop, crop).to(device) # 메인: 3
around = torch.randn(B, C, F_guide, crop, crop).to(device) # 가이드: 2
aheadCropped = torch.randn(B, C, F_guide, crop, crop).to(device) # 가이드: 2
aheadScaled = torch.randn(B, C, F_guide, crop, crop).to(device) # 가이드: 2

# loc 및 weightDecay (이전과 동일)
loc = torch.stack([
    torch.randint(0, 86, (B,)), torch.randint(0, 151, (B,)),
    torch.randint(0, 8192, (B,)), torch.randint(0, 4320, (B,)),
    torch.randint(0, crop, (B,)), torch.randint(0, crop, (B,))
], dim=0).to(device).long()

weightDecay = torch.tensor([0.5]).view(B, 1, 1, 1, 1).to(device)

# 측정 시작
with torch.no_grad():
    try:
        # 모델 실행 테스트
        inputs = (x, around, aheadCropped, aheadScaled, loc, weightDecay)
        out = model(*inputs)
        print("Forward Pass Success! Output shape:", out.shape)
        
        # GFLOPs 계산
        flops = FlopCountAnalysis(model, inputs)
        print(f"\nTotal GFLOPs: {flops.total() / 1e9:.4f}")
        print(flop_count_table(flops, max_depth=3))
        
    except Exception as e:
        print(f"Failed: {e}")
