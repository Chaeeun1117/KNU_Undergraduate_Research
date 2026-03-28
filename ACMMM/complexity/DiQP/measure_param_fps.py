import torch
from torchinfo import summary
from model import DiQP
import numpy as np

# 1. 환경 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
crop = 512
B, C = 1, 3
F_main = 3  # x용
F_guide = 2 # 가이드용

# 2. 모델 초기화 (test.py 설정값 반영)
model = DiQP(
    crop_size=crop, 
    embed_dim=15, 
    depths=[1, 2, 8, 8, 2, 8, 8, 2, 1], 
    win_size=8, 
    mlp_ratio=3., 
    token_projection='linear', 
    token_mlp='steff', 
    shift_flag=True,
    inputFrames=F_main
).to(device)
model.eval()

# 3. 입력 데이터 준비 (torchinfo 및 FPS 측정용)
x = torch.randn(B, C, F_main, crop, crop).to(device)
around = torch.randn(B, C, F_guide, crop, crop).to(device)
aheadCropped = torch.randn(B, C, F_guide, crop, crop).to(device)
aheadScaled = torch.randn(B, C, F_guide, crop, crop).to(device)
loc = torch.stack([torch.randint(0, 10, (B,)) for _ in range(6)], dim=0).to(device).long()
weightDecay = torch.tensor([0.5]).view(B, 1, 1, 1, 1).to(device)

# --- [PART 1] torchinfo를 이용한 Parameter 계산 ---
print("\n" + "="*30)
print("1. Model Summary (torchinfo)")
print("="*30)

# DiQP는 입력이 여러 개이므로 input_data 리스트로 전달합니다.
model_stats = summary(
    model, 
    input_data=[x, around, aheadCropped, aheadScaled, loc, weightDecay],
    col_names=["input_size", "output_size", "num_params", "mult_adds"],
    depth=3,
    device=device
)
print(model_stats)


# --- [PART 2] FPS 측정 (GPU Event 이용) ---
print("\n" + "="*30)
print("2. FPS Measurement")
print("="*30)

repetitions = 100
timings = []

# Warm-up: GPU를 예열합니다 (처음 몇 번은 연산이 느릴 수 있음)
print("Warming up...")
for _ in range(10):
    _ = model(x, around, aheadCropped, aheadScaled, loc, weightDecay)

# 실제 측정
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

print(f"Measuring over {repetitions} iterations...")
with torch.no_grad():
    for i in range(repetitions):
        starter.record()
        _ = model(x, around, aheadCropped, aheadScaled, loc, weightDecay)
        ender.record()
        
        # GPU 연산이 완료될 때까지 대기 (동기화)
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender) # 단위: ms
        timings.append(curr_time)

# 통계 계산
avg_time_ms = sum(timings) / repetitions
avg_time_sec = avg_time_ms / 1000.0
fps = 1 / avg_time_sec

print(f"\n[Result] Average Latency: {avg_time_ms:.2f} ms")
print(f"[Result] FPS: {fps:.2f} frames/sec")
