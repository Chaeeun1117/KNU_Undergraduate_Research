import os
import sys
import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis, flop_count_table
from torchinfo import summary
from argparse import Namespace

# 1. 경로 설정
current_dir = os.path.abspath(os.getcwd())
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "diffusion"))
sys.path.append(os.path.join(current_dir, "diffusion", "my_utils"))
sys.path.append(os.path.join(current_dir, "utils"))

from cave.cave import CaVE
from diffusion.codiff import CODiff_test

# 측정용 전체 모델 정의
class FullCODiffPipeline(nn.Module):
    def __init__(self, codiff_model, cave_model):
        super().__init__()
        self.codiff = codiff_model
        self.cave = cave_model

    def forward(self, lq_raw, lq):
        # 모든 입력을 모델의 가중치 타입(Half)으로 맞춤
        dtype = next(self.codiff.unet.parameters()).dtype
        lq_raw = lq_raw.to(dtype)
        lq = lq.to(dtype)
        
        visual_embedding = self.cave.get_visual_embedding(lq_raw)
        visual_embedding = visual_embedding.to(dtype)
        
        out = self.codiff(lq, visual_embedding)
        return out

# 2. 가상 설정 (mixed_precision을 fp16으로 설정)
args = Namespace(
    pretrained_model="model_zoo/sd-2.1-base",
    codiff_path="model_zoo/codiff.pkl",
    cave_path="model_zoo/cave.pth",
    mixed_precision="fp16", # fp16으로 설정하여 모델 로드 최적화
    merge_and_unload_lora=False,
    vae_decoder_tiled_size=224,
    vae_encoder_tiled_size=1024,
    latent_tiled_size=96,
    latent_tiled_overlap=32
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Initializing models and forcing Half precision...")
try:
    codiff_model = CODiff_test(args).to(device)
    cave_model = CaVE(in_nc=3, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='BR').to(device)
    cave_model.load_state_dict(torch.load(args.cave_path), strict=True)
    
    full_pipeline = FullCODiffPipeline(codiff_model, cave_model).to(device)
    
    # [핵심] 모델 전체를 Half(fp16)로 강제 전환
    full_pipeline.half()
    full_pipeline.eval()
except Exception as e:
    print(f"\n[Initialization Failed] {e}")
    sys.exit()

# 3. 입력 데이터 준비 (Half 정밀도)
crop = 512
lq_raw = torch.randn(1, 3, crop, crop).to(device).half()
lq = (lq_raw * 2 - 1).half()
inputs = (lq_raw, lq)

print("\n" + "="*50)
print("Analyzing Complexity (fvcore + torchinfo)...")
print("="*50)

# --- [PART 1] fvcore FLOPs 측정 ---
print("\n[1] Analyzing FLOPs (fvcore)...")
try:
    # fvcore는 JIT tracing을 사용하므로 타입이 일치하면 정확히 동작합니다.
    with torch.no_grad():
        flops = FlopCountAnalysis(full_pipeline, inputs)
        print(f"\nTotal GFLOPs: {flops.total() / 1e9:.4f}")
        print(flop_count_table(flops, max_depth=2))
except Exception as e:
    print(f"\n[fvcore Error] {e}")

# --- [PART 2] torchinfo Parameters 측정 ---
print("\n[2] Analyzing Parameters (torchinfo)...")
try:
    # 파라미터 수는 정밀도와 무관하게 동일합니다.
    print(summary(full_pipeline, input_data=inputs, device=device, verbose=0))
except Exception as e:
    print(f"\n[torchinfo Error] {e}")

# --- [PART 3] FPS 측정 ---
print("\n[3] Measuring FPS (50 iterations)...")
repetitions = 50
starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# Warm-up
with torch.no_grad():
    for _ in range(5):
        _ = full_pipeline(*inputs)

timings = []
with torch.no_grad():
    for i in range(repetitions):
        starter.record()
        _ = full_pipeline(*inputs)
        ender.record()
        torch.cuda.synchronize()
        timings.append(starter.elapsed_time(ender))

avg_time = sum(timings) / repetitions
print(f"Average Latency: {avg_time:.2f} ms")
print(f"Inference FPS: {1000.0 / avg_time:.2f}")
