# result summary

### DiQP
Total GFLOPs: 591.2812
Total params: 79,428,874
FPS: 4.03 frames/sec

### CODiff
Total GFLOPs: 2368.1444
Total params: 2,453,269,290
FPS: 6.57 frames/sec

# DiQP Complexity Analysis

## Environment
- **Input setting**: complexity measured on the evaluation input used in the benchmark
- **Analysis tools**: `fvcore`, `torchinfo`

---

## Overall Complexity

| Metric | Value |
|---|---:|
| **Total Parameters** | **79,428,874** |
| **Trainable Parameters** | **79,428,874** |
| **Non-trainable Parameters** | **0** |
| **Total FLOPs (fvcore)** | **591.281 GFLOPs** |
| **Total Mult-Adds (torchinfo)** | **78.05 G** |
| **Average Latency** | **248.27 ms** |
| **Inference FPS** | **4.03** |

---

## Module-wise FLOPs (fvcore)

| Module | Parameters | FLOPs |
|---|---:|---:|
| **Full model** | 79.373M | **0.591T** |
| `input_proj.proj.0` | 1.23K | 0.956G |
| `output_proj.proj.0` | 2.433K | 1.911G |
| `lost` | 55.492K | 3.072K |
| `encoderGuide.encoder` | 57.354K | 6.176G |
| `decoderGuide` | 39.27K | 1.419G |
| `encoderlayer_0.blocks.0` | 22.275K | 7.892G |
| `dowsample_0.conv.0` | 28.83K | 5.662G |
| `encoderlayer_1.blocks` | 0.17M | 13.2G |
| `dowsample_1.conv.0` | 0.104M | 5.096G |
| `encoderlayer_2.blocks` | 2.657M | 47.634G |
| `dowsample_2.conv.0` | 0.392M | 4.813G |
| `encoderlayer_3.blocks` | 10.498M | 45.051G |
| `dowsample_3.conv.0` | 1.498M | 4.601G |
| `conv.blocks` | 10.433M | 10.94G |
| `upsample_0.deconv` | 0.27M | 1.094G |
| `decoderlayer_0.blocks` | 41.731M | **0.175T** |
| `upsample_1.deconv` | 0.127M | 1.823G |
| `decoderlayer_1.blocks` | 10.498M | **0.180T** |
| `upsample_2.deconv` | 32.34K | 1.852G |
| `decoderlayer_2.blocks` | 0.664M | 47.634G |
| `upsample_3.deconv` | 8.28K | 1.887G |
| `decoderlayer_3.blocks.0` | 85.05K | 26.401G |

> **Observation:** The main computational bottleneck lies in the **decoder transformer blocks**, especially `decoderlayer_0` and `decoderlayer_1`.

---

## Memory Estimate (torchinfo)

| Metric | Value |
|---|---:|
| **Input size** | 28.31 MB |
| **Forward / backward pass size** | 20,368.66 MB |
| **Parameter size** | 317.19 MB |
| **Estimated total size** | **20,714.16 MB** |

---

## Notes

### 1) FLOPs vs Mult-Adds
Two profiling tools were used:

- **fvcore** → reports **FLOPs**
- **torchinfo** → reports **mult-adds**

These values are not always numerically identical because each tool uses a slightly different counting convention.

### 2) Embedding-heavy modules
Some embedding-related components (e.g., `lost.stepEmbedding`, `frameNoEmbedding`, `widthEmbedding*`, `lentghEmbedding*`) show **0 FLOPs**.

> This is expected, since lookup-based embedding operations contribute parameters and memory usage, but are often not counted as arithmetic FLOPs.

---

## Summary

- The model has **79.4M trainable parameters**.
- The total computational complexity is approximately **591.3 GFLOPs**.
- The model runs at approximately **4.03 FPS** (**248.27 ms latency**) under the tested setting.
- The **decoder transformer blocks** dominate the total computation cost.

# CODiff Complexity Analysis

## Environment
- **Input resolution**: `1 × 3 × 512 × 512`
- **Analysis tools**: `fvcore`, `torchinfo`

---

## Overall Complexity

| Metric | Value |
|---|---:|
| **Total Parameters** | **2,453,269,290** |
| **Trainable Parameters** | **117,658,420** |
| **Non-trainable Parameters** | **2,335,610,870** |
| **Total FLOPs (fvcore)** | **2.368 TFLOPs** |
| **Total Mult-Adds (torchinfo)** | **2.22 T** |
| **Average Latency** | **152.30 ms** |
| **Inference FPS** | **6.57** |

---

## Module-wise FLOPs (fvcore)

| Module | Parameters | FLOPs |
|---|---:|---:|
| **Full model** | 1.043G | **2.368T** |
| `codiff` | 0.968G | 2.184T |
| `codiff.vae` | 85.131M | **1.827T** |
| `codiff.unet` | 0.882G | 0.356T |
| `codiff.proj` | 0.525M | 0.134G |
| `cave` | 74.887M | 0.184T |
| `cave.m_head` | 1.792K | 0.453G |
| `cave.m_down1` | 0.329M | 79.591G |
| `cave.m_down2` | 1.706M | 79.524G |
| `cave.m_down3` | 6.82M | 19.873G |
| `cave.m_body_encoder` | 18.883M | 4.833G |

> **Observation:** The computational cost is dominated by the **VAE** (`codiff.vae`), followed by the **UNet**.

---

## Memory Estimate (torchinfo)

| Metric | Value |
|---|---:|
| **Input size** | 3.15 MB |
| **Forward / backward pass size** | 11,415.56 MB |
| **Parameter size** | 1,991.06 MB |
| **Estimated total size** | **13,409.77 MB** |

---

## Notes

### 1) Unsupported operators in FLOPs analysis
The following operators were reported as unsupported by `fvcore` during tracing:

- `aten::scaled_dot_product_attention`
- `aten::silu`
- `aten::gelu`
- `aten::pad`
- `aten::add`, `aten::mul`, `aten::div`, etc.

> Therefore, the reported FLOPs should be interpreted as an **approximate estimate**, not an exact count.

### 2) Unused / untraced submodules
Some submodules (e.g., parts of `cave.m_body_decoder`, `cave.m_up*`, `cave.qf_pred`) were reported as **never called during trace**.

> These modules may either:
> - be unused in the current inference path, or
> - be accessed indirectly in a way that `fvcore` tracing could not capture.

As a result, their individual FLOPs may appear as missing or zero, while their parent module may still contribute partially to the total.

---

## Summary

- The model contains **2.45B parameters**, but only **117.7M are trainable**.
- The total computational complexity is approximately **2.37 TFLOPs** per forward pass at **512×512** input resolution.
- The model runs at approximately **6.57 FPS** (**152.30 ms latency**) under the tested setting.
- The **VAE** is the primary computational bottleneck.
