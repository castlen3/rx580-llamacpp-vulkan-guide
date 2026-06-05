# RX 580 2048SP + llama.cpp Vulkan — Complete Troubleshooting Guide

[中文版](README.md)

> **TL;DR**: A "definitely-not-a-mining-card" RX 580 2048SP bought from Taobao, rescued by one environment variable — and it runs **35B MoE models** at 16.3 t/s (73% of GTX 1070 CUDA performance).

---

## Story: A "Gaming Card" from Taobao

Believe it or not — this RX 580 2048SP came from Taobao.

The seller swore up and down: "Not a mining card, pure gaming, previously owned by a university student." (Sure, buddy.)

An 8GB Polaris card from 2017. It struggles with modern AAA games, let alone running LLMs. The conventional wisdom is clear: old AMD GPU + Vulkan = disaster. LM Studio crashes. Ollama refuses to load. Every Google result says "just buy NVIDIA."

But 8GB of VRAM is 8GB of VRAM. In an era where video memory costs more than gold, throwing it away felt wrong.

So here we are — the complete journey from segfault hell to running a 35B MoE model at 16.3 t/s, stable.

## The Core Problem: Why Old AMD Cards Crash

If you try to use an RX 580 with LM Studio, Ollama, or the official llama.cpp Vulkan build, here's what happens 99% of the time:

1. **Model loading crashes midway (segfault)**
2. **No error message — just vanishes**
3. **Doesn't matter which model or quantization you pick**

It's not your model. It's not a llama.cpp bug. **It's an AMD driver + Polaris queue family selection issue.**

## Root Cause: The Queue Family Trap

The RX 580 exposes three Vulkan queue families:

| QF | Flags | Purpose |
|----|-------|---------|
| 0 | GRAPHICS \| COMPUTE \| TRANSFER | Combined queue |
| **1** | **COMPUTE \| TRANSFER** | **Dedicated compute queue** |
| 2 | TRANSFER | Dedicated transfer queue |

llama.cpp **deliberately avoids** QF0 (to prevent interfering with display rendering) and selects QF1 as its compute queue.

But AMD driver + Polaris **segfaults** when creating a device queue on QF1.

Vulkan validation layer confirms:

```
VUID-vkGetDeviceQueue-queueFamilyIndex-00384
vkGetDeviceQueue(): queueFamilyIndex is VK_QUEUE_FAMILY_IGNORED
```

## The Fix (One Line)

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1
```

This environment variable tells llama.cpp: "Stop avoiding the graphics queue. Use QF0."

One line. From crash to working. One line.

| llama.cpp Version | Supports This Env Var | Vulkan Works |
|-------------------|----------------------|--------------|
| b9519 (latest) | ✅ | ✅ |
| b9022 | ✅ | ✅ |
| b8626 | ✅ | ✅ |
| b6421 and older | ❌ | ❌ |

## Performance Benchmarks

### Small Model: Qwen3-1.7B Q6_K

| Mode | Prompt | Generation | Speedup |
|------|--------|------------|---------|
| CPU only | 78 t/s | 25 t/s | 1.0x |
| Vulkan GPU | 25 t/s | **66 t/s** | **2.6x** |

### MoE Model: Qwen3.6-35B-A3B Q4_K_M (20 GB)

This is the main event. A 20GB MoE model on an 8GB card. How? **Dense layers on GPU, expert layers on CPU.**

| Config | Prompt | Generation | VRAM | Speedup |
|--------|--------|------------|------|---------|
| CPU only | 3.8 t/s | 3.1 t/s | 0 | 1.0x |
| Vulkan `--cpu-moe` | 12.0 t/s | 10.6 t/s | ~1.8 GB | 3.4x |
| **Vulkan `--n-cpu-moe 30` (optimal)** | **22.6 t/s** | **16.3 t/s** | **~6.7 GB** | **5.3x** |
| GTX 1070 CUDA (reference) | — | 22.3 t/s | — | 7.2x |

RX 580 Vulkan achieves **73%** of GTX 1070 CUDA performance. The GTX 1070 cost more than double the RX 580 back in the day.

### Actual VRAM Usage

Tested: RX 580 can use up to **6.6 GB / 7.4 GB (89%)** — not the "only 4 GB" myth floating around the internet.

32K context with q8_0 KV cache runs stable.

### Key Parameters

| Parameter | Value | Why |
|-----------|-------|-----|
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` | `1` | **Required**, or segfault |
| `-ngl` | `99` | Offload as much as possible to GPU |
| `--n-cpu-moe` | `30` | 10 expert layers on GPU, VRAM limit |
| `-t` | `4` | Reduce CPU contention |
| `-fa` | `on` | Flash attention saves VRAM |
| `--cache-type-k` | `q8_0` | Halves KV cache size |
| `--cache-type-v` | `q8_0` | Halves KV cache size |
| `-fit` | `off` | Prevent auto-tuning |
| `--no-mmap` | — | Avoid mmap overhead |
| `-c` | `32768` | Max context (requires q8_0 cache) |

### `-tb` Thread Tuning

Under RX 580 2048SP + Qwen3.6-35B-A3B hybrid offload (`--n-cpu-moe 30`),
we tested `-tb` (`n_threads_batch`) at 8 and 12 vs the default of 4:

| `-tb` | Prefill | Decode | Verdict |
|-------|---------|--------|---------|
| 4 (default, same as `-t`) | ~78 tok/s | ~14 tok/s | Baseline |
| 8 | ~79 tok/s | ~15 tok/s | No improvement |
| 12 | ~80 tok/s | ~14 tok/s | No improvement |

**Conclusion: the prefill bottleneck is NOT in CPU batch threads.** Raising `-tb` provides
no meaningful throughput gain. Keep the original `-t 4` — **do not set `-tb` separately**.

## Tested Models

| Model | Quant | Size | Works | Generation |
|-------|-------|------|-------|------------|
| Qwen3-1.7B | Q6_K | 1.6 GB | ✅ | 66 t/s |
| Qwen3-4B | Q4_K_M | 2.3 GB | ✅ | ~60 t/s |
| **Qwen3.6-35B-A3B** | **Q4_K_M** | **20 GB** | ✅ | **16.3 t/s** |
| Qwen3-30B-A3B | Q4_K_M | 18 GB | ✅ | ~15 t/s |
| Qwen3-32B (Dense) | Q4_K_M | 19 GB | ⚠️ | Needs testing |

## Quick Start

### 1. Download llama.cpp Vulkan Build

```bash
cd ~
curl -L -o llama-vulkan.zip \
  https://github.com/ggml-org/llama.cpp/releases/download/b9519/llama-b9519-bin-win-vulkan-x64.zip
unzip -o llama-vulkan.zip -d ~/llama-vulkan
```

### 2. Verify GPU Detection

```bash
./llama-cli.exe --list-devices
# Should show: Vulkan0: AMD Radeon RX 580 2048SP
```

### 3. Run Small Model

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-cli.exe \
  -m Qwen3-1.7B-Q6_K.gguf \
  -ngl 99 -t 10 -c 4096 \
  --device Vulkan0
```

### 4. Run MoE Model (35B A3B)

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-cli.exe \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -t 4 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fit off --no-mmap --n-cpu-moe 30 \
  --device Vulkan0
```

### 5. Server Mode

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-server.exe \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -t 4 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fit off --no-mmap --n-cpu-moe 30 \
  --device Vulkan0 --port 8080
```

## Known Limitations

1. **Must set `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`**, or it crashes
2. MoE models need `--n-cpu-moe 30` (VRAM limit); full GPU offload causes OOM
3. 32K context + q8_0 KV cache is the VRAM ceiling (6.6 GB / 7.4 GB)
4. Polaris lacks fp16 / int dot product / matrix cores hardware support
5. llama.cpp b6421 and older don't support this env var

## Test Environment

| Item | Value |
|------|-------|
| OS | Windows 10 22H2 |
| CPU | Intel Core i7-6900K @ 3.20GHz (8C/16T) |
| RAM | 32 GB DDR4 |
| GPU | AMD Radeon RX 580 2048SP (Polaris, 8GB) |
| Driver | AMD Adrenalin 26.5.2 |
| Driver Vulkan API | 1.3.260 |
| llama.cpp | b9519 (2026-06-05) |

---

## Related Projects

- [castlen3/gtx1070-qwen3.6-35b-guide](https://github.com/castlen3/gtx1070-qwen3.6-35b-guide) — GTX 1070 same model optimization guide
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — llama.cpp source
