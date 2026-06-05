<p align="center">
  <a href="#-chinese"><img src="https://img.shields.io/badge/中文-Chinese-red?style=for-the-badge" alt="中文"></a>
  &nbsp;
  <a href="#-english"><img src="https://img.shields.io/badge/English-English-blue?style=for-the-badge" alt="English"></a>
</p>

---

> **TL;DR**: 一張淘寶買的「號稱非礦卡」RX 580 2048SP 老 A 卡，靠一個環境變數 `GGML_VK_ALLOW_GRAPHICS_QUEUE=1` 就能跑 **35B MoE 大模型**，達到 16.3 t/s（GTX 1070 CUDA 的 73%）。
>
> A "definitely-not-a-mining-card" RX 580 2048SP bought from Taobao, rescued by one environment variable — and it runs **35B MoE models** at 16.3 t/s (73% of GTX 1070 CUDA performance).

---

# 🇹🇼 Chinese

<details open>
<summary><b>📖 點擊展開/收合中文完整內容</b></summary>

---

## 故事：一張來自淘寶的「遊戲卡」

說出來你可能不信——這塊 RX 580 2048SP 是我在淘寶買的。

賣家信誓旦旦：「非礦卡，純遊戲卡，女大學生自用退役。」（笑）

8GB VRAM 的 Polaris 架構老卡，2017 年的東西了。跑 3A 大作都有點喘，更別提跑 LLM。多數人的直覺是：老 A 卡 + Vulkan = 災難。LM Studio 打不開，Ollama 載入失敗，Google 到的答案都是「換 NVIDIA 吧」。

但 8GB VRAM 就是 8GB VRAM。在顯存比金礦還貴的年代，丟掉太可惜。

於是有了這篇——從 segfault 地獄到 35B MoE 大模型穩定跑出 16.3 t/s 的踩坑全記錄。

## 老 A 卡跑 LLM 的核心問題

如果你拿 RX 580 直接用 LM Studio、Ollama、或是 llama.cpp 官方 Vulkan build，99% 會遇到以下狀況：

1. **模型載入到一半就 crash（segfault）**
2. **什麼錯誤訊息都沒有，就是閃退**
3. **換什麼模型、什麼量化格式都一樣**

這不是你的模型有問題，也不是 llama.cpp 有 bug——**這是 AMD 驅動 + Polaris 架構的 queue family 選擇問題**。

## 根本原因：Queue Family 的陷阱

RX 580 有三個 Vulkan queue family：

| QF | Flags | 用途 |
|----|-------|------|
| 0 | GRAPHICS \| COMPUTE \| TRANSFER | 混合佇列 |
| **1** | **COMPUTE \| TRANSFER** | **純計算佇列** |
| 2 | TRANSFER | 純傳輸佇列 |

llama.cpp 預設會**避開**包含 GRAPHICS 的 QF0（怕干擾畫面渲染），選擇 QF1 作為 compute queue。

但 AMD 驅動 + Polaris 在 QF1 上建立 device queue 時會**直接 segfault**。

用 Vulkan validation layer 抓到的錯誤：

```
VUID-vkGetDeviceQueue-queueFamilyIndex-00384
vkGetDeviceQueue(): queueFamilyIndex is VK_QUEUE_FAMILY_IGNORED
```

## 解決方案（就一行）

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1
```

這個環境變數告訴 llama.cpp：「不用避開 GRAPHICS queue，直接選 QF0。」

就這一行。從 crash 到正常運行，就這一行。

| llama.cpp 版本 | 支援此變數 | Vulkan 可用 |
|----------------|-----------|------------|
| b9519 (最新) | ✅ | ✅ |
| b9022 | ✅ | ✅ |
| b8626 | ✅ | ✅ |
| b6421 及更舊 | ❌ | ❌ |

## 效能實測

### 小模型：Qwen3-1.7B Q6_K

| 模式 | Prompt | Generation | 加速比 |
|------|--------|------------|--------|
| CPU only | 78 t/s | 25 t/s | 1.0x |
| Vulkan GPU | 25 t/s | **66 t/s** | **2.6x** |

### MoE 大模型：Qwen3.6-35B-A3B Q4_K_M（20 GB）

這是重頭戲。20GB 的 MoE 模型，8GB VRAM 怎麼跑？答案：**密集層放 GPU，專家層放 CPU**。

| 配置 | Prompt | Generation | VRAM | 加速比 |
|------|--------|------------|------|--------|
| CPU only | 3.8 t/s | 3.1 t/s | 0 | 1.0x |
| Vulkan `--cpu-moe` | 12.0 t/s | 10.6 t/s | ~1.8 GB | 3.4x |
| **Vulkan `--n-cpu-moe 30`（最佳）** | **22.6 t/s** | **16.3 t/s** | **~6.7 GB** | **5.3x** |
| GTX 1070 CUDA（參考） | — | 22.3 t/s | — | 7.2x |

RX 580 Vulkan 達到 GTX 1070 CUDA 的 **73%**。作為參考，這張 GTX 1070 當初的價格是 RX 580 的兩倍以上。

### VRAM 實際用量

實測結果：RX 580 可以吃到 **6.6 GB / 7.4 GB（89%）**，不是網路上流傳的「只能用 4 GB」。

32K context + q8_0 KV cache 可穩定運行。

### 關鍵參數

| 參數 | 值 | 原因 |
|------|-----|------|
| `GGML_VK_ALLOW_GRAPHICS_QUEUE` | `1` | **必設**，否則 segfault |
| `-ngl` | `99` | 能 offload 的全部丟 GPU |
| `--n-cpu-moe` | `30` | 10 層專家放 GPU，VRAM 極限 |
| `-t` | `4` | 減少 CPU contention |
| `-fa` | `on` | Flash attention 省 VRAM |
| `--cache-type-k` | `q8_0` | KV cache 減半 |
| `--cache-type-v` | `q8_0` | KV cache 減半 |
| `-fit` | `off` | 避免自動調整 |
| `--no-mmap` | — | 避免 mmap overhead |
| `-c` | `32768` | 最大 context（需配 q8_0 cache） |

## 實測可跑模型

| 模型 | 量化 | 大小 | 可用 | Generation |
|------|------|------|------|------------|
| Qwen3-1.7B | Q6_K | 1.6 GB | ✅ | 66 t/s |
| Qwen3-4B | Q4_K_M | 2.3 GB | ✅ | ~60 t/s |
| **Qwen3.6-35B-A3B** | **Q4_K_M** | **20 GB** | ✅ | **16.3 t/s** |
| Qwen3-30B-A3B | Q4_K_M | 18 GB | ✅ | ~15 t/s |
| Qwen3-32B (Dense) | Q4_K_M | 19 GB | ⚠️ | 需測試 |

## 快速開始

### 1. 下載 llama.cpp Vulkan build

```bash
cd ~
curl -L -o llama-vulkan.zip \
  https://github.com/ggml-org/llama.cpp/releases/download/b9519/llama-b9519-bin-win-vulkan-x64.zip
unzip -o llama-vulkan.zip -d ~/llama-vulkan
```

### 2. 確認 GPU 偵測

```bash
./llama-cli.exe --list-devices
# 應顯示：Vulkan0: AMD Radeon RX 580 2048SP
```

### 3. 跑小模型

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-cli.exe \
  -m Qwen3-1.7B-Q6_K.gguf \
  -ngl 99 -t 10 -c 4096 \
  --device Vulkan0
```

### 4. 跑 MoE 大模型（35B A3B）

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-cli.exe \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -t 4 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fit off --no-mmap --n-cpu-moe 30 \
  --device Vulkan0
```

### 5. Server 模式

```bash
GGML_VK_ALLOW_GRAPHICS_QUEUE=1 ./llama-server.exe \
  -m Qwen3.6-35B-A3B-Q4_K_M.gguf \
  -ngl 99 -t 4 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -fit off --no-mmap --n-cpu-moe 30 \
  --device Vulkan0 --port 8080
```

## 已知限制

1. **必須設 `GGML_VK_ALLOW_GRAPHICS_QUEUE=1`**，不設就 crash
2. MoE 大模型需 `--n-cpu-moe 30`（VRAM 極限），全放 GPU 會 OOM
3. 32K context + q8_0 KV cache 是 VRAM 上限（6.6 GB / 7.4 GB）
4. Polaris 無 fp16/int dot product/matrix cores 硬體支援
5. llama.cpp b6421 及更舊版本不支援此變數

## 測試環境

| 項目 | 數值 |
|------|------|
| OS | Windows 10 22H2 |
| CPU | Intel Core i7-6900K @ 3.20GHz (8C/16T) |
| RAM | 32 GB DDR4 |
| GPU | AMD Radeon RX 580 2048SP (Polaris, 8GB) |
| Driver | AMD Adrenalin 26.5.2 |
| Driver Vulkan API | 1.3.260 |
| llama.cpp | b9519 (2026-06-05) |

<p align="right"><a href="#-english">🔽 跳到英文版 / Jump to English</a></p>

</details>

---

# 🇺🇸 English

<details>
<summary><b>📖 Click to expand English version</b></summary>

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

<p align="right"><a href="#-chinese">🔼 跳到中文版 / Jump to Chinese</a></p>

</details>

---

## 相關專案 / Related Projects

- [castlen3/gtx1070-qwen3.6-35b-guide](https://github.com/castlen3/gtx1070-qwen3.6-35b-guide) — GTX 1070 同模型優化指南
- [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) — llama.cpp 本體
