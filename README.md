# cardputer_ai — TinyLLama-v0 on the M5Stack Cardputer ADV

A llama2.c-style inference engine running [Maykeye/TinyLLama-v0][hf] (4.6M params,
8 layers, 32K-vocab SentencePiece tokenizer) on the **Cardputer ADV**
(ESP32-S3FN8, 512 KB SRAM, 8 MB flash, no PSRAM).

Weights are quantized to **Q4_0** and embedded into the firmware binary, so
flashing through [bmorcelli/Launcher][lc] installs everything in one step —
no SD card, no model partition flashing, no setup.

Matmul rows are split across both ESP32-S3 cores. Expect roughly **2–5 tok/s**
during generation.

[hf]: https://huggingface.co/Maykeye/TinyLLama-v0
[lc]: https://github.com/bmorcelli/Launcher

## What's in the box

```
cardputer_ai/
├── cardputer_ai.ino           boot + chat loop
├── llm.{h,cpp}                Q4_0 inference engine, dual-core matmul
├── ui.{h,cpp}                 chat UI
├── partitions.csv             6 MB factory app slot for app + embedded model
├── model_data.cpp             generated — Q4 model bytes (~2.5 MB)
├── tok_data.cpp               generated — SentencePiece tokenizer (~500 KB)
└── tools/
    └── convert_tinyllama_v0.py   one-time HF → Q4_0 converter
```

## One-time setup

You need Python + pip to convert the model once. From the sketch directory:

```sh
python -m pip install huggingface_hub safetensors sentencepiece numpy
python tools/convert_tinyllama_v0.py
```

The converter downloads `Maykeye/TinyLLama-v0` from Hugging Face, quantizes
the weights to Q4_0, and writes **`model_data.cpp`** and **`tok_data.cpp`**
next to `cardputer_ai.ino`. Arduino IDE picks both files up automatically
as part of the sketch.

`model_data.cpp` ends up around 7–8 MB of source text (a long `uint8_t`
literal). The compiler is fine with it but the first build is slow (~30s for
parsing+codegen of the byte array).

## Build & flash

Arduino IDE 2.x with the M5Stack ESP32 board package:

- Board: **M5Stack-CardputerADV**
- PSRAM: **Disabled**
- Flash Size: **8 MB (64 Mb)**
- Partition Scheme: **Custom** — Arduino picks up `partitions.csv` from the
  sketch folder automatically.
- (Recommended) Library Manager → install `esp-dsp` for the SIMD dot product.

Then either:

- **Flash via USB** straight from Arduino IDE — fastest iteration loop.
- **Flash via bmorcelli/Launcher**: export the compiled `.bin`
  (Sketch → Export Compiled Binary), then in Launcher use
  *Install from SD card* or *Install from web* with that file. Launcher
  honors our `partitions.csv`, so the 6 MB factory slot is created on first
  install.

## How it works

```
boot ──> map embedded model from flash ──> chat
                                            │
                                            ▼
                                       per-token:
                                       ─ dequant 1 row of token_embedding
                                       ─ 8 layers (Q4 matmul on both cores)
                                       ─ Q4 classifier → 32K logits
                                       ─ argmax / multinomial sample
```

Weights are Q4_0 quantized in blocks of 32 (2-byte BF16 scale + 16 bytes of
packed nibbles → 18 bytes per block, 7× smaller than fp32). The forward pass
loads the active row of each weight tensor straight from the MMU-mapped
flash region — there's no scratch buffer and no SD I/O on the hot path.

`llm.cpp::dot_q4_f32` is the inner loop. It decodes 32 nibbles per block and
multiplies them by 32 fp32 activations. The matmul caller splits rows between
core 0 (a long-lived worker task) and core 1 (the Arduino loop task), with a
pair of binary semaphores for the join.

## Memory budget (Cardputer ADV, ~280 KB free heap)

| Buffer                            | Bytes    |
|-----------------------------------|----------|
| KV cache, ctx=32, fp32            | 128 KB   |
| Logits (vocab=32000)              | 128 KB   |
| Activations + attention scores    |  ~10 KB  |
| FreeRTOS + matmul worker stack    |   4 KB   |

Top-p sampling is intentionally **dropped** — its `probindex` would cost
another 256 KB, which we don't have. Argmax (T=0) and full multinomial
sampling work fine.

The KV-cache window is `KV_SEQ_LEN = 32` in `cardputer_ai.ino`. Raising it
linearly grows the KV cache — past ~48 the heap runs out.

## Regenerating with a different model

The converter currently hard-codes the Maykeye architecture
(dim=64, n_layers=8, n_heads=16, hidden_dim=256, vocab=32000). Any other
LLaMA-architecture model with the same shape will work just by pointing
`--model-dir` at a local HF snapshot:

```sh
python tools/convert_tinyllama_v0.py --model-dir /path/to/local/hf/snapshot
```

Larger architectures need the constants in `convert_tinyllama_v0.py` updated
and (more importantly) need to physically fit — anything past ~3.5 MB Q4 will
overflow the factory partition.

## What's not great yet

- 32-token context. The KV cache + 128 KB logits buffer are the binding
  constraints. With PSRAM this would lift to 2048.
- Encoding a prompt walks the 32K-token tokenizer once per BPE merge pass,
  which is a few hundred ms for normal-length prompts. Fine for chat;
  noticeable if you paste a paragraph.
- TinyLLama-v0 is trained on TinyStories — it's a *completion* model. Good
  prompts: `Once upon a time there was`. Bad prompts: `How do I cook rice?`.
- Generation tops out around 2–5 tok/s. SIMD-accelerated Q4 matmul on
  ESP32-S3 PIE would roughly double it but isn't implemented yet.
