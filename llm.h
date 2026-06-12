// llama2.c-style inference for Maykeye/TinyLLama-v0 (4.6M params, 32K vocab),
// ported to the M5Stack Cardputer ADV (ESP32-S3FN8, 512KB SRAM, 8MB flash, no PSRAM).
//
// Differences from the previous fp32 stories260K port:
//   - Q4_0 quantized weights (~2.5 MB total) instead of fp32 (~18 MB) — fits!
//   - Weights live in the application .bin via .incbin and are read directly
//     from MMU-mapped flash. No SD card, no partition install step.
//   - Tokenizer is walked from flash; no 256 KB index in heap.
//   - Sampler: argmax + multinomial only (no top-p / probindex).
#pragma once
#include <Arduino.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
  int dim;
  int hidden_dim;
  int n_layers;
  int n_heads;
  int n_kv_heads;
  int vocab_size;
  int seq_len;     // max from training; runtime KV is smaller
  int quant_type;  // 4 = Q4_0
  int shared_classifier;
} Config;

typedef struct {
  float* x;       // [dim]
  float* xb;      // [dim]
  float* xb2;     // [dim]
  float* hb;      // [hidden_dim]
  float* hb2;     // [hidden_dim]
  float* q;       // [dim]
  float* k;       // [kv_dim] fp32 scratch for the current position
  float* v;       // [kv_dim] fp32 scratch for the current position
  float* att;     // [n_heads, kv_seq_len]
  float* logits;  // [vocab_size]
  // KV cache is stored as bf16 (2 KB/position instead of 4 KB), which is what
  // lets KV_SEQ_LEN=64 fit in the same 128 KB an fp32 cache needed for 32.
  uint16_t* key_cache;   // bf16 [n_layers, kv_seq_len, kv_dim]
  uint16_t* value_cache; // bf16 [n_layers, kv_seq_len, kv_dim]
} RunState;

typedef struct {
  Config config;
  RunState state;
  int kv_seq_len;
  const uint8_t* base;   // start of embedded model bytes

  // Pointers into `base` (set after parsing the header).
  const float* rms_att;          // fp32 [n_layers, dim]
  const float* rms_ffn;          // fp32 [n_layers, dim]
  const float* rms_final;        // fp32 [dim]
  const uint8_t* token_embed_q4; // Q4_0 [vocab, dim]
  const uint8_t* wq_q4, *wk_q4, *wv_q4, *wo_q4;
  const uint8_t* w1_q4, *w2_q4, *w3_q4;
  const uint8_t* wcls_q4;

  // Per-layer stride in bytes for the Q4 tensors above.
  size_t stride_wq, stride_wk, stride_wv, stride_wo;
  size_t stride_w1w3, stride_w2;
} Transformer;

// Walking tokenizer — no in-RAM vocab table. The tokenizer.bin lives in flash
// and we linear-scan it for each lookup. Trades CPU for ~128 KB of heap.
typedef struct {
  const uint8_t* base;        // start of tokenizer blob (in flash)
  size_t size;
  int vocab_size;
  int max_token_length;
  char byte_pieces[512];      // 256 single-byte fallback pieces (data only)
} Tokenizer;

typedef struct {
  int vocab_size;
  float temperature;
  uint64_t rng_state;
} Sampler;

// Initialize a transformer that lives in flash. `model_bytes` must point at
// the converter's model_q4.bin contents. `kv_seq_len` is the active context
// window; the runstate is sized for it.
bool   llm_init_embedded(Transformer* t, const uint8_t* model_bytes, size_t model_size,
                         int kv_seq_len);

// Parse a tokenizer blob (also in flash). Allocates almost nothing.
bool   llm_tokenizer_from_memory(Tokenizer* tk, const uint8_t* data, size_t size,
                                 int vocab_size);

void   llm_build_sampler(Sampler* s, int vocab_size, float temperature, uint64_t seed);

float* llm_forward(Transformer* t, int token, int pos);
int    llm_sample(Sampler* s, float* logits);

void   llm_encode(Tokenizer* tk, const char* text, int8_t bos, int8_t eos,
                  int* tokens, int* n_tokens);
const char* llm_decode(Tokenizer* tk, int prev, int token, char* scratch, size_t scratch_size);
