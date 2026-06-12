// Inference engine for Maykeye/TinyLLama-v0 on Cardputer ADV.
// Weights are Q4_0-quantized and embedded into the firmware .bin via .incbin —
// the MMU maps the app partition into the data space, so reads are zero-copy
// from flash. Matmul rows are split between core 0 and core 1.

#include "llm.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <esp_heap_caps.h>
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/semphr.h>

// ============================================================================
//  Constants — file format
// ============================================================================
static const uint8_t  MAGIC[4]      = { 'C', 'R', 'D', 'P' };
static const uint32_t VERSION       = 1;
static const int      BLOCK_SIZE    = 32;   // weights per Q4_0 block
static const size_t   BYTES_PER_BLK = 18;   // 2B bf16 scale + 16B nibbles

static void* ram_alloc(size_t n) {
  return heap_caps_malloc(n, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
}

// ============================================================================
//  BF16 <-> FP32
// ============================================================================
static inline float bf16_to_fp32(uint16_t b) {
  union { uint32_t u; float f; } v = { (uint32_t)b << 16 };
  return v.f;
}

static inline uint16_t fp32_to_bf16(float f) {
  union { float f; uint32_t u; } v = { f };
  // round-to-nearest-even, same as the converter script
  return (uint16_t)((v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16);
}

// ============================================================================
//  Q4_0 dot product: dot(q4_row, fp32_vec).
//  n MUST be a multiple of 32. Each block is 18 bytes:
//    - 2-byte BF16 scale
//    - 16 packed-nibble bytes: byte k has q[k] in low nibble, q[k+16] in high.
//  Dequantized value = scale * (qn - 8).
// ============================================================================
static inline float dot_q4_f32(const uint8_t* w, const float* x, int n) {
  int blocks = n >> 5;
  float acc = 0.0f;
  for (int b = 0; b < blocks; b++) {
    uint16_t sb; memcpy(&sb, w, 2);
    float scale = bf16_to_fp32(sb);
    const uint8_t* nib = w + 2;
    // Two interleaved accumulators help the compiler schedule muladds.
    float a0 = 0.0f, a1 = 0.0f;
    const float* x0 = x;
    const float* x1 = x + 16;
    for (int k = 0; k < 16; k++) {
      uint8_t  bk = nib[k];
      int      lo = (int)(bk & 0x0F) - 8;
      int      hi = (int)(bk >> 4)   - 8;
      a0 += (float)lo * x0[k];
      a1 += (float)hi * x1[k];
    }
    acc += scale * (a0 + a1);
    w += BYTES_PER_BLK;
    x += BLOCK_SIZE;
  }
  return acc;
}

static inline void dequant_q4_row(float* out, const uint8_t* w, int n) {
  int blocks = n >> 5;
  for (int b = 0; b < blocks; b++) {
    uint16_t sb; memcpy(&sb, w, 2);
    float scale = bf16_to_fp32(sb);
    const uint8_t* nib = w + 2;
    for (int k = 0; k < 16; k++) {
      uint8_t bk = nib[k];
      int lo = (int)(bk & 0x0F) - 8;
      int hi = (int)(bk >> 4)   - 8;
      out[k]      = scale * (float)lo;
      out[k + 16] = scale * (float)hi;
    }
    out += BLOCK_SIZE;
    w   += BYTES_PER_BLK;
  }
}

// ============================================================================
//  Dual-core Q4 matmul. Same persistent-worker design as before: one task is
//  pinned to the opposite core for the life of the program; per-call cost is
//  two semaphore operations.
// ============================================================================
struct MatmulArgs {
  float* xout;
  const float* x;
  const uint8_t* w;
  int n;           // inner dim (multiple of 32)
  size_t row_bytes;
  int row_start;
  int row_end;
};

static void matmul_rows(const MatmulArgs* a) {
  const int n = a->n;
  const size_t rb = a->row_bytes;
  for (int row = a->row_start; row < a->row_end; row++) {
    a->xout[row] = dot_q4_f32(a->w + row * rb, a->x, n);
  }
}

static SemaphoreHandle_t mm_go = NULL, mm_done = NULL;
static MatmulArgs        mm_args;
static TaskHandle_t      mm_task_handle = NULL;

static void mm_worker(void*) {
  for (;;) {
    xSemaphoreTake(mm_go, portMAX_DELAY);
    matmul_rows(&mm_args);
    xSemaphoreGive(mm_done);
  }
}

static void ensure_worker() {
  if (mm_task_handle) return;
  mm_go   = xSemaphoreCreateBinary();
  mm_done = xSemaphoreCreateBinary();
  BaseType_t main_core  = xPortGetCoreID();
  BaseType_t other_core = (main_core == 0) ? 1 : 0;
  xTaskCreatePinnedToCore(mm_worker, "mm_w", 3072, NULL, 5,
                          &mm_task_handle, other_core);
}

static void matmul_q4(float* xout, const float* x, const uint8_t* w, int n, int d) {
  ensure_worker();
  size_t row_bytes = ((size_t)(n / BLOCK_SIZE)) * BYTES_PER_BLK;
  int split = d / 2;

  mm_args.xout = xout; mm_args.x = x; mm_args.w = w;
  mm_args.n = n; mm_args.row_bytes = row_bytes;
  mm_args.row_start = 0; mm_args.row_end = split;
  xSemaphoreGive(mm_go);

  MatmulArgs self = { xout, x, w, n, row_bytes, split, d };
  matmul_rows(&self);

  xSemaphoreTake(mm_done, portMAX_DELAY);
  yield();  // feed both cores' IDLE so the WDT stays happy
}

// ============================================================================
//  Norms + softmax
// ============================================================================
static void rmsnorm(float* o, const float* x, const float* w, int size) {
  float ss = 0.0f;
  for (int i = 0; i < size; i++) ss += x[i] * x[i];
  ss /= size; ss += 1e-6f; ss = 1.0f / sqrtf(ss);
  for (int i = 0; i < size; i++) o[i] = w[i] * (ss * x[i]);
}

static void softmax(float* x, int size) {
  float mx = x[0];
  for (int i = 1; i < size; i++) if (x[i] > mx) mx = x[i];
  float sum = 0.0f;
  for (int i = 0; i < size; i++) { x[i] = expf(x[i] - mx); sum += x[i]; }
  for (int i = 0; i < size; i++) x[i] /= sum;
}

// ============================================================================
//  Forward pass
// ============================================================================
float* llm_forward(Transformer* T, int token, int pos) {
  Config* p = &T->config;
  RunState* s = &T->state;
  int dim        = p->dim;
  int kv_dim     = (p->dim * p->n_kv_heads) / p->n_heads;
  int kv_mul     = p->n_heads / p->n_kv_heads;
  int hidden_dim = p->hidden_dim;
  int head_size  = dim / p->n_heads;
  int kvL        = T->kv_seq_len;
  if (pos >= kvL) pos = kvL - 1;

  // ---- token embedding (dequant one row of the embedding matrix into x) ----
  {
    size_t row_bytes = ((size_t)(dim / BLOCK_SIZE)) * BYTES_PER_BLK;
    const uint8_t* row = T->token_embed_q4 + (size_t)token * row_bytes;
    dequant_q4_row(s->x, row, dim);
  }

  for (int l = 0; l < p->n_layers; l++) {
    rmsnorm(s->xb, s->x, T->rms_att + (size_t)l * dim, dim);

    matmul_q4(s->q, s->xb, T->wq_q4 + (size_t)l * T->stride_wq, dim, dim);
    matmul_q4(s->k, s->xb, T->wk_q4 + (size_t)l * T->stride_wk, dim, kv_dim);
    matmul_q4(s->v, s->xb, T->wv_q4 + (size_t)l * T->stride_wv, dim, kv_dim);

    // RoPE
    for (int i = 0; i < dim; i += 2) {
      int head_dim = i % head_size;
      float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
      float val  = pos * freq;
      float fcr  = cosf(val), fci = sinf(val);
      int rotn = (i < kv_dim) ? 2 : 1;
      for (int v_ = 0; v_ < rotn; v_++) {
        float* vec = (v_ == 0) ? s->q : s->k;
        float v0 = vec[i], v1 = vec[i+1];
        vec[i]   = v0 * fcr - v1 * fci;
        vec[i+1] = v0 * fci + v1 * fcr;
      }
    }

    // Store this position's k/v into the bf16 cache (after RoPE).
    {
      uint16_t* krow = s->key_cache   + (size_t)l * kvL * kv_dim + (size_t)pos * kv_dim;
      uint16_t* vrow = s->value_cache + (size_t)l * kvL * kv_dim + (size_t)pos * kv_dim;
      for (int i = 0; i < kv_dim; i++) {
        krow[i] = fp32_to_bf16(s->k[i]);
        vrow[i] = fp32_to_bf16(s->v[i]);
      }
    }

    // Attention
    for (int h = 0; h < p->n_heads; h++) {
      float* q   = s->q   + h * head_size;
      float* att = s->att + h * kvL;
      for (int t = 0; t <= pos; t++) {
        const uint16_t* k = s->key_cache + (size_t)l * kvL * kv_dim
                                         + (size_t)t * kv_dim
                                         + (h / kv_mul) * head_size;
        float score = 0;
        for (int i = 0; i < head_size; i++) score += q[i] * bf16_to_fp32(k[i]);
        att[t] = score / sqrtf((float)head_size);
      }
      softmax(att, pos + 1);
      float* xb = s->xb + h * head_size;
      memset(xb, 0, head_size * sizeof(float));
      for (int t = 0; t <= pos; t++) {
        const uint16_t* v = s->value_cache + (size_t)l * kvL * kv_dim
                                           + (size_t)t * kv_dim
                                           + (h / kv_mul) * head_size;
        float a = att[t];
        for (int i = 0; i < head_size; i++) xb[i] += a * bf16_to_fp32(v[i]);
      }
    }

    matmul_q4(s->xb2, s->xb, T->wo_q4 + (size_t)l * T->stride_wo, dim, dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];

    rmsnorm(s->xb, s->x, T->rms_ffn + (size_t)l * dim, dim);

    matmul_q4(s->hb,  s->xb, T->w1_q4 + (size_t)l * T->stride_w1w3, dim, hidden_dim);
    matmul_q4(s->hb2, s->xb, T->w3_q4 + (size_t)l * T->stride_w1w3, dim, hidden_dim);

    // SwiGLU: hb = silu(hb) * hb2
    for (int i = 0; i < hidden_dim; i++) {
      float v = s->hb[i];
      v *= 1.0f / (1.0f + expf(-v));
      v *= s->hb2[i];
      s->hb[i] = v;
    }

    matmul_q4(s->xb, s->hb, T->w2_q4 + (size_t)l * T->stride_w2, hidden_dim, dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
  }

  rmsnorm(s->x, s->x, T->rms_final, dim);

  // ---- classifier: one big Q4 matmul producing 32K logits ----
  matmul_q4(s->logits, s->x, T->wcls_q4, dim, p->vocab_size);

  yield();
  return s->logits;
}

// ============================================================================
//  Header parse + offset compute
// ============================================================================
bool llm_init_embedded(Transformer* T, const uint8_t* model_bytes, size_t model_size,
                       int kv_seq_len) {
  if (model_size < 64) return false;
  if (memcmp(model_bytes, MAGIC, 4) != 0) return false;

  uint32_t version;
  memcpy(&version, model_bytes + 4, 4);
  if (version != VERSION) return false;

  int hdr[7];
  memcpy(hdr, model_bytes + 8, sizeof(hdr));
  T->config.dim         = hdr[0];
  T->config.hidden_dim  = hdr[1];
  T->config.n_layers    = hdr[2];
  T->config.n_heads     = hdr[3];
  T->config.n_kv_heads  = hdr[4];
  T->config.vocab_size  = hdr[5];
  T->config.seq_len     = hdr[6];
  uint8_t shared = model_bytes[36];
  uint8_t quant  = model_bytes[37];
  T->config.shared_classifier = shared;
  T->config.quant_type        = quant;
  if (quant != 4) return false;

  Config* p = &T->config;
  int head_size = p->dim / p->n_heads;
  int kv_dim    = head_size * p->n_kv_heads;

  T->kv_seq_len = (kv_seq_len > 0 && kv_seq_len < p->seq_len) ? kv_seq_len : p->seq_len;
  T->base = model_bytes;

  // Layout after the 64-byte header:
  //   fp32 rms_att [L*dim]
  //   fp32 rms_ffn [L*dim]
  //   fp32 rms_final [dim]
  //   Q4_0 token_embed [V*dim]
  //   Q4_0 wq, wk, wv, wo  (per-layer)
  //   Q4_0 w1, w2, w3      (per-layer)
  //   Q4_0 wcls            (or aliased to token_embed if shared)
  size_t off = 64;
  T->rms_att   = (const float*)(model_bytes + off); off += (size_t)p->n_layers * p->dim * sizeof(float);
  T->rms_ffn   = (const float*)(model_bytes + off); off += (size_t)p->n_layers * p->dim * sizeof(float);
  T->rms_final = (const float*)(model_bytes + off); off += (size_t)p->dim * sizeof(float);

  auto q4_bytes = [](size_t n_weights) -> size_t {
    return (n_weights / BLOCK_SIZE) * BYTES_PER_BLK;
  };

  T->token_embed_q4 = model_bytes + off;
  off += q4_bytes((size_t)p->vocab_size * p->dim);

  T->stride_wq    = q4_bytes((size_t)(p->n_heads    * head_size) * p->dim);
  T->stride_wk    = q4_bytes((size_t)kv_dim * p->dim);
  T->stride_wv    = q4_bytes((size_t)kv_dim * p->dim);
  T->stride_wo    = q4_bytes((size_t)p->dim * (p->n_heads * head_size));
  T->stride_w1w3  = q4_bytes((size_t)p->hidden_dim * p->dim);
  T->stride_w2    = q4_bytes((size_t)p->dim * p->hidden_dim);

  T->wq_q4 = model_bytes + off; off += T->stride_wq * p->n_layers;
  T->wk_q4 = model_bytes + off; off += T->stride_wk * p->n_layers;
  T->wv_q4 = model_bytes + off; off += T->stride_wv * p->n_layers;
  T->wo_q4 = model_bytes + off; off += T->stride_wo * p->n_layers;
  T->w1_q4 = model_bytes + off; off += T->stride_w1w3 * p->n_layers;
  T->w2_q4 = model_bytes + off; off += T->stride_w2 * p->n_layers;
  T->w3_q4 = model_bytes + off; off += T->stride_w1w3 * p->n_layers;
  if (shared) {
    T->wcls_q4 = T->token_embed_q4;
  } else {
    T->wcls_q4 = model_bytes + off;
    off += q4_bytes((size_t)p->vocab_size * p->dim);
  }
  if (off > model_size) return false;

  // RunState
  RunState* s = &T->state;
  s->x      = (float*) ram_alloc(p->dim * sizeof(float));
  s->xb     = (float*) ram_alloc(p->dim * sizeof(float));
  s->xb2    = (float*) ram_alloc(p->dim * sizeof(float));
  s->hb     = (float*) ram_alloc(p->hidden_dim * sizeof(float));
  s->hb2    = (float*) ram_alloc(p->hidden_dim * sizeof(float));
  s->q      = (float*) ram_alloc(p->dim * sizeof(float));
  s->k      = (float*) ram_alloc(kv_dim * sizeof(float));
  s->v      = (float*) ram_alloc(kv_dim * sizeof(float));
  s->att    = (float*) ram_alloc(p->n_heads * T->kv_seq_len * sizeof(float));
  s->logits = (float*) ram_alloc(p->vocab_size * sizeof(float));
  s->key_cache   = (uint16_t*) ram_alloc((size_t)p->n_layers * T->kv_seq_len * kv_dim * sizeof(uint16_t));
  s->value_cache = (uint16_t*) ram_alloc((size_t)p->n_layers * T->kv_seq_len * kv_dim * sizeof(uint16_t));

  return s->x && s->xb && s->xb2 && s->hb && s->hb2 && s->q && s->k && s->v
      && s->att && s->logits && s->key_cache && s->value_cache;
}

// ============================================================================
//  Walking tokenizer — no per-token index in heap. The blob lives in flash
//  and we linear-scan it for lookups. Each scan touches ~256 KB of mmap'd
//  flash and runs in a few ms.
//
//  Format (matches llama2.c export):
//    [u32 max_token_length]
//    repeated vocab_size times:
//      [f32 score]
//      [i32 length]
//      [bytes string]   (no NUL terminator)
// ============================================================================
bool llm_tokenizer_from_memory(Tokenizer* tk, const uint8_t* data, size_t size,
                               int vocab_size) {
  if (!data || size < 4) return false;
  tk->base       = data;
  tk->size       = size;
  tk->vocab_size = vocab_size;
  uint32_t mtl; memcpy(&mtl, data, 4);
  tk->max_token_length = (int)mtl;

  for (int i = 0; i < 256; i++) {
    tk->byte_pieces[i*2]   = (unsigned char)i;
    tk->byte_pieces[i*2+1] = 0;
  }
  return true;
}

// Walk the tokenizer blob. `cb(id, score, str, len)` returns true to stop early.
typedef bool (*tok_visit_fn)(void* ctx, int id, float score, const char* str, int len);
static void tokenizer_walk(const Tokenizer* tk, tok_visit_fn cb, void* ctx) {
  const uint8_t* p = tk->base + 4;
  const uint8_t* end = tk->base + tk->size;
  for (int i = 0; i < tk->vocab_size; i++) {
    if (p + 8 > end) return;
    float s;     memcpy(&s, p, 4); p += 4;
    int32_t len; memcpy(&len, p, 4); p += 4;
    if (len < 0 || p + len > end) return;
    if (cb(ctx, i, s, (const char*)p, len)) return;
    p += len;
  }
}

// Lookup by string match. Returns id or -1.
struct lookup_ctx { const char* needle; int needle_len; int found; };
static bool lookup_cb(void* c, int id, float, const char* str, int len) {
  lookup_ctx* L = (lookup_ctx*)c;
  if (len == L->needle_len && memcmp(str, L->needle, len) == 0) {
    L->found = id; return true;
  }
  return false;
}
static int tok_lookup(const Tokenizer* tk, const char* s, int len) {
  lookup_ctx L = { s, len, -1 };
  tokenizer_walk(tk, lookup_cb, &L);
  return L.found;
}

// Lookup by id — walk to position. Returns true and fills out pointers if found.
struct byid_ctx { int target; const char* str; int len; float score; bool found; };
static bool byid_cb(void* c, int id, float score, const char* str, int len) {
  byid_ctx* B = (byid_ctx*)c;
  if (id == B->target) {
    B->str = str; B->len = len; B->score = score; B->found = true; return true;
  }
  return false;
}
static bool tok_byid(const Tokenizer* tk, int id, const char** str, int* len, float* score) {
  byid_ctx B = { id, NULL, 0, 0, false };
  tokenizer_walk(tk, byid_cb, &B);
  if (!B.found) return false;
  if (str)   *str   = B.str;
  if (len)   *len   = B.len;
  if (score) *score = B.score;
  return true;
}

// One-shot BPE merge scan: for each adjacent token pair, look up the merged
// string. Pre-stage the merged candidate strings into a flat buffer, then walk
// the tokenizer ONCE testing them all. This turns O(N * V) into O(V).
struct merge_scan_ctx {
  const char** cands;        // [N-1] candidate concat strings
  const int*   cand_lens;
  int          n_cands;
  float        best_score;
  int          best_id;
  int          best_idx;     // which adjacent-pair index won
};
static bool merge_scan_cb(void* c, int id, float score, const char* str, int len) {
  merge_scan_ctx* M = (merge_scan_ctx*)c;
  if (score <= M->best_score) return false;
  for (int i = 0; i < M->n_cands; i++) {
    if (M->cand_lens[i] == len && memcmp(M->cands[i], str, len) == 0) {
      M->best_score = score; M->best_id = id; M->best_idx = i;
      return false;
    }
  }
  return false;
}

void llm_encode(Tokenizer* tk, const char* text, int8_t bos, int8_t eos,
                int* tokens, int* n_tokens) {
  *n_tokens = 0;
  if (bos) tokens[(*n_tokens)++] = 1;

  // SentencePiece convention: prepend a dummy space so the first word starts
  // with the same leading-space marker as later words.
  if (text[0] != 0) {
    int id = tok_lookup(tk, " ", 1);
    if (id >= 0) tokens[(*n_tokens)++] = id;
  }

  // Per-char tokenization with UTF-8 grouping (lookup, fall back to <0xNN>).
  char ch[8];
  int  chlen = 0;
  for (const char* c = text; *c; c++) {
    if ((*c & 0xC0) != 0x80) chlen = 0;
    ch[chlen++] = *c;
    if ((*(c+1) & 0xC0) == 0x80 && chlen < 4) continue;
    int id = tok_lookup(tk, ch, chlen);
    if (id >= 0) tokens[(*n_tokens)++] = id;
    else for (int i = 0; i < chlen; i++) tokens[(*n_tokens)++] = (unsigned char)ch[i] + 3;
    chlen = 0;
  }

  // Iterative BPE merges. Each iteration walks the tokenizer once, testing
  // every adjacent pair against every vocab entry as we go.
  const int max_pairs = *n_tokens;
  char* cand_buf = (char*) malloc(max_pairs * (tk->max_token_length * 2 + 4));
  if (!cand_buf) return;
  const char** cand_ptrs = (const char**) malloc(max_pairs * sizeof(char*));
  int*         cand_lens = (int*)         malloc(max_pairs * sizeof(int));
  if (!cand_ptrs || !cand_lens) { free(cand_buf); free(cand_ptrs); free(cand_lens); return; }

  // We need per-token string slices for assembly. Cache them per iter.
  while (true) {
    int n = *n_tokens;
    if (n < 2) break;

    int      n_cands = 0;
    char*    cursor  = cand_buf;
    // Cache id→string for the current tokens so we don't re-walk per pair.
    struct Slice { const char* p; int len; };
    Slice* slices = (Slice*) malloc(n * sizeof(Slice));
    if (!slices) break;
    for (int i = 0; i < n; i++) {
      const char* sp; int sl;
      if (!tok_byid(tk, tokens[i], &sp, &sl, NULL)) { sl = 0; sp = ""; }
      slices[i].p = sp; slices[i].len = sl;
    }
    for (int i = 0; i < n - 1; i++) {
      int la = slices[i].len, lb = slices[i+1].len;
      memcpy(cursor, slices[i].p, la);
      memcpy(cursor + la, slices[i+1].p, lb);
      cand_ptrs[n_cands] = cursor;
      cand_lens[n_cands] = la + lb;
      cursor += la + lb;
      n_cands++;
    }
    free(slices);

    merge_scan_ctx M = { cand_ptrs, cand_lens, n_cands, -1e10f, -1, -1 };
    tokenizer_walk(tk, merge_scan_cb, &M);
    if (M.best_idx < 0) break;

    tokens[M.best_idx] = M.best_id;
    for (int i = M.best_idx + 1; i < n - 1; i++) tokens[i] = tokens[i+1];
    (*n_tokens)--;
  }
  free(cand_buf); free(cand_ptrs); free(cand_lens);

  if (eos) tokens[(*n_tokens)++] = 2;
}

const char* llm_decode(Tokenizer* tk, int prev, int token, char* scratch, size_t scratch_size) {
  const char* str = NULL; int len = 0;
  if (!tok_byid(tk, token, &str, &len, NULL)) return "";

  // After BOS, strip a single leading space (matches llama2.c behaviour).
  if (prev == 1 && len > 0 && str[0] == ' ') { str++; len--; }

  // <0xNN> byte-fallback piece: return the byte itself.
  unsigned b;
  if (len > 0 && str[0] == '<' && len <= 6 && str[len-1] == '>') {
    char tmp[8]; if (len < 8) { memcpy(tmp, str, len); tmp[len] = 0;
      if (sscanf(tmp, "<0x%02X>", &b) == 1) return &tk->byte_pieces[b*2];
    }
  }
  // Copy into the caller-provided scratch and NUL-terminate.
  if (scratch_size == 0) return "";
  size_t cp = (size_t)len < scratch_size - 1 ? (size_t)len : scratch_size - 1;
  memcpy(scratch, str, cp);
  scratch[cp] = 0;
  return scratch;
}

// ============================================================================
//  Sampler — argmax (T=0) or full-vocab multinomial. No probindex (no top-p).
// ============================================================================
void llm_build_sampler(Sampler* s, int vocab_size, float temperature, uint64_t seed) {
  s->vocab_size  = vocab_size;
  s->temperature = temperature;
  s->rng_state   = seed ? seed : 0xC0FFEEull;
}
static uint32_t rng_u32(uint64_t* st) {
  *st ^= *st >> 12; *st ^= *st << 25; *st ^= *st >> 27;
  return (uint32_t)((*st * 0x2545F4914F6CDD1Dull) >> 32);
}
static float rng_f32(uint64_t* st) { return (rng_u32(st) >> 8) / 16777216.0f; }

int llm_sample(Sampler* s, float* logits) {
  if (s->temperature == 0.0f) {
    int   mi = 0;
    float m  = logits[0];
    for (int i = 1; i < s->vocab_size; i++) if (logits[i] > m) { m = logits[i]; mi = i; }
    return mi;
  }
  for (int i = 0; i < s->vocab_size; i++) logits[i] /= s->temperature;
  softmax(logits, s->vocab_size);
  float coin = rng_f32(&s->rng_state);
  float c = 0;
  for (int i = 0; i < s->vocab_size; i++) {
    c += logits[i];
    if (coin < c) return i;
  }
  return s->vocab_size - 1;
}
