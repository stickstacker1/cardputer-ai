# Cardputer AI — AdvanceOS Integration Guide

This guide explains how to integrate **Cardputer AI** as a native app within **AdvanceOS**.

## Architecture Overview

### Cardputer AI (Current)
- **Standalone firmware**: Replaces entire OS
- **Model**: Q4_0 quantized TinyStories-Instruct-3M (GPT-Neo)
- **Engine**: Embedded LLM (llm.cpp, llm.h)
- **UI**: Custom chat UI (ui.cpp, ui.h)
- **Size**: ~1.9 MB model + engine

### AdvanceOS
- **Modular app framework**: Each app inherits from `GlobalParentClass`
- **App pattern**: `Begin()`, `Loop()`, `Draw()`, `OnExit()` lifecycle
- **Display**: M5Canvas-based sprite rendering (240×135)
- **Keyboard**: M5Cardputer keyboard handling
- **Memory**: Shared heap with other apps

## Integration Approach: Option A (Recommended)

**Create an AdvanceOS app wrapper** that runs the LLM engine within AdvanceOS's lifecycle.

### Pros
- ✅ Full AdvanceOS integration (menu, themes, settings)
- ✅ Can switch between apps without rebooting
- ✅ Access to OS features (screenshots, themes)
- ✅ Minimal memory overhead when not active

### Cons
- ⚠️ Need to optimize memory (context window may shrink)
- ⚠️ Model loading takes time
- ⚠️ Both model and OS share heap

---

## File Structure

```
TheProject/src/Classes/
├── CardputerAI.h          (header)
├── CardputerAI.cpp        (app implementation)
└── llm_adapter/           (adapted LLM engine)
    ├── llm_minimal.h      (simplified LLM interface)
    ├── llm_minimal.cpp    (Q4_0 inference core)
    ├── tokenizer.h
    ├── tokenizer.cpp
    └── model_data.cpp     (generated: embedded model)
```

---

## Key Integration Points

### 1. App Class (CardputerAI.h)

```cpp
#pragma once
#include "../GlobalParentClass.h"
#include "llm_adapter/llm_minimal.h"

class CardputerAI : public GlobalParentClass {
public:
    CardputerAI(MyOS *os) : GlobalParentClass(os) {}
    
    void Begin() override;      // Initialize LLM, UI
    void Loop() override;       // Process keyboard, generation step
    void Draw() override;       // Render chat UI
    void OnExit() override;     // Cleanup, free model
    
private:
    LLMEngine engine;
    ChatUIAdapter ui;
    bool modelLoaded = false;
};
```

### 2. Memory Constraints

**Current Cardputer AI**: 280 KB free heap
- KV cache: 165 KB
- Logits: 48 KB
- Activations: 12 KB

**AdvanceOS context**: Shared heap with OS
- May need to reduce `KV_SEQ_LEN` from 80 → 64 tokens (~132 KB)
- Or reduce model size (use 1.5M instead of 3M)

### 3. Model Embedding

The model lives in flash (MMU-mapped as `.rodata`). To embed in AdvanceOS:

```bash
# In cardputer-ai/tools/:
python3 convert_tinystories_instruct.py --model 3M --keep-bin --no-cpp

# Generate embedding for AdvanceOS:
# Use a hex-dumper or xxd to convert the binary
xxd -i embed/model_neo_q4.bin > model_data.cpp
xxd -i embed/tok_neo.bin > tok_data.cpp
```

---

## Implementation Checklist

- [ ] Create `CardputerAI.h` header with app class
- [ ] Create `CardputerAI.cpp` with lifecycle methods
- [ ] Copy & adapt LLM engine files from `cardputer-ai/main/`:
  - [ ] `llm.h` → `llm_adapter/llm_minimal.h`
  - [ ] `llm.cpp` → `llm_adapter/llm_minimal.cpp`
  - [ ] `tokenizer code` → `llm_adapter/tokenizer.cpp`
- [ ] Generate `model_data.cpp` and `tok_data.cpp`
- [ ] Adapt UI rendering for AdvanceOS sprite system
- [ ] Register app in `MyOS.cpp` menu
- [ ] Test memory usage (free heap while running)
- [ ] Optimize context window if needed

---

## Integration with MyOS Menu

In `MyOS.cpp`, add to the app registration:

```cpp
// Inside MyOS::begin() or app registration function
MenuItem aiApp;
aiApp.name = "Cardputer AI";
aiApp.color = 0x07FF;  // cyan
aiApp.type = APP;
aiApp.onLaunch = [this]() {
    os.ChangeMenu(new CardputerAI(&os));
};
allApps.push_back(aiApp);
```

---

## Memory Optimization Tips

1. **Reduce context**: `KV_SEQ_LEN = 64` (132 KB) instead of 80 (165 KB)
2. **Use smaller model**: TinyStories-1.5M instead of 3M
3. **Load model on-demand**: Only when app launches (not at boot)
4. **Reduce history**: Store only last 2 exchanges instead of 4
5. **Lazy UI rendering**: Only update changed regions

---

## Testing Checklist

- [ ] App launches from menu without crashing
- [ ] Chat input works (keyboard input)
- [ ] Model inference runs (~7 tok/s on device)
- [ ] Generation completes or times out gracefully
- [ ] Memory doesn't corrupt other apps (relaunch another app after AI)
- [ ] Settings persist (temperature, mode)
- [ ] `/new` command resets conversation
- [ ] OnExit() properly frees model from heap

---

## References

- **Cardputer AI**: `/cardputer-ai/main/main.cpp`, `llm.h`, `ui.h`
- **AdvanceOS app pattern**: `/AdvanceOS-for-cardputer/TheProject/src/Classes/Notes.h`
- **GlobalParentClass**: `/AdvanceOS-for-cardputer/TheProject/src/GlobalParentClass.h`
- **M5Cardputer**: https://github.com/m5stack/M5Cardputer

---

## Next Steps

1. **Clone the LLM engine** from Cardputer AI into `llm_adapter/`
2. **Create CardputerAI app class** inheriting `GlobalParentClass`
3. **Generate embedded model** (`model_data.cpp`)
4. **Test memory** with reduced KV context
5. **Integrate UI** with AdvanceOS sprite system
6. **Register** in MyOS menu
7. **Iterate** based on memory constraints

---

## Questions or Issues?

- Model too large? Try the 1.5M version
- Memory crashes? Reduce `KV_SEQ_LEN` or context history
- UI not rendering? Check sprite coordinates (top offset + 18px for top bar)
