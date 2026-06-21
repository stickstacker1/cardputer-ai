#pragma once

// AdvanceOS Integration for Cardputer AI
// This app runs the LLM engine within AdvanceOS's lifecycle system
// Inherits from GlobalParentClass (AdvanceOS app pattern)

#include <Arduino.h>
#include <M5Cardputer.h>
#include <string>
#include <vector>

// Forward declaration - replace with actual AdvanceOS path
// #include "../GlobalParentClass.h"  // From AdvanceOS

class CardputerAI {
public:
    CardputerAI();
    ~CardputerAI();
    
    // AdvanceOS lifecycle methods
    void Begin();      // Initialize LLM, allocate memory
    void Loop();       // Process input, generation step
    void Draw();       // Render to sprite
    void OnExit();     // Cleanup, free model

    // Internal state
private:
    struct LLMState {
        bool initialized = false;
        bool modelLoaded = false;
        int contextWindow = 64;  // Optimized for AdvanceOS (80 for standalone)
        float temperature = 0.8f;
    } llm_state;

    struct UIState {
        bool isGenerating = false;
        std::string inputBuffer;
        std::string chatHistory[4];  // Store last 4 exchanges
        int historyCount = 0;
    } ui_state;

    // LLM engine functions (adapted from cardputer-ai)
    void initializeModel();
    void cleanupModel();
    void stepGeneration();
    void processKeyboard();
    void renderUI();

    // Memory optimization
    void optimizeMemory();
    int getHeapUsage();

public:
    // Settings (accessible from AdvanceOS settings)
    void setTemperature(float temp) { llm_state.temperature = temp; }
    void setContextWindow(int ctxLen) { llm_state.contextWindow = ctxLen; }
    float getTemperature() const { return llm_state.temperature; }
    int getContextWindow() const { return llm_state.contextWindow; }
};
