# Supported Models Guide

## 🤖 **Multi-Model Evaluation Framework**

The Graph RAG evaluation framework now supports comparing multiple language models on the same datasets and tasks. This enables comprehensive model evaluation and selection for your specific use case.

## 📋 **Available Models**

### **Instruction-Tuned Models**
| Model | Size | Specialization | Best For |
|-------|------|----------------|----------|
| **Phi-3 Mini** | 3.8B | General reasoning | 🥇 Mathematical problems, balanced performance |
| **Mistral 7B** | 7B | Multilingual | Code generation, multiple languages |
| **Llama-2 7B** | 7B | Conversational | General Q&A, dialogue tasks |
| **Llama-2 13B** | 13B | Enhanced reasoning | Complex reasoning tasks |
| **CodeLlama 7B** | 7B | Code & Math | 🧮 Mathematical reasoning, coding |
| **Falcon 7B** | 7B | General purpose | Balanced performance |
| **Vicuna 7B** | 7B | Conversation | Chat and dialogue |
| **Alpaca 7B** | 7B | Instruction following | Task completion |

## 🚀 **Quick Start**

### **Compare 3 Popular Models**
```bash
python meta_qa_eval.py --compare_models phi3_mini mistral_7b codellama_7b --num_samples 50
```

### **Mathematical Reasoning Focus**
```bash
python meta_qa_eval.py --dataset "meta-math/MetaMathQA-40K" --compare_models phi3_mini codellama_7b --graph_rag --num_samples 100
```

### **Movie Knowledge Comparison**
```bash
python meta_qa_eval.py --dataset "HiTruong/movie_QA" --compare_models phi3_mini llama2_7b --num_samples 75
```

## 📊 **Performance Expectations**

### **Mathematical Reasoning (MetaMathQA)**
- **Phi-3 Mini**: ~55-60% accuracy, fast inference
- **CodeLlama 7B**: ~50-55% accuracy, strong on computational problems
- **Mistral 7B**: ~50-55% accuracy, good general reasoning
- **Llama-2 7B**: ~45-50% accuracy, decent reasoning

### **Factual Knowledge (MovieQA)**
- **Phi-3 Mini**: ~75-80% accuracy, excellent factual recall
- **Llama-2 7B**: ~70-75% accuracy, good knowledge retention
- **Mistral 7B**: ~70-75% accuracy, multilingual knowledge
- **CodeLlama 7B**: ~65-70% accuracy, focused on technical domains

## 🔧 **Usage Patterns**

### **Model Selection Research**
```bash
# Test all major models
python meta_qa_eval.py --compare_models phi3_mini llama2_7b llama2_13b mistral_7b codellama_7b --num_samples 200

# Focus on efficiency vs accuracy trade-offs
python meta_qa_eval.py --compare_models phi3_mini llama2_13b --graph_rag --num_samples 150
```

### **Domain-Specific Evaluation**
```bash
# Mathematical reasoning specialists
python meta_qa_eval.py --dataset "meta-math/MetaMathQA-40K" --compare_models phi3_mini codellama_7b --graph_rag

# Factual knowledge specialists
python meta_qa_eval.py --dataset "HiTruong/movie_QA" --compare_models phi3_mini llama2_7b
```

### **Comprehensive Analysis**
```bash
# Full precision analysis
python meta_qa_eval.py --compare_models phi3_mini mistral_7b --precision_k 1 3 5 10 20 --num_samples 300

# Cross-dataset robustness
python meta_qa_eval.py --test_folder ./comprehensive_test_data --compare_models phi3_mini llama2_7b mistral_7b
```

## 📈 **Output Interpretation**

The comparison generates:

1. **Accuracy Rankings**: Direct performance comparison
2. **Precision@k Analysis**: Answer quality at different ranks
3. **Runtime Comparison**: Inference speed analysis
4. **Token Usage**: Efficiency metrics
5. **Best Model Identification**: Automatic winner selection

### **Sample Output**
```
🏆 MODEL COMPARISON RESULTS
================================================================================
Model                                    Accuracy   P@1      P@3      P@5      Runtime    Tokens
-------------------------------------------------------------------------------------------------
microsoft/Phi-3-mini-4k-instruct         55.6%     55.6%    68.7%    74.5%    4.23s      298
mistralai/Mistral-7B-Instruct-v0.1       52.1%     52.1%    65.9%    71.8%    3.87s      276
codellama/CodeLlama-7b-Instruct-hf       49.8%     49.8%    63.1%    69.3%    4.12s      312

🥇 Best performing model: microsoft/Phi-3-mini-4k-instruct (Accuracy: 55.6%)
```

## 🎯 **Model Selection Guidelines**

- **For Mathematical Reasoning**: Phi-3 Mini or CodeLlama 7B
- **For General Q&A**: Phi-3 Mini or Llama-2 7B
- **For Code Tasks**: CodeLlama 7B or Mistral 7B
- **For Multilingual**: Mistral 7B
- **For Large-Scale**: Llama-2 13B
- **For Speed**: Phi-3 Mini

## 🔍 **Adding New Models**

To add a new model to the framework:

1. **Add to MODEL_PRESETS** in `meta_qa_eval.py`:
```python
MODEL_PRESETS = {
    "your_model": "huggingface/your-model-name",
    # ...
}
```

2. **Test compatibility**:
```bash
python meta_qa_eval.py --model_name huggingface/your-model-name --num_samples 10
```

3. **Add to comparison**:
```bash
python meta_qa_eval.py --compare_models your_model phi3_mini --num_samples 50
```

## 💡 **Tips for Effective Comparison**

1. **Start Small**: Use `--num_samples 20-50` for quick tests
2. **Scale Up**: Use `--num_samples 200+` for research-grade results
3. **Enable Graph RAG**: Add `--graph_rag` for comprehensive analysis
4. **Use Multiple K Values**: `--precision_k 1 3 5 10` for detailed analysis
5. **Save Results**: Comparison results auto-save to timestamped JSON files

## 🚧 **Current Limitations**

- Models must be available on HuggingFace Hub
- Requires sufficient GPU memory for larger models
- Sequential evaluation (no parallel model loading)
- Some models may require specific tokenizer configurations

## 🔮 **Future Enhancements**

- Parallel model evaluation
- Custom model loading from local files
- Advanced statistical significance testing
- Model-specific optimization settings
- Ensemble evaluation capabilities