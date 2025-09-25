#!/usr/bin/env python3
"""
Test multi-model comparison functionality
"""

import json
import os

def demonstrate_model_comparison():
    """Demonstrate different ways to use the multi-model comparison"""

    print("=" * 80)
    print("MULTI-MODEL COMPARISON EXAMPLES")
    print("=" * 80)

    examples = [
        {
            "name": "Basic Model Comparison",
            "description": "Compare 3 popular instruction-tuned models",
            "command": "python meta_qa_eval.py --compare_models phi3_mini mistral_7b codellama_7b --num_samples 50",
            "explanation": "Uses model presets for easy comparison of Phi-3, Mistral, and CodeLlama"
        },
        {
            "name": "Full Model Names",
            "description": "Compare models using full HuggingFace model names",
            "command": "python meta_qa_eval.py --compare_models microsoft/Phi-3-mini-4k-instruct meta-llama/Llama-2-7b-chat-hf --graph_rag",
            "explanation": "Direct model comparison with Graph RAG enabled"
        },
        {
            "name": "MovieQA Model Comparison",
            "description": "Compare models on movie question answering",
            "command": "python meta_qa_eval.py --dataset HiTruong/movie_QA --compare_models phi3_mini llama2_7b --num_samples 75",
            "explanation": "Model comparison on factual knowledge (movie data)"
        },
        {
            "name": "Mathematical Reasoning Comparison",
            "description": "Focus on mathematical reasoning capabilities",
            "command": "python meta_qa_eval.py --dataset meta-math/MetaMathQA-40K --compare_models phi3_mini mistral_7b codellama_7b --graph_rag --num_samples 100",
            "explanation": "Compare mathematical problem-solving with Graph RAG"
        },
        {
            "name": "Comprehensive Analysis",
            "description": "Large-scale model comparison with full metrics",
            "command": "python meta_qa_eval.py --compare_models phi3_mini llama2_7b llama2_13b mistral_7b --graph_rag --num_samples 200 --precision_k 1 3 5 10",
            "explanation": "Comprehensive evaluation with extended Precision@k metrics"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 60)
        print(f"Description: {example['description']}")
        print(f"Command: {example['command']}")
        print(f"Explanation: {example['explanation']}")

    print(f"\n{'=' * 80}")
    print("AVAILABLE MODEL PRESETS")
    print(f"{'=' * 80}")

    model_presets = {
        "phi3_mini": "microsoft/Phi-3-mini-4k-instruct",
        "llama2_7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama2_13b": "meta-llama/Llama-2-13b-chat-hf",
        "mistral_7b": "mistralai/Mistral-7B-Instruct-v0.1",
        "codellama_7b": "codellama/CodeLlama-7b-Instruct-hf",
        "falcon_7b": "tiiuae/falcon-7b-instruct",
        "vicuna_7b": "lmsys/vicuna-7b-v1.5",
        "alpaca_7b": "chavinlo/alpaca-native"
    }

    print("Preset Name          | Full Model Path")
    print("-" * 80)
    for preset, full_name in model_presets.items():
        print(f"{preset:<18} | {full_name}")

    print(f"\n{'=' * 80}")
    print("EXPECTED OUTPUT")
    print(f"{'=' * 80}")

    expected_output = """
    1. Model loading and initialization for each model
    2. Dataset processing (same for all models)
    3. Sequential evaluation of each model with progress bars
    4. Individual model results with metrics
    5. Comparative analysis table showing:
       - Model name
       - Accuracy percentage
       - Precision@1, @3, @5
       - Average runtime
       - Token usage
    6. Best performing model identification
    7. Results saved to JSON file with timestamp
    8. Individual Tensorboard logs for each model
    """

    print(expected_output)

    print(f"\n🎯 QUICK START:")
    print("python meta_qa_eval.py --compare_models phi3_mini mistral_7b --num_samples 20")

def create_sample_model_comparison_result():
    """Create a sample model comparison result for demonstration"""

    sample_results = {
        "microsoft/Phi-3-mini-4k-instruct": {
            "accuracy": 0.556,
            "precision@1": 0.556,
            "precision@3": 0.687,
            "precision@5": 0.745,
            "avg_runtime": 4.23,
            "avg_tokens": 298,
            "total_queries": 100,
            "successful_queries": 100
        },
        "mistralai/Mistral-7B-Instruct-v0.1": {
            "accuracy": 0.521,
            "precision@1": 0.521,
            "precision@3": 0.659,
            "precision@5": 0.718,
            "avg_runtime": 3.87,
            "avg_tokens": 276,
            "total_queries": 100,
            "successful_queries": 100
        },
        "codellama/CodeLlama-7b-Instruct-hf": {
            "accuracy": 0.498,
            "precision@1": 0.498,
            "precision@3": 0.631,
            "precision@5": 0.693,
            "avg_runtime": 4.12,
            "avg_tokens": 312,
            "total_queries": 100,
            "successful_queries": 100
        },
        "meta_info": {
            "timestamp": "2025-01-15 16:45:30",
            "dataset": "meta-math/MetaMathQA-40K",
            "num_samples": 100,
            "graph_rag_enabled": True,
            "precision_k_values": [1, 3, 5],
            "best_model": "microsoft/Phi-3-mini-4k-instruct",
            "best_accuracy": 0.556
        }
    }

    filename = "sample_model_comparison_results.json"
    with open(filename, 'w') as f:
        json.dump(sample_results, f, indent=2)

    print(f"\n📊 Sample results saved to: {filename}")

    # Display the comparison table
    print(f"\n{'=' * 100}")
    print("🏆 SAMPLE MODEL COMPARISON RESULTS")
    print(f"{'=' * 100}")
    print(f"{'Model':<40} {'Accuracy':<10} {'P@1':<8} {'P@3':<8} {'P@5':<8} {'Runtime':<10} {'Tokens':<8}")
    print("-" * 100)

    for model_name, metrics in sample_results.items():
        if model_name == "meta_info":
            continue

        accuracy = metrics["accuracy"] * 100
        p1 = metrics["precision@1"] * 100
        p3 = metrics["precision@3"] * 100
        p5 = metrics["precision@5"] * 100
        runtime = metrics["avg_runtime"]
        tokens = metrics["avg_tokens"]

        print(f"{model_name:<40} {accuracy:<10.1f}% {p1:<8.1f}% {p3:<8.1f}% {p5:<8.1f}% {runtime:<10.2f}s {tokens:<8.0f}")

    print(f"\n🥇 Best performing model: {sample_results['meta_info']['best_model']}")
    print(f"   Accuracy: {sample_results['meta_info']['best_accuracy']*100:.1f}%")

if __name__ == "__main__":
    demonstrate_model_comparison()
    create_sample_model_comparison_result()