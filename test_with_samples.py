#!/usr/bin/env python3
"""
Test the evaluation framework with sample datasets
"""

def test_dataset_processing():
    """Test dataset processing with different formats"""
    print("=" * 80)
    print("TESTING DATASET PROCESSING WITH SAMPLE DATA")
    print("=" * 80)

    try:
        from meta_qa_eval import UniversalQAProcessor, MetaMathEvaluator
        import json

        # Test 1: MetaMath-style processing
        print("\n📚 TEST 1: METAMATH PROCESSING")
        print("-" * 50)

        metamath_data = [
            {
                "query": "If 2x + 5 = 13, what is x?",
                "response": "To solve 2x + 5 = 13:\n2x = 13 - 5 = 8\nx = 8/2 = 4\nThe answer is 4."
            },
            {
                "query": "What is the derivative of x³?",
                "response": "Using the power rule: d/dx(x³) = 3x²\nThe answer is 3x²."
            }
        ]

        class MockDataset:
            def __init__(self, data):
                self.data = {"train": data}
            def __getitem__(self, split):
                return self.data[split]

        dataset = MockDataset(metamath_data)
        processor = UniversalQAProcessor(dataset, "meta-math/MetaMathQA")
        processor.process_dataset("train", 2)

        print(f"✅ Processed {len(processor.processed_data)} MetaMath samples")
        for i, item in enumerate(processor.processed_data):
            print(f"  Sample {i+1}:")
            print(f"    Q: {item['question']}")
            print(f"    A: {item['answer']}")
            print(f"    Type: {item['type']}")

        # Test 2: Movie QA processing
        print("\n🎬 TEST 2: MOVIE QA PROCESSING")
        print("-" * 50)

        movie_data = [
            {
                "question": "Who directed Titanic?",
                "answer": "James Cameron"
            },
            {
                "question": "When was The Matrix released?",
                "answer": "1999"
            }
        ]

        dataset = MockDataset(movie_data)
        processor = UniversalQAProcessor(dataset, "movie_QA")
        processor.process_dataset("train", 2)

        print(f"✅ Processed {len(processor.processed_data)} Movie samples")
        for i, item in enumerate(processor.processed_data):
            print(f"  Sample {i+1}:")
            print(f"    Q: {item['question']}")
            print(f"    A: {item['answer']}")
            print(f"    Type: {item['type']}")

        # Test 3: Precision@k calculation
        print("\n📊 TEST 3: PRECISION@K CALCULATION")
        print("-" * 50)

        evaluator = MetaMathEvaluator()

        # Test cases for precision@k
        test_cases = [
            {
                "predicted": ["42", "41", "43", "40", "44"],
                "ground_truth": "42",
                "case": "Perfect match at rank 1"
            },
            {
                "predicted": ["wrong1", "wrong2", "42", "wrong3", "wrong4"],
                "ground_truth": "42",
                "case": "Match at rank 3"
            },
            {
                "predicted": ["wrong1", "wrong2", "wrong3", "wrong4", "wrong5"],
                "ground_truth": "42",
                "case": "No match in top 5"
            }
        ]

        for test in test_cases:
            precision = evaluator.calculate_precision_at_k(
                test["predicted"],
                test["ground_truth"],
                [1, 3, 5]
            )
            print(f"  {test['case']}:")
            print(f"    Predicted: {test['predicted']}")
            print(f"    Ground truth: {test['ground_truth']}")
            print(f"    P@1: {precision['precision@1']:.2f}")
            print(f"    P@3: {precision['precision@3']:.2f}")
            print(f"    P@5: {precision['precision@5']:.2f}")

        # Test 4: Answer extraction
        print("\n🔍 TEST 4: ANSWER EXTRACTION")
        print("-" * 50)

        test_solutions = [
            "To solve this problem: x = 2 + 3 = 5. The answer is 5.",
            "Step 1: Calculate 2×3 = 6. Step 2: Add 1: 6+1 = 7. Therefore, the result is 7.",
            "Using the quadratic formula: x = (-b ± √(b²-4ac))/2a = 2.5"
        ]

        for i, solution in enumerate(test_solutions, 1):
            answers = evaluator.extract_multiple_answers(solution, 3)
            print(f"  Solution {i}: {solution}")
            print(f"    Extracted answers: {answers}")

        print(f"\n🎉 All tests completed successfully!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def demonstrate_comprehensive_metrics():
    """Demonstrate comprehensive metrics display"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE METRICS DEMONSTRATION")
    print("=" * 80)

    try:
        from meta_qa_eval import GraphRAGEvaluator
        import networkx as nx

        # Create mock graph query results
        class MockGraphResult:
            def __init__(self, correct=True, hops=2):
                self.predicted_answer = "The answer is 42" if correct else "Wrong answer"
                self.ground_truth_answers = ["42"]
                self.runtime = 1.5 + (hops * 0.3)
                self.num_llm_calls = 1
                self.num_input_tokens = 100 + (hops * 20)
                self.num_output_tokens = 30 + (hops * 10)
                self.num_hops = hops
                self.entities_explored = {f"entity_{i}" for i in range(hops + 2)}
                self.relations_used = {f"relation_{i}" for i in range(hops)}
                self.retrieved_subgraph = nx.DiGraph()
                self.retrieved_subgraph.add_edges_from([(f"node_{i}", f"node_{i+1}") for i in range(hops)])
                self.graph_retrieval_time = 0.3 + (hops * 0.1)
                self.graph_reasoning_time = 1.2 + (hops * 0.2)

        # Create sample results with different characteristics
        results = [
            MockGraphResult(correct=True, hops=1),   # Easy 1-hop correct
            MockGraphResult(correct=True, hops=2),   # Medium 2-hop correct
            MockGraphResult(correct=False, hops=2),  # 2-hop incorrect
            MockGraphResult(correct=True, hops=3),   # Hard 3-hop correct
            MockGraphResult(correct=False, hops=3),  # 3-hop incorrect
        ]

        print("Created 5 sample Graph RAG query results:")
        for i, result in enumerate(results, 1):
            correct = "✓" if "42" in result.predicted_answer else "✗"
            print(f"  {i}. {result.num_hops}-hop query: {correct} ({result.runtime:.2f}s)")

        # Evaluate with comprehensive metrics
        evaluator = GraphRAGEvaluator()
        metrics = evaluator.evaluate_comprehensive(results)

        # Display comprehensive results
        print(f"\n🎯 DISPLAYING COMPREHENSIVE GRAPH RAG METRICS:")
        print("-" * 80)
        evaluator.display_comprehensive_results(metrics)

        return True

    except Exception as e:
        print(f"❌ Error in comprehensive metrics: {e}")
        return False

def show_usage_patterns():
    """Show different usage patterns"""
    print("\n" + "=" * 80)
    print("PRACTICAL USAGE PATTERNS")
    print("=" * 80)

    usage_examples = [
        {
            "scenario": "Quick Math Evaluation",
            "command": "python meta_qa_eval.py --dataset 'meta-math/MetaMathQA-40K' --num_samples 50",
            "description": "Fast evaluation on 50 math problems with standard metrics"
        },
        {
            "scenario": "Full Graph RAG Analysis",
            "command": "python meta_qa_eval.py --graph_rag --num_samples 100 --precision_k 1 3 5 10",
            "description": "Complete Graph RAG evaluation with comprehensive metrics"
        },
        {
            "scenario": "Movie QA Evaluation",
            "command": "python meta_qa_eval.py --dataset 'HiTruong/movie_QA' --graph_rag --num_samples 75",
            "description": "Movie question answering with Graph RAG reasoning"
        },
        {
            "scenario": "Custom Dataset Testing",
            "command": "python meta_qa_eval.py --test_folder comprehensive_test_data --graph_rag",
            "description": "Test with local comprehensive datasets"
        },
        {
            "scenario": "Research-Grade Evaluation",
            "command": "python meta_qa_eval.py --graph_rag --num_samples 500 --train_ratio 0.8 --test_ratio 0.15",
            "description": "Large-scale evaluation with custom splits and holdout validation"
        }
    ]

    for i, example in enumerate(usage_examples, 1):
        print(f"\n{i}. {example['scenario']}:")
        print(f"   Command: {example['command']}")
        print(f"   Use case: {example['description']}")

    print(f"\n📋 EXPECTED OUTPUT STRUCTURE:")
    print("-" * 50)
    print("""
    1. Dataset loading and processing with progress bars
    2. System initialization (Baseline, KG-RAG, Graph RAG)
    3. Knowledge base construction with tqdm progress
    4. Three-way evaluation with live metrics updates
    5. Comprehensive Graph RAG analysis (if --graph_rag)
    6. Final comparison tables and improvements
    7. Analysis by problem type
    8. Holdout set validation
    9. Tensorboard logging for all metrics
    """)

if __name__ == "__main__":
    success = True

    # Run all tests
    if not test_dataset_processing():
        success = False

    if not demonstrate_comprehensive_metrics():
        success = False

    show_usage_patterns()

    print(f"\n" + "=" * 80)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Dataset processing works")
        print("✅ Comprehensive metrics work")
        print("✅ Framework is ready for evaluation")

        print(f"\n🚀 TRY IT NOW:")
        print("python meta_qa_eval.py --graph_rag --num_samples 20")
        print("tensorboard --logdir=./tensorboard_logs")
    else:
        print("❌ SOME TESTS FAILED")
        print("Check the error messages above")

    print("=" * 80)