#!/usr/bin/env python3
"""
Test script for Graph RAG integration with MetaMathQA
"""

def test_basic_functionality():
    """Test basic imports and class creation"""
    try:
        from meta_qa_eval import (
            MathGraphRAG,
            MathKnowledgeGraphBuilder,
            MetaMathEvaluator,
            GraphQueryResult,
            GraphRAGMetrics
        )
        print("✓ All imports successful")

        # Test class creation
        kg_builder = MathKnowledgeGraphBuilder()
        print("✓ MathKnowledgeGraphBuilder created")

        evaluator = MetaMathEvaluator()
        print("✓ MetaMathEvaluator created")

        # Test precision metrics
        predicted = ['4', '3.5', '5']
        ground_truth = '4'
        metrics = evaluator.calculate_precision_at_k(predicted, ground_truth, [1, 3, 5])
        print(f"✓ Precision@k calculation works: {metrics}")

        # Test answer extraction
        solution = 'The calculation gives us: 2x + 5 = 13, so 2x = 8, therefore x = 4. The answer is 4.'
        answers = evaluator.extract_multiple_answers(solution, 3)
        print(f"✓ Answer extraction works: {answers}")

        print("\n🎉 Basic functionality test PASSED!")
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_command_line():
    """Test command line argument parsing"""
    try:
        import subprocess

        # Test help command
        result = subprocess.run(['python', 'meta_qa_eval.py', '--help'],
                              capture_output=True, text=True, timeout=10)

        if result.returncode == 0 and '--graph_rag' in result.stdout:
            print("✓ Command line arguments work")
            print("✓ Graph RAG option available")
            return True
        else:
            print(f"❌ Command line test failed: {result.stderr}")
            return False

    except Exception as e:
        print(f"❌ Command line test error: {e}")
        return False

if __name__ == "__main__":
    print("=== Graph RAG Integration Test ===\n")

    success = True

    print("1. Testing basic functionality...")
    if not test_basic_functionality():
        success = False

    print("\n2. Testing command line interface...")
    if not test_command_line():
        success = False

    print("\n" + "="*50)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("\nYou can now run:")
        print("  python meta_qa_eval.py --graph_rag --num_samples 50")
        print("  python meta_qa_eval.py --dataset meta-math/MetaMathQA-40K --graph_rag")
    else:
        print("❌ SOME TESTS FAILED!")

    print("="*50)