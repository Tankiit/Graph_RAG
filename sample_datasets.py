#!/usr/bin/env python3
"""
Sample QA datasets for testing the evaluation framework
"""

import json
import os
from pathlib import Path

# Sample MetaMathQA data
METAMATH_SAMPLES = [
    {
        "query": "If 3x + 2 = 14, what is the value of x?",
        "response": "To solve 3x + 2 = 14:\nFirst, subtract 2 from both sides: 3x = 14 - 2 = 12\nThen divide by 3: x = 12/3 = 4\nTherefore, x = 4.",
        "type": "algebra"
    },
    {
        "query": "What is the area of a circle with radius 5?",
        "response": "The area of a circle is given by A = πr²\nWith radius r = 5:\nA = π × 5² = π × 25 = 25π\nTherefore, the area is 25π square units (approximately 78.54 square units).",
        "type": "geometry"
    },
    {
        "query": "Calculate the derivative of f(x) = x³ + 2x² - 5x + 1",
        "response": "To find the derivative, we apply the power rule:\nf'(x) = d/dx(x³) + d/dx(2x²) - d/dx(5x) + d/dx(1)\nf'(x) = 3x² + 4x - 5 + 0\nTherefore, f'(x) = 3x² + 4x - 5.",
        "type": "calculus"
    },
    {
        "query": "If you roll two dice, what's the probability of getting a sum of 7?",
        "response": "When rolling two dice, there are 6 × 6 = 36 possible outcomes.\nThe ways to get sum = 7 are: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1)\nThat's 6 favorable outcomes.\nProbability = 6/36 = 1/6 ≈ 0.167 or 16.67%",
        "type": "statistics"
    },
    {
        "query": "What is 127 + 89?",
        "response": "To add 127 + 89:\n127 + 89 = 216\nTherefore, 127 + 89 = 216.",
        "type": "arithmetic"
    }
]

# Sample MovieQA data
MOVIE_QA_SAMPLES = [
    {
        "question": "Who directed the movie 'Inception'?",
        "answer": "Christopher Nolan",
        "context": "Inception is a 2010 science fiction action film written and directed by Christopher Nolan.",
        "type": "movies"
    },
    {
        "question": "Which actor played the main character in 'The Matrix'?",
        "answer": "Keanu Reeves",
        "context": "Keanu Reeves plays Neo, the main character in The Matrix trilogy.",
        "type": "movies"
    },
    {
        "question": "What year was 'Titanic' released?",
        "answer": "1997",
        "context": "Titanic is a 1997 American epic romance and disaster film directed by James Cameron.",
        "type": "movies"
    },
    {
        "question": "Who won the Oscar for Best Actor in 2020?",
        "answer": "Joaquin Phoenix",
        "context": "Joaquin Phoenix won the Academy Award for Best Actor in 2020 for his role in 'Joker'.",
        "type": "movies"
    },
    {
        "question": "Which movie features the quote 'May the Force be with you'?",
        "answer": "Star Wars",
        "context": "The famous quote 'May the Force be with you' is from the Star Wars movie franchise.",
        "type": "movies"
    }
]

# Sample WikiQA data
WIKI_QA_SAMPLES = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "context": "Paris is the capital and most populous city of France.",
        "type": "general"
    },
    {
        "question": "When was the iPhone first released?",
        "answer": "2007",
        "context": "The original iPhone was announced by Steve Jobs on January 9, 2007, and was released on June 29, 2007.",
        "type": "general"
    },
    {
        "question": "What is the largest planet in our solar system?",
        "answer": "Jupiter",
        "context": "Jupiter is the largest planet in the Solar System and the fifth planet from the Sun.",
        "type": "general"
    },
    {
        "question": "Who wrote 'Romeo and Juliet'?",
        "answer": "William Shakespeare",
        "context": "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
        "type": "general"
    },
    {
        "question": "What is the speed of light?",
        "answer": "299,792,458 meters per second",
        "context": "The speed of light in vacuum is exactly 299,792,458 metres per second.",
        "type": "general"
    }
]

# Complex multi-hop reasoning samples
MULTI_HOP_SAMPLES = [
    {
        "question": "If the director of 'Inception' also directed 'The Dark Knight', and 'The Dark Knight' was released 2 years before 'Inception', what year was 'The Dark Knight' released?",
        "answer": "2008",
        "reasoning_steps": [
            "Inception was directed by Christopher Nolan",
            "The Dark Knight was also directed by Christopher Nolan",
            "Inception was released in 2010",
            "The Dark Knight was released 2 years before, so 2010 - 2 = 2008"
        ],
        "hops": 3,
        "type": "movies"
    },
    {
        "question": "If f(x) = x² + 3x and g(x) = 2x - 1, what is f(g(2))?",
        "answer": "24",
        "reasoning_steps": [
            "First calculate g(2) = 2(2) - 1 = 4 - 1 = 3",
            "Then calculate f(3) = 3² + 3(3) = 9 + 9 = 18",
            "Wait, let me recalculate: f(3) = 3² + 3(3) = 9 + 9 = 18",
            "Actually f(g(2)) = f(3) = 3² + 3(3) = 9 + 9 = 18"
        ],
        "hops": 2,
        "type": "algebra"
    },
    {
        "question": "If the area of a square is 64 square units, and this square is inscribed in a circle, what is the radius of the circle?",
        "answer": "4√2",
        "reasoning_steps": [
            "Area of square = 64, so side length = √64 = 8",
            "For a square inscribed in a circle, diagonal = diameter of circle",
            "Diagonal of square = side × √2 = 8√2",
            "Radius = diameter/2 = 8√2/2 = 4√2"
        ],
        "hops": 3,
        "type": "geometry"
    }
]

def create_sample_datasets():
    """Create sample dataset files for testing"""

    # Create test_datasets directory
    test_dir = Path("test_datasets")
    test_dir.mkdir(exist_ok=True)

    # Create MetaMathQA-style dataset
    metamath_data = {
        "train": METAMATH_SAMPLES,
        "test": METAMATH_SAMPLES[:3],  # Smaller test set
        "validation": METAMATH_SAMPLES[3:]
    }

    with open(test_dir / "sample_metamath.json", "w") as f:
        json.dump(metamath_data, f, indent=2)

    # Create MovieQA-style dataset
    movie_data = {
        "train": MOVIE_QA_SAMPLES,
        "test": MOVIE_QA_SAMPLES[:3],
        "validation": MOVIE_QA_SAMPLES[3:]
    }

    with open(test_dir / "sample_movieqa.json", "w") as f:
        json.dump(movie_data, f, indent=2)

    # Create WikiQA-style dataset
    wiki_data = {
        "train": WIKI_QA_SAMPLES,
        "test": WIKI_QA_SAMPLES[:3],
        "validation": WIKI_QA_SAMPLES[3:]
    }

    with open(test_dir / "sample_wikiqa.json", "w") as f:
        json.dump(wiki_data, f, indent=2)

    # Create multi-hop reasoning dataset
    multihop_data = {
        "train": MULTI_HOP_SAMPLES,
        "test": MULTI_HOP_SAMPLES[:2],
        "validation": MULTI_HOP_SAMPLES[2:]
    }

    with open(test_dir / "sample_multihop.json", "w") as f:
        json.dump(multihop_data, f, indent=2)

    return test_dir

def demonstrate_datasets():
    """Demonstrate the different dataset formats"""

    print("="*80)
    print("SAMPLE QA DATASETS FOR GRAPH RAG EVALUATION")
    print("="*80)

    print("\n📚 1. METAMATHQA SAMPLES")
    print("-" * 40)
    for i, sample in enumerate(METAMATH_SAMPLES[:2], 1):
        print(f"\nSample {i} ({sample['type'].upper()}):")
        print(f"Question: {sample['query']}")
        print(f"Solution: {sample['response'][:100]}...")

    print("\n🎬 2. MOVIEQA SAMPLES")
    print("-" * 40)
    for i, sample in enumerate(MOVIE_QA_SAMPLES[:2], 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Context: {sample['context']}")

    print("\n🌍 3. WIKIQA SAMPLES")
    print("-" * 40)
    for i, sample in enumerate(WIKI_QA_SAMPLES[:2], 1):
        print(f"\nSample {i}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Context: {sample['context']}")

    print("\n🧠 4. MULTI-HOP REASONING SAMPLES")
    print("-" * 40)
    for i, sample in enumerate(MULTI_HOP_SAMPLES[:1], 1):
        print(f"\nSample {i} ({sample['hops']}-hop):")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print("Reasoning Steps:")
        for j, step in enumerate(sample['reasoning_steps'], 1):
            print(f"  {j}. {step}")

    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    print(f"MetaMathQA samples: {len(METAMATH_SAMPLES)}")
    print(f"  - Algebra: {len([s for s in METAMATH_SAMPLES if s['type'] == 'algebra'])}")
    print(f"  - Geometry: {len([s for s in METAMATH_SAMPLES if s['type'] == 'geometry'])}")
    print(f"  - Calculus: {len([s for s in METAMATH_SAMPLES if s['type'] == 'calculus'])}")
    print(f"  - Statistics: {len([s for s in METAMATH_SAMPLES if s['type'] == 'statistics'])}")
    print(f"  - Arithmetic: {len([s for s in METAMATH_SAMPLES if s['type'] == 'arithmetic'])}")

    print(f"\nMovieQA samples: {len(MOVIE_QA_SAMPLES)}")
    print(f"WikiQA samples: {len(WIKI_QA_SAMPLES)}")
    print(f"Multi-hop samples: {len(MULTI_HOP_SAMPLES)}")
    print(f"  - 2-hop: {len([s for s in MULTI_HOP_SAMPLES if s['hops'] == 2])}")
    print(f"  - 3-hop: {len([s for s in MULTI_HOP_SAMPLES if s['hops'] == 3])}")

def test_with_framework():
    """Test sample datasets with the evaluation framework"""
    print("\n" + "="*80)
    print("TESTING WITH EVALUATION FRAMEWORK")
    print("="*80)

    try:
        from meta_qa_eval import UniversalQAProcessor

        # Test processor with sample data
        class MockDataset:
            def __init__(self, data):
                self.data = data

            def __getitem__(self, split):
                return self.data[split]

        # Test MetaMath processing
        print("\n🧮 Testing MetaMath Processing:")
        mock_dataset = MockDataset({"train": METAMATH_SAMPLES})
        processor = UniversalQAProcessor(mock_dataset, "meta-math/MetaMathQA")
        processor.process_dataset("train", 3)

        print(f"✅ Processed {len(processor.processed_data)} MetaMath samples")
        sample = processor.processed_data[0]
        print(f"   Question: {sample['question'][:50]}...")
        print(f"   Answer: {sample['answer']}")
        print(f"   Type: {sample['type']}")

        # Test Movie processing
        print("\n🎬 Testing Movie Processing:")
        mock_dataset = MockDataset({"train": MOVIE_QA_SAMPLES})
        processor = UniversalQAProcessor(mock_dataset, "movie_QA")
        processor.process_dataset("train", 3)

        print(f"✅ Processed {len(processor.processed_data)} Movie samples")
        sample = processor.processed_data[0]
        print(f"   Question: {sample['question']}")
        print(f"   Answer: {sample['answer']}")
        print(f"   Type: {sample['type']}")

        print(f"\n🎉 Framework testing successful!")

    except ImportError:
        print("❌ Cannot import evaluation framework. Run this from the main directory.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Demonstrate datasets
    demonstrate_datasets()

    # Create sample files
    print(f"\n📁 Creating sample dataset files...")
    test_dir = create_sample_datasets()
    print(f"✅ Sample datasets created in: {test_dir}")

    # Test with framework
    test_with_framework()

    print(f"\n🚀 USAGE EXAMPLES:")
    print(f"python meta_qa_eval.py --test_folder {test_dir}")
    print(f"python meta_qa_eval.py --graph_rag --num_samples 10")