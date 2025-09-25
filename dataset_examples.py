#!/usr/bin/env python3
"""
Real QA Dataset Examples and Test Data Generator
"""

# Real dataset structure examples from actual datasets

# 1. MetaMathQA Structure (from meta-math/MetaMathQA)
METAMATHQA_REAL_EXAMPLE = {
    "query": "Let's think step by step. Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4924 into muffins for her friends. If she sells the remainder at the farmers' market for $2 per egg, how much does she make per day?",
    "response": "Janet's ducks lay 16 eggs per day.\nShe eats 3 for breakfast.\nShe bakes 4924 into muffins.\nSo she has 16 - 3 - 4924 = -4911 eggs left.\nSince she can't have negative eggs, this problem doesn't make sense as stated. \nLet me assume she bakes 4 eggs into muffins instead.\nThen she has 16 - 3 - 4 = 9 eggs left.\nShe sells these at $2 per egg.\nSo she makes 9 * $2 = $18 per day.",
    "original": "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins for her friends. If she sells the remainder at the farmers' market for $2 per egg, how much does she make per day?"
}

# 2. WikiMovies Structure (from facebook/wiki_movies)
WIKIMOVIES_REAL_EXAMPLE = {
    "question": "what does Grégoire Colin appear in?",
    "answer": "Before the Rain"
}

# 3. WikiQA Structure (from microsoft/wiki_qa)
WIKIQA_REAL_EXAMPLE = {
    "question": "what is the difference between a firefighter and a paramedic",
    "answer": "Firefighters fight fires and often provide emergency medical services, while paramedics are specifically trained in emergency medical care.",
    "document_title": "Firefighter",
    "question_id": "Q1",
    "document_id": "D1"
}

# Sample test datasets for comprehensive evaluation
TEST_DATASETS = {
    "math_reasoning": [
        {
            "id": "math_001",
            "question": "If f(x) = 2x + 3 and g(x) = x², what is (f ∘ g)(2)?",
            "answer": "11",
            "solution_steps": [
                "First, find g(2) = 2² = 4",
                "Then, find f(g(2)) = f(4) = 2(4) + 3 = 8 + 3 = 11"
            ],
            "type": "algebra",
            "difficulty": "medium",
            "concepts": ["function composition", "polynomials"]
        },
        {
            "id": "math_002",
            "question": "A triangle has vertices at A(0,0), B(4,0), and C(2,3). What is its area?",
            "answer": "6",
            "solution_steps": [
                "Use the formula: Area = ½|x₁(y₂-y₃) + x₂(y₃-y₁) + x₃(y₁-y₂)|",
                "Area = ½|0(0-3) + 4(3-0) + 2(0-0)| = ½|0 + 12 + 0| = ½(12) = 6"
            ],
            "type": "geometry",
            "difficulty": "medium",
            "concepts": ["coordinate geometry", "area calculation"]
        },
        {
            "id": "math_003",
            "question": "What is the derivative of ln(x²) with respect to x?",
            "answer": "2/x",
            "solution_steps": [
                "Use the chain rule: d/dx[ln(u)] = (1/u) × du/dx",
                "Here u = x², so du/dx = 2x",
                "Therefore: d/dx[ln(x²)] = (1/x²) × 2x = 2x/x² = 2/x"
            ],
            "type": "calculus",
            "difficulty": "medium",
            "concepts": ["derivatives", "chain rule", "logarithms"]
        }
    ],

    "movie_qa": [
        {
            "id": "movie_001",
            "question": "In which year was the movie 'Pulp Fiction' released, and who directed it?",
            "answer": "1994, Quentin Tarantino",
            "context": {
                "movie_title": "Pulp Fiction",
                "release_year": 1994,
                "director": "Quentin Tarantino",
                "genre": ["Crime", "Drama"],
                "imdb_rating": 8.9
            },
            "type": "movies",
            "difficulty": "easy",
            "answer_type": "factual"
        },
        {
            "id": "movie_002",
            "question": "Which actor won an Oscar for their role in 'Joker' (2019)?",
            "answer": "Joaquin Phoenix",
            "context": {
                "movie_title": "Joker",
                "release_year": 2019,
                "lead_actor": "Joaquin Phoenix",
                "award": "Academy Award for Best Actor",
                "award_year": 2020
            },
            "type": "movies",
            "difficulty": "medium",
            "answer_type": "factual"
        }
    ],

    "multi_hop_reasoning": [
        {
            "id": "multihop_001",
            "question": "If the director of 'Inception' was born in 1970, and 'Inception' was released when he was 40 years old, what year was 'Inception' released?",
            "answer": "2010",
            "reasoning_chain": [
                {
                    "step": 1,
                    "reasoning": "Identify that Christopher Nolan directed 'Inception'",
                    "entities": ["Christopher Nolan", "Inception"],
                    "relation": "director_of"
                },
                {
                    "step": 2,
                    "reasoning": "Christopher Nolan was born in 1970",
                    "entities": ["Christopher Nolan", "1970"],
                    "relation": "birth_year"
                },
                {
                    "step": 3,
                    "reasoning": "If he was 40 when Inception was released: 1970 + 40 = 2010",
                    "entities": ["1970", "40", "2010"],
                    "relation": "arithmetic"
                }
            ],
            "type": "multi_hop",
            "hops": 3,
            "difficulty": "hard",
            "domains": ["movies", "arithmetic"]
        },
        {
            "id": "multihop_002",
            "question": "If the area of a square is equal to the area of a circle with radius 3, what is the side length of the square?",
            "answer": "3√π",
            "reasoning_chain": [
                {
                    "step": 1,
                    "reasoning": "Calculate area of circle: A = πr² = π(3)² = 9π",
                    "entities": ["circle", "radius", "3", "9π"],
                    "relation": "area_formula"
                },
                {
                    "step": 2,
                    "reasoning": "Square area equals circle area: s² = 9π",
                    "entities": ["square", "area", "9π"],
                    "relation": "equal_areas"
                },
                {
                    "step": 3,
                    "reasoning": "Solve for side length: s = √(9π) = 3√π",
                    "entities": ["s", "√(9π)", "3√π"],
                    "relation": "square_root"
                }
            ],
            "type": "multi_hop",
            "hops": 3,
            "difficulty": "hard",
            "domains": ["geometry", "algebra"]
        }
    ],

    "knowledge_graph_test": [
        {
            "id": "kg_001",
            "question": "What is the relationship between differentiation and integration in calculus?",
            "answer": "They are inverse operations (Fundamental Theorem of Calculus)",
            "knowledge_entities": [
                {"entity": "differentiation", "type": "mathematical_operation"},
                {"entity": "integration", "type": "mathematical_operation"},
                {"entity": "Fundamental Theorem of Calculus", "type": "theorem"},
                {"entity": "inverse operations", "type": "relationship"}
            ],
            "knowledge_relations": [
                {"subject": "differentiation", "predicate": "inverse_of", "object": "integration"},
                {"subject": "Fundamental Theorem of Calculus", "predicate": "establishes", "object": "inverse operations"},
            ],
            "type": "conceptual",
            "difficulty": "medium",
            "requires_reasoning": True
        }
    ]
}

def print_dataset_examples():
    """Print examples of different dataset structures"""

    print("=" * 100)
    print("COMPREHENSIVE QA DATASET EXAMPLES")
    print("=" * 100)

    print(f"\n📚 1. METAMATHQA REAL STRUCTURE")
    print("-" * 60)
    example = METAMATHQA_REAL_EXAMPLE
    print(f"Query: {example['query'][:100]}...")
    print(f"Response: {example['response'][:200]}...")
    print(f"Original: {example.get('original', 'N/A')[:80]}...")

    print(f"\n🎬 2. WIKIMOVIES REAL STRUCTURE")
    print("-" * 60)
    example = WIKIMOVIES_REAL_EXAMPLE
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")

    print(f"\n🌐 3. WIKIQA REAL STRUCTURE")
    print("-" * 60)
    example = WIKIQA_REAL_EXAMPLE
    print(f"Question: {example['question']}")
    print(f"Answer: {example['answer']}")
    print(f"Document Title: {example.get('document_title', 'N/A')}")

    print(f"\n🧮 4. ENHANCED MATH REASONING")
    print("-" * 60)
    math_example = TEST_DATASETS["math_reasoning"][0]
    print(f"ID: {math_example['id']}")
    print(f"Question: {math_example['question']}")
    print(f"Answer: {math_example['answer']}")
    print(f"Type: {math_example['type']} | Difficulty: {math_example['difficulty']}")
    print(f"Concepts: {', '.join(math_example['concepts'])}")
    print("Solution Steps:")
    for i, step in enumerate(math_example['solution_steps'], 1):
        print(f"  {i}. {step}")

    print(f"\n🎭 5. ENHANCED MOVIE QA")
    print("-" * 60)
    movie_example = TEST_DATASETS["movie_qa"][0]
    print(f"ID: {movie_example['id']}")
    print(f"Question: {movie_example['question']}")
    print(f"Answer: {movie_example['answer']}")
    print(f"Context: {movie_example['context']}")

    print(f"\n🧠 6. MULTI-HOP REASONING")
    print("-" * 60)
    multihop_example = TEST_DATASETS["multi_hop_reasoning"][0]
    print(f"ID: {multihop_example['id']} | Hops: {multihop_example['hops']}")
    print(f"Question: {multihop_example['question']}")
    print(f"Answer: {multihop_example['answer']}")
    print("Reasoning Chain:")
    for step in multihop_example['reasoning_chain']:
        print(f"  Step {step['step']}: {step['reasoning']}")
        print(f"    Entities: {step['entities']}")
        print(f"    Relation: {step['relation']}")

    print(f"\n🕸️ 7. KNOWLEDGE GRAPH STRUCTURE")
    print("-" * 60)
    kg_example = TEST_DATASETS["knowledge_graph_test"][0]
    print(f"ID: {kg_example['id']}")
    print(f"Question: {kg_example['question']}")
    print(f"Answer: {kg_example['answer']}")
    print("Knowledge Entities:")
    for entity in kg_example['knowledge_entities']:
        print(f"  - {entity['entity']} ({entity['type']})")
    print("Knowledge Relations:")
    for rel in kg_example['knowledge_relations']:
        print(f"  - {rel['subject']} → {rel['predicate']} → {rel['object']}")

def create_test_files():
    """Create JSON test files"""
    import json
    from pathlib import Path

    test_dir = Path("comprehensive_test_data")
    test_dir.mkdir(exist_ok=True)

    for dataset_name, data in TEST_DATASETS.items():
        file_path = test_dir / f"{dataset_name}.json"
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    print(f"\n📁 Created comprehensive test datasets in: {test_dir}")
    print(f"Files created:")
    for file_path in test_dir.glob("*.json"):
        print(f"  - {file_path.name}")

    return test_dir

def usage_examples():
    """Show usage examples"""

    print(f"\n" + "=" * 100)
    print("USAGE EXAMPLES WITH DIFFERENT DATASETS")
    print("=" * 100)

    examples = [
        {
            "name": "MetaMathQA Evaluation",
            "command": "python meta_qa_eval.py --dataset 'meta-math/MetaMathQA-40K' --graph_rag --num_samples 100",
            "description": "Full Graph RAG evaluation on mathematical reasoning"
        },
        {
            "name": "MovieQA Evaluation",
            "command": "python meta_qa_eval.py --dataset 'HiTruong/movie_QA' --graph_rag --num_samples 50",
            "description": "Movie question answering with graph reasoning"
        },
        {
            "name": "Custom Test Data",
            "command": "python meta_qa_eval.py --test_folder comprehensive_test_data --graph_rag",
            "description": "Use local comprehensive test datasets"
        },
        {
            "name": "High Precision Evaluation",
            "command": "python meta_qa_eval.py --graph_rag --precision_k 1 3 5 10 20 --num_samples 200",
            "description": "Detailed precision analysis with multiple k values"
        },
        {
            "name": "Custom Dataset Splits",
            "command": "python meta_qa_eval.py --graph_rag --train_ratio 0.8 --test_ratio 0.15",
            "description": "Custom train/test/holdout ratios (80%/15%/5%)"
        }
    ]

    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}:")
        print(f"   Command: {example['command']}")
        print(f"   Description: {example['description']}")

if __name__ == "__main__":
    print_dataset_examples()
    test_dir = create_test_files()
    usage_examples()

    print(f"\n🚀 QUICK TEST:")
    print(f"python sample_datasets.py  # View sample data")
    print(f"python dataset_examples.py  # View comprehensive examples")
    print(f"python meta_qa_eval.py --help  # See all options")