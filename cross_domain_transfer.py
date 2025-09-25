import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    T5Tokenizer,
    pipeline
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import evaluate
import nltk
from nltk.corpus import wordnet
import random
import jiwer  # For WER calculation
import logging
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import string

# Download required NLTK data
try:
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    pass

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CrossDomainEncoderEvaluator:
    """
    Evaluate encoder transfer capabilities across domains and languages
    with focus on consistency, robustness, and embedding quality
    """
    
    def __init__(self, encoder_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 generation_backend: str = "hf",
                 generation_model_name: str = "t5-small"):
        """
        Initialize the evaluator with a pre-trained encoder
        
        Args:
            encoder_model_name: Name of the encoder model to test
        """
        self.device = "cuda" if torch.cuda.is_available() else "mps"
        self.encoder_model_name = encoder_model_name

        # Generation backend settings
        self.generation_backend = generation_backend.lower()
        self.generation_model_name = generation_model_name

        if self.generation_backend == "mlx" and not MLX_AVAILABLE:
            raise ImportError("Requested generation_backend='mlx' but mlx-lm is not installed.\n"
                              "Install it via 'pip install --upgrade mlx-lm' or switch generation_backend='hf'.")
        
        # Load encoder
        logger.info(f"Loading encoder: {encoder_model_name}")
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(encoder_model_name)
        self.encoder_model = AutoModel.from_pretrained(encoder_model_name)
        self.encoder_model.to(self.device)
        
        # Initialize metrics
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")
        
        # Track results
        self.results = {}
        
    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Extract embeddings from the encoder"""
        embeddings = []
        
        for text in tqdm(texts, desc="Extracting embeddings"):
            inputs = self.encoder_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.encoder_model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)
    
    def test_cross_domain_transfer(self) -> Dict:
        """
        Test 1: Cross-domain transfer capabilities
        Train on one domain, test embedding quality on another
        """
        logger.info("=== Testing Cross-Domain Transfer ===")
        
        # Define domain datasets
        domains = {
            "news": {
                "dataset": "ag_news",
                "text_field": "text",
                "samples": 1000
            },
            "reviews": {
                "dataset": "amazon_polarity", 
                "text_field": "content",
                "samples": 1000
            },
            "scientific": {
                "dataset": "scientific_papers",
                "config": "arxiv",
                "text_field": "abstract",
                "samples": 500
            }
        }
        
        domain_embeddings = {}
        domain_texts = {}
        
        for domain_name, config in domains.items():
            try:
                logger.info(f"Processing {domain_name} domain...")
                
                # Load dataset
                if "config" in config:
                    dataset = load_dataset(config["dataset"], config["config"], split="train")
                else:
                    dataset = load_dataset(config["dataset"], split="train")
                
                # Sample texts
                texts = [item[config["text_field"]] for item in dataset.select(range(config["samples"]))]
                domain_texts[domain_name] = texts
                
                # Get embeddings
                embeddings = self.get_embeddings(texts)
                domain_embeddings[domain_name] = embeddings
                
            except Exception as e:
                logger.warning(f"Could not load {domain_name}: {e}")
                continue
        
        # Analyze cross-domain similarity
        transfer_results = self._analyze_cross_domain_similarity(domain_embeddings)
        
        self.results["cross_domain_transfer"] = {
            "domain_embeddings": domain_embeddings,
            "transfer_analysis": transfer_results,
            "domain_texts": domain_texts
        }
        
        return transfer_results
    
    def test_consistency_metrics(self, test_texts: List[str], num_runs: int = 5) -> Dict:
        """
        Test 2: Consistency metrics for text generation
        Generate multiple outputs for same input and measure consistency
        """
        logger.info("=== Testing Consistency Metrics ===")
        
        # Select generation backend (Hugging Face or MLX-LM)
        if self.generation_backend == "hf":
            generator = pipeline(
                "text2text-generation",
                model=self.generation_model_name,
                device=0 if self.device == "cuda" else -1
            )
        else:
            # MLX backend
            logger.info(f"Loading MLX-LM model: {self.generation_model_name}")
            mlx_model, mlx_tokenizer = mlx_load(self.generation_model_name)
        
        consistency_results = {
            "factual_consistency": [],
            "style_consistency": [],
            "embedding_consistency": [],
            "bleu_self_consistency": []
        }
        
        for text in tqdm(test_texts[:10], desc="Testing consistency"):  # Limit for demo
            generations = []
            
            # Generate multiple outputs
            for run in range(num_runs):
                try:
                    prompt = f"Summarize: {text}"

                    if self.generation_backend == "hf":
                        # Hugging Face generation
                        output = generator(prompt, max_length=100, do_sample=True, temperature=0.7)
                        generations.append(output[0]['generated_text'])
                    else:
                        # MLX-LM generation (chat-style template if available)
                        if hasattr(mlx_tokenizer, 'chat_template') and mlx_tokenizer.chat_template is not None:
                            chat_prompt = mlx_tokenizer.apply_chat_template(
                                [{"role": "user", "content": prompt}],
                                add_generation_prompt=True
                            )
                        else:
                            chat_prompt = prompt

                        gen_text = mlx_generate(
                            mlx_model,
                            mlx_tokenizer,
                            prompt=chat_prompt,
                            verbose=False,
                            temperature=0.7,
                            max_tokens=100
                        )
                        generations.append(gen_text)
                except:
                    generations.append("")
            
            # Calculate consistency metrics
            consistency_results["factual_consistency"].append(
                self._calculate_factual_consistency(generations)
            )
            consistency_results["style_consistency"].append(
                self._calculate_style_consistency(generations)
            )
            consistency_results["embedding_consistency"].append(
                self._calculate_embedding_consistency(generations)
            )
            consistency_results["bleu_self_consistency"].append(
                self._calculate_bleu_consistency(generations)
            )
        
        # Aggregate results
        final_consistency = {
            metric: np.mean(scores) for metric, scores in consistency_results.items()
        }
        
        self.results["consistency_metrics"] = final_consistency
        return final_consistency
    
    def test_adversarial_robustness(self, test_texts: List[str]) -> Dict:
        """
        Test 3: Adversarial robustness with synonym replacement and perturbations
        """
        logger.info("=== Testing Adversarial Robustness ===")
        
        robustness_results = {
            "synonym_robustness": [],
            "punctuation_robustness": [],
            "case_robustness": [],
            "tokenization_sensitivity": []
        }
        
        for text in tqdm(test_texts[:20], desc="Testing robustness"):  # Limit for demo
            original_embedding = self.get_embeddings([text])[0]
            
            # Test 1: Synonym replacement
            synonym_text = self._replace_with_synonyms(text)
            synonym_embedding = self.get_embeddings([synonym_text])[0]
            synonym_similarity = cosine_similarity([original_embedding], [synonym_embedding])[0][0]
            robustness_results["synonym_robustness"].append(synonym_similarity)
            
            # Test 2: Punctuation changes
            punct_text = self._modify_punctuation(text)
            punct_embedding = self.get_embeddings([punct_text])[0]
            punct_similarity = cosine_similarity([original_embedding], [punct_embedding])[0][0]
            robustness_results["punctuation_robustness"].append(punct_similarity)
            
            # Test 3: Case changes
            case_text = self._modify_case(text)
            case_embedding = self.get_embeddings([case_text])[0]
            case_similarity = cosine_similarity([original_embedding], [case_embedding])[0][0]
            robustness_results["case_robustness"].append(case_similarity)
            
            # Test 4: Tokenization sensitivity
            token_sensitivity = self._test_tokenization_sensitivity(text)
            robustness_results["tokenization_sensitivity"].append(token_sensitivity)
        
        # Aggregate results
        final_robustness = {
            metric: {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            } for metric, scores in robustness_results.items()
        }
        
        self.results["adversarial_robustness"] = final_robustness
        return final_robustness
    
    def test_advanced_metrics(self, reference_texts: List[str], generated_texts: List[str]) -> Dict:
        """
        Test 4: Advanced metrics beyond BLEU (WER, tokenization effects)
        """
        logger.info("=== Testing Advanced Metrics ===")
        
        # Ensure we have paired texts
        min_len = min(len(reference_texts), len(generated_texts))
        references = reference_texts[:min_len]
        hypotheses = generated_texts[:min_len]
        
        metrics = {}
        
        # 1. Word Error Rate (WER)
        try:
            wer_scores = []
            for ref, hyp in zip(references, hypotheses):
                wer = jiwer.wer(ref, hyp)
                wer_scores.append(wer)
            metrics["wer"] = {
                "mean": np.mean(wer_scores),
                "std": np.std(wer_scores)
            }
        except Exception as e:
            logger.warning(f"WER calculation failed: {e}")
            metrics["wer"] = None
        
        # 2. Character Error Rate (CER)
        try:
            cer_scores = []
            for ref, hyp in zip(references, hypotheses):
                cer = jiwer.cer(ref, hyp)
                cer_scores.append(cer)
            metrics["cer"] = {
                "mean": np.mean(cer_scores),
                "std": np.std(cer_scores)
            }
        except:
            metrics["cer"] = None
        
        # 3. BLEU with different tokenizations
        metrics["bleu_variants"] = self._test_bleu_tokenization_effects(references, hypotheses)
        
        # 4. Semantic similarity via embeddings
        ref_embeddings = self.get_embeddings(references)
        hyp_embeddings = self.get_embeddings(hypotheses)
        
        semantic_similarities = []
        for ref_emb, hyp_emb in zip(ref_embeddings, hyp_embeddings):
            sim = cosine_similarity([ref_emb], [hyp_emb])[0][0]
            semantic_similarities.append(sim)
        
        metrics["semantic_similarity"] = {
            "mean": np.mean(semantic_similarities),
            "std": np.std(semantic_similarities)
        }
        
        # 5. Length bias analysis
        metrics["length_bias"] = self._analyze_length_bias(references, hypotheses)
        
        self.results["advanced_metrics"] = metrics
        return metrics
    
    # Helper methods for calculations
    def _analyze_cross_domain_similarity(self, domain_embeddings: Dict) -> Dict:
        """Analyze how well embeddings transfer across domains"""
        results = {}
        
        domains = list(domain_embeddings.keys())
        
        for i, domain1 in enumerate(domains):
            for j, domain2 in enumerate(domains[i+1:], i+1):
                # Calculate average cosine similarity between domains
                emb1 = domain_embeddings[domain1]
                emb2 = domain_embeddings[domain2]
                
                # Sample for computational efficiency
                sample_size = min(100, len(emb1), len(emb2))
                sample1 = emb1[:sample_size]
                sample2 = emb2[:sample_size]
                
                # Calculate similarity matrix
                similarity_matrix = cosine_similarity(sample1, sample2)
                
                results[f"{domain1}_vs_{domain2}"] = {
                    "mean_similarity": np.mean(similarity_matrix),
                    "max_similarity": np.max(similarity_matrix),
                    "std_similarity": np.std(similarity_matrix)
                }
        
        return results
    
    def _calculate_factual_consistency(self, generations: List[str]) -> float:
        """Calculate factual consistency score (simplified)"""
        if len(generations) < 2:
            return 0.0
        
        # Simple approach: extract named entities and check consistency
        # This is a simplified version - in practice, you'd use more sophisticated NER
        entities_per_gen = []
        for gen in generations:
            # Extract capitalized words as proxy for entities
            entities = set(re.findall(r'\b[A-Z][a-z]+\b', gen))
            entities_per_gen.append(entities)
        
        if not entities_per_gen:
            return 1.0
        
        # Calculate intersection over union across all generations
        intersection = entities_per_gen[0]
        union = entities_per_gen[0]
        
        for entities in entities_per_gen[1:]:
            intersection = intersection.intersection(entities)
            union = union.union(entities)
        
        return len(intersection) / len(union) if len(union) > 0 else 1.0
    
    def _calculate_style_consistency(self, generations: List[str]) -> float:
        """Calculate style consistency based on length and complexity"""
        if len(generations) < 2:
            return 1.0
        
        # Calculate style features
        features = []
        for gen in generations:
            words = gen.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            sentence_count = len(gen.split('.'))
            features.append([len(gen), len(words), avg_word_length, sentence_count])
        
        features = np.array(features)
        
        # Calculate coefficient of variation for each feature
        cv_scores = []
        for i in range(features.shape[1]):
            mean_val = np.mean(features[:, i])
            std_val = np.std(features[:, i])
            cv = std_val / mean_val if mean_val > 0 else 0
            cv_scores.append(1 - cv)  # Higher consistency = lower coefficient of variation
        
        return np.mean(cv_scores)
    
    def _calculate_embedding_consistency(self, generations: List[str]) -> float:
        """Calculate consistency based on embedding similarity"""
        if len(generations) < 2:
            return 1.0
        
        embeddings = self.get_embeddings(generations)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                similarities.append(sim)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_bleu_consistency(self, generations: List[str]) -> float:
        """Calculate BLEU-based self-consistency"""
        if len(generations) < 2:
            return 1.0
        
        bleu_scores = []
        for i in range(len(generations)):
            references = [generations[j] for j in range(len(generations)) if j != i]
            hypothesis = generations[i]
            
            try:
                score = self.bleu_metric.compute(
                    predictions=[hypothesis],
                    references=[references]
                )['bleu']
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        return np.mean(bleu_scores)
    
    def _replace_with_synonyms(self, text: str, replacement_prob: float = 0.3) -> str:
        """Replace words with synonyms"""
        words = text.split()
        new_words = []
        
        for word in words:
            if random.random() < replacement_prob:
                # Find synonyms
                synonyms = []
                try:
                    for syn in wordnet.synsets(word):
                        for lemma in syn.lemmas():
                            synonyms.append(lemma.name())
                    
                    if synonyms:
                        new_word = random.choice(synonyms).replace('_', ' ')
                        new_words.append(new_word)
                    else:
                        new_words.append(word)
                except:
                    new_words.append(word)
            else:
                new_words.append(word)
        
        return ' '.join(new_words)
    
    def _modify_punctuation(self, text: str) -> str:
        """Modify punctuation in text"""
        # Add/remove some punctuation
        text = text.replace('.', '!')
        text = text.replace(',', ';')
        return text
    
    def _modify_case(self, text: str) -> str:
        """Modify case in text"""
        words = text.split()
        new_words = []
        for word in words:
            if random.random() < 0.3:
                new_words.append(word.upper() if word.islower() else word.lower())
            else:
                new_words.append(word)
        return ' '.join(new_words)
    
    def _test_tokenization_sensitivity(self, text: str) -> float:
        """Test sensitivity to different tokenization approaches"""
        # Get embedding with original tokenization
        original_embedding = self.get_embeddings([text])[0]
        
        # Test with different tokenizations
        variations = [
            text.replace(' ', ''),  # Remove spaces
            ' '.join(text),  # Add spaces between characters
            text.replace(' ', '_'),  # Replace spaces with underscores
        ]
        
        similarities = []
        for variation in variations:
            try:
                var_embedding = self.get_embeddings([variation])[0]
                sim = cosine_similarity([original_embedding], [var_embedding])[0][0]
                similarities.append(sim)
            except:
                similarities.append(0.0)
        
        return np.mean(similarities)
    
    def _test_bleu_tokenization_effects(self, references: List[str], hypotheses: List[str]) -> Dict:
        """Test BLEU with different tokenization approaches"""
        results = {}
        
        # Standard BLEU
        try:
            results["standard"] = self.bleu_metric.compute(
                predictions=hypotheses,
                references=[[ref] for ref in references]
            )['bleu']
        except:
            results["standard"] = 0.0
        
        # Character-level BLEU
        try:
            char_refs = [' '.join(ref) for ref in references]
            char_hyps = [' '.join(hyp) for hyp in hypotheses]
            results["character_level"] = self.bleu_metric.compute(
                predictions=char_hyps,
                references=[[ref] for ref in char_refs]
            )['bleu']
        except:
            results["character_level"] = 0.0
        
        return results
    
    def _analyze_length_bias(self, references: List[str], hypotheses: List[str]) -> Dict:
        """Analyze if there's a bias towards certain text lengths"""
        ref_lengths = [len(ref.split()) for ref in references]
        hyp_lengths = [len(hyp.split()) for hyp in hypotheses]
        
        return {
            "ref_mean_length": np.mean(ref_lengths),
            "hyp_mean_length": np.mean(hyp_lengths),
            "length_correlation": np.corrcoef(ref_lengths, hyp_lengths)[0][1] if len(ref_lengths) > 1 else 0,
            "length_difference": np.mean([abs(r-h) for r, h in zip(ref_lengths, hyp_lengths)])
        }
    
    def visualize_results(self):
        """Create visualizations of the evaluation results"""
        if not self.results:
            logger.warning("No results to visualize. Run evaluations first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Cross-domain transfer similarities
        if "cross_domain_transfer" in self.results:
            transfer_data = self.results["cross_domain_transfer"]["transfer_analysis"]
            domains = list(transfer_data.keys())
            similarities = [transfer_data[domain]["mean_similarity"] for domain in domains]
            
            axes[0, 0].bar(range(len(domains)), similarities)
            axes[0, 0].set_title("Cross-Domain Transfer Similarities")
            axes[0, 0].set_xticks(range(len(domains)))
            axes[0, 0].set_xticklabels(domains, rotation=45)
            axes[0, 0].set_ylabel("Mean Cosine Similarity")
        
        # Plot 2: Consistency metrics
        if "consistency_metrics" in self.results:
            consistency_data = self.results["consistency_metrics"]
            metrics = list(consistency_data.keys())
            scores = list(consistency_data.values())
            
            axes[0, 1].bar(range(len(metrics)), scores)
            axes[0, 1].set_title("Consistency Metrics")
            axes[0, 1].set_xticks(range(len(metrics)))
            axes[0, 1].set_xticklabels(metrics, rotation=45)
            axes[0, 1].set_ylabel("Consistency Score")
        
        # Plot 3: Adversarial robustness
        if "adversarial_robustness" in self.results:
            robustness_data = self.results["adversarial_robustness"]
            tests = list(robustness_data.keys())
            means = [robustness_data[test]["mean"] for test in tests]
            stds = [robustness_data[test]["std"] for test in tests]
            
            axes[1, 0].bar(range(len(tests)), means, yerr=stds, capsize=5)
            axes[1, 0].set_title("Adversarial Robustness")
            axes[1, 0].set_xticks(range(len(tests)))
            axes[1, 0].set_xticklabels(tests, rotation=45)
            axes[1, 0].set_ylabel("Robustness Score")
        
        # Plot 4: Advanced metrics
        if "advanced_metrics" in self.results:
            advanced_data = self.results["advanced_metrics"]
            
            # Create a summary plot
            metric_names = []
            metric_values = []
            
            for metric, value in advanced_data.items():
                if isinstance(value, dict) and "mean" in value:
                    metric_names.append(metric)
                    metric_values.append(value["mean"])
                elif isinstance(value, (int, float)):
                    metric_names.append(metric)
                    metric_values.append(value)
            
            if metric_names:
                axes[1, 1].bar(range(len(metric_names)), metric_values)
                axes[1, 1].set_title("Advanced Metrics")
                axes[1, 1].set_xticks(range(len(metric_names)))
                axes[1, 1].set_xticklabels(metric_names, rotation=45)
                axes[1, 1].set_ylabel("Score")
        
        plt.tight_layout()
        plt.show()
    
    def run_comprehensive_evaluation(self, sample_texts: List[str] = None) -> Dict:
        """Run all evaluation tests"""
        logger.info("Starting comprehensive evaluation...")
        
        # Use sample texts if provided, otherwise create some
        if sample_texts is None:
            sample_texts = [
                "The artificial intelligence system demonstrated remarkable performance across multiple domains.",
                "Climate change represents one of the most significant challenges of our time.",
                "Modern technology has revolutionized the way we communicate and work.",
                "Economic policies have far-reaching implications for social development.",
                "Scientific research continues to push the boundaries of human knowledge."
            ]
        
        # Run all tests
        cross_domain_results = self.test_cross_domain_transfer()
        consistency_results = self.test_consistency_metrics(sample_texts)
        robustness_results = self.test_adversarial_robustness(sample_texts)
        
        advanced_results = self.test_advanced_metrics(sample_texts, sample_texts)
        
        # Visualize results
        self.visualize_results()
        
        # Return summary
        return {
            "cross_domain_transfer": cross_domain_results,
            "consistency_metrics": consistency_results,
            "adversarial_robustness": robustness_results,
            "advanced_metrics": advanced_results,
            "summary": self._generate_summary()
        }
    
    def _generate_summary(self) -> Dict:
        """Generate a summary of all results"""
        summary = {}
        
        if "cross_domain_transfer" in self.results:
            transfer_data = self.results["cross_domain_transfer"]["transfer_analysis"]
            avg_similarity = np.mean([data["mean_similarity"] for data in transfer_data.values()])
            summary["cross_domain_score"] = avg_similarity
        
        if "consistency_metrics" in self.results:
            consistency_data = self.results["consistency_metrics"]
            avg_consistency = np.mean(list(consistency_data.values()))
            summary["consistency_score"] = avg_consistency
        
        if "adversarial_robustness" in self.results:
            robustness_data = self.results["adversarial_robustness"]
            avg_robustness = np.mean([data["mean"] for data in robustness_data.values()])
            summary["robustness_score"] = avg_robustness
        
        return summary

# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = CrossDomainEncoderEvaluator("sentence-transformers/all-MiniLM-L6-v2")
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n=== EVALUATION SUMMARY ===")
    for metric, score in results["summary"].items():
        print(f"{metric}: {score:.4f}")