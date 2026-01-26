from typing import List, Dict, Any
import json
import numpy as np
from datetime import datetime
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset

class RAGEvaluator:
    """Comprehensive evaluation using multiple metrics."""
    
    def __init__(self, quality_thresholds: Dict[str, float] = None):
        self.quality_thresholds = quality_thresholds or {
            'faithfulness': 0.90,
            'answer_relevancy': 0.85,
            'context_precision': 0.80,
            'context_recall': 0.75
        }
        self.evaluation_history = []
    
    def run_evaluation_suite(self, test_dataset: List[Dict], pipeline) -> Dict[str, Any]:
        """Run comprehensive evaluation on test dataset."""
        print(f"Running evaluation on {len(test_dataset)} test cases...")
        
        # Run pipeline on test cases
        results = []
        for i, test_case in enumerate(test_dataset):
            print(f"Processing test case {i+1}/{len(test_dataset)}")
            
            try:
                result = pipeline.query(test_case["question"])
                results.append({
                    "question": test_case["question"],
                    "answer": result["answer"],
                    "contexts": result["contexts"],
                    "ground_truth": test_case.get("reference_answer", "")
                })
            except Exception as e:
                print(f"Error processing test case {i+1}: {e}")
                continue
        
        if not results:
            return {"error": "No successful evaluations"}
        
        # Convert to Dataset format for Ragas
        dataset = Dataset.from_dict({
            "question": [r["question"] for r in results],
            "answer": [r["answer"] for r in results],
            "contexts": [r["contexts"] for r in results],
            "ground_truth": [r["ground_truth"] for r in results]
        })
        
        # Calculate metrics
        evaluation_result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
        )
        
        # Process results
        scores = {
            'faithfulness': float(evaluation_result['faithfulness']),
            'answer_relevancy': float(evaluation_result['answer_relevancy']),
            'context_precision': float(evaluation_result['context_precision']),
            'context_recall': float(evaluation_result['context_recall'])
        }
        
        # Log to monitoring system
        self._log_to_monitoring(scores, len(results))
        
        # Check against thresholds
        quality_gates = self._check_quality_gates(scores)
        
        evaluation_summary = {
            "timestamp": datetime.now().isoformat(),
            "num_test_cases": len(results),
            "scores": scores,
            "quality_gates": quality_gates,
            "passed_all_gates": all(quality_gates.values())
        }
        
        self.evaluation_history.append(evaluation_summary)
        
        return evaluation_summary
    
    def _log_to_monitoring(self, scores: Dict[str, float], num_cases: int):
        """Log evaluation results to monitoring system."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "scores": scores,
            "num_test_cases": num_cases
        }
        
        # In production, this would log to your monitoring system
        print(f"Evaluation logged: {json.dumps(log_entry, indent=2)}")
    
    def _check_quality_gates(self, scores: Dict[str, float]) -> Dict[str, bool]:
        """Check if scores meet quality thresholds."""
        gates = {}
        for metric, threshold in self.quality_thresholds.items():
            score = scores.get(metric, 0)
            gates[metric] = score >= threshold
        
        return gates
    
    def create_sample_dataset(self) -> List[Dict]:
        """Create a sample evaluation dataset."""
        return [
            {
                "question": "What is RAG and how does it work?",
                "reference_answer": "RAG stands for Retrieval-Augmented Generation. It combines retrieval systems with language models to generate responses based on retrieved documents."
            },
            {
                "question": "What are the main components of a RAG system?",
                "reference_answer": "The main components include: document ingestion, chunking, embedding generation, vector storage, retrieval, and generation."
            },
            {
                "question": "Why is hybrid search important in RAG?",
                "reference_answer": "Hybrid search combines vector search with keyword search (like BM25) to improve retrieval quality by capturing both semantic and lexical similarities."
            },
            {
                "question": "What is re-ranking in RAG systems?",
                "reference_answer": "Re-ranking is the process of reordering retrieved documents using a more sophisticated model (like cross-encoders) to improve relevance."
            },
            {
                "question": "How do you evaluate RAG system performance?",
                "reference_answer": "RAG systems are evaluated using metrics like faithfulness, answer relevancy, context precision, and context recall to measure the quality of retrieved information and generated answers."
            }
        ]
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations."""
        if not self.evaluation_history:
            return {"message": "No evaluations run yet"}
        
        latest = self.evaluation_history[-1]
        avg_scores = {}
        
        if len(self.evaluation_history) > 1:
            # Calculate average scores across all evaluations
            for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']:
                scores = [eval['scores'].get(metric, 0) for eval in self.evaluation_history]
                avg_scores[metric] = np.mean(scores)
        
        return {
            "latest_evaluation": latest,
            "total_evaluations": len(self.evaluation_history),
            "average_scores": avg_scores if avg_scores else latest["scores"],
            "trend": "improving" if len(self.evaluation_history) > 1 else "baseline"
        }
