import os
import json
from dotenv import load_dotenv
from langsmith.evaluation import evaluate, LangChainStringEvaluator
from langsmith.schemas import Run, Example
import google.generativeai as genai
from GraphRAG import BuildingCodeRAG 
from langsmith import Client, wrappers
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
from openai import OpenAI

load_dotenv()

# LangSmith setup
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = "building-code-rag-evaluation"

# Neo4j and API credentials
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


class RAGEvaluator:
    """Evaluation system for Building Code RAG systems"""
    
    def __init__(self, dataset_path: str = "evaluation_questions.json"):
        self.client = Client()
        self.dataset_path = dataset_path
        self.dataset_name = "building-code-qa-dataset"
        genai.configure(api_key=GOOGLE_API_KEY)
        
    def load_or_create_dataset(self):
        """Load questions from JSON and create/update LangSmith dataset"""
        
        # Load questions from JSON
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        
        # Check if dataset exists
        try:
            dataset = self.client.read_dataset(dataset_name=self.dataset_name)
            print(f"Found existing dataset: {self.dataset_name}")
        except:
            # Create new dataset
            dataset = self.client.create_dataset(
                dataset_name=self.dataset_name,
                description="Building Code Q&A evaluation dataset"
            )
            print(f"Created new dataset: {self.dataset_name}")
        
        # Add examples to dataset
        for item in questions_data:
            example = {
                "question": item["question"],
                "reference_answer": item.get("reference_answer", ""),
                "relevant_sections": item.get("relevant_sections", []),
                "category": item.get("category", "general")
            }
            
            try:
                self.client.create_example(
                    inputs={"question": example["question"]},
                    outputs={
                        "reference_answer": example["reference_answer"],
                        "relevant_sections": example["relevant_sections"]
                    },
                    metadata={"category": example["category"]},
                    dataset_id=dataset.id
                )
            except Exception as e:
                # Example might already exist, skip
                pass
        
        print(f"Dataset ready with examples")
        return dataset
    
    def create_rag_predictor(self, rag_system_name: str = "GraphRAG"):
        """Create a predictor function for the RAG system"""
        
        def predict(inputs: dict) -> dict:
            """Run RAG system on a question"""
            question = inputs["question"]
            
            # Initialize RAG system
            rag = BuildingCodeRAG(
                uri=NEO4J_URI,
                user=NEO4J_USER,
                password=NEO4J_PASSWORD,
                google_api_key=GOOGLE_API_KEY
            )
            
            try:
                result = rag.answer_question(question, top_k=3)
                return {
                    "answer": result["answer"],
                    "source_sections": result["source_sections"],
                    "prompt_length": result.get("prompt_length", 0)
                }
            finally:
                rag.close()
        
        return predict
    
    def create_llm_judge_evaluator(self):
        """Create an LLM-as-judge evaluator using Gemini"""
        
        def llm_judge(run: Run, example: Example) -> dict:
            """Evaluate answer quality using Gemini as judge"""
            
            question = example.inputs["question"]
            reference_answer = example.outputs.get("reference_answer", "")
            relevant_sections = example.outputs.get("relevant_sections", [])
            
            generated_answer = run.outputs["answer"]
            source_sections = run.outputs.get("source_sections", [])
            
            # Create evaluation prompt
            eval_prompt = f"""You are evaluating answers to building code questions. Rate the answer on multiple criteria.

Question: {question}

Generated Answer:
{generated_answer}

Source Sections Used: {', '.join(source_sections) if source_sections else 'None'}

Reference Answer (if available):
{reference_answer if reference_answer else 'No reference answer provided'}

Expected Relevant Sections: {', '.join(relevant_sections) if relevant_sections else 'Not specified'}

Please evaluate the generated answer on the following criteria (0-10 scale for each):

1. ACCURACY: Is the answer factually correct based on building code requirements?
2. COMPLETENESS: Does the answer address all parts of the question?
3. RELEVANCE: Does the answer stay focused on the question asked?
4. CITATIONS: Are appropriate section numbers cited?
5. CLARITY: Is the answer clear and well-structured?
6. SOURCE_QUALITY: Are the retrieved sections relevant to the question?

Provide your evaluation in the following JSON format:
{{
    "accuracy": <score 0-10>,
    "completeness": <score 0-10>,
    "relevance": <score 0-10>,
    "citations": <score 0-10>,
    "clarity": <score 0-10>,
    "source_quality": <score 0-10>,
    "overall": <average of all scores>,
    "reasoning": "<brief explanation of your scores>",
    "missing_sections": ["<any critical sections that should have been referenced but weren't>"]
}}

IMPORTANT: Respond ONLY with the JSON object, no other text."""

            try:
                model = genai.GenerativeModel('gemini-2.5-flash')
                response = model.generate_content(eval_prompt)
                
                # Parse JSON response
                response_text = response.text.strip()
                # Remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                scores = json.loads(response_text)
                
                return {
                    "key": "llm_judge_scores",
                    "score": scores["overall"],
                    "comment": json.dumps(scores, indent=2)
                }
                
            except Exception as e:
                print(f"Error in LLM judge evaluation: {e}")
                return {
                    "key": "llm_judge_scores",
                    "score": 0,
                    "comment": f"Evaluation failed: {str(e)}"
                }
        
        return llm_judge
    
    def create_retrieval_evaluator(self):
        """Evaluate if the correct sections were retrieved"""
        
        def retrieval_eval(run: Run, example: Example) -> dict:
            """Check if relevant sections were retrieved"""
            
            relevant_sections = set(example.outputs.get("relevant_sections", []))
            retrieved_sections = set(run.outputs.get("source_sections", []))
            
            if not relevant_sections:
                # No ground truth, can't evaluate
                return {
                    "key": "retrieval_quality",
                    "score": None,
                    "comment": "No ground truth sections provided"
                }
            
            # Calculate precision and recall
            if retrieved_sections:
                intersection = relevant_sections.intersection(retrieved_sections)
                precision = len(intersection) / len(retrieved_sections)
                recall = len(intersection) / len(relevant_sections)
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            else:
                precision = recall = f1 = 0
            
            return {
                "key": "retrieval_quality",
                "score": f1,
                "comment": json.dumps({
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "retrieved": list(retrieved_sections),
                    "expected": list(relevant_sections),
                    "missing": list(relevant_sections - retrieved_sections),
                    "extra": list(retrieved_sections - relevant_sections)
                }, indent=2)
            }
        
        return retrieval_eval
    
    def run_evaluation(self, experiment_name: str = "graphrag-baseline"):
        """Run the complete evaluation pipeline"""
        
        print("="*60)
        print(f"Starting RAG Evaluation: {experiment_name}")
        print("="*60)
        
        # Load dataset
        dataset = self.load_or_create_dataset()
        
        # Create predictor
        predictor = self.create_rag_predictor()
        
        # Create evaluators
        llm_judge = self.create_llm_judge_evaluator()
        retrieval_eval = self.create_retrieval_evaluator()
        
        # Run evaluation
        print("\nRunning evaluation...")
        results = evaluate(
            predictor,
            data=self.dataset_name,
            evaluators=[llm_judge, retrieval_eval],
            experiment_prefix=experiment_name,
            metadata={
                "rag_system": "GraphRAG",
                "top_k": 3,
                "model": "gemini-2.5-flash"
            }
        )
        
        print("\n" + "="*60)
        print("Evaluation Complete!")
        print("="*60)
        print(f"\nView results at: https://smith.langchain.com/")
        print(f"Project: building-code-rag-evaluation")
        print(f"Experiment: {experiment_name}")
        
        return results
    
    def compare_systems(self, system_configs: list):
        """
        Compare multiple RAG systems
        
        Args:
            system_configs: List of dicts with 'name' and 'predictor' keys
        """
        
        dataset = self.load_or_create_dataset()
        llm_judge = self.create_llm_judge_evaluator()
        retrieval_eval = self.create_retrieval_evaluator()
        
        all_results = {}
        
        for config in system_configs:
            name = config["name"]
            predictor = config["predictor"]
            
            print(f"\n{'='*60}")
            print(f"Evaluating: {name}")
            print(f"{'='*60}")
            
            results = evaluate(
                predictor,
                data=self.dataset_name,
                evaluators=[llm_judge, retrieval_eval],
                experiment_prefix=f"comparison-{name}",
                metadata=config.get("metadata", {})
            )
            
            all_results[name] = results
        
        print("\n" + "="*60)
        print("Comparison Complete!")
        print("="*60)
        print(f"\nView comparison at: https://smith.langchain.com/")
        
        return all_results


def create_sample_questions_file():
    """Create a sample evaluation questions JSON file"""
    
    sample_questions = [
        {
            "question": "What are the requirements for fire separation between dwelling units?",
            "reference_answer": "Dwelling units must be separated from each other by a fire separation with a fire-resistance rating of at least 1 hour.",
            "relevant_sections": ["3.3.1.3", "3.3.4.4"],
            "category": "fire_safety"
        },
        {
            "question": "What is the minimum ceiling height required for a habitable room?",
            "reference_answer": "The minimum ceiling height for habitable rooms is 2.3 m (7 ft 6 in).",
            "relevant_sections": ["9.5.3.1"],
            "category": "dimensions"
        },
        {
            "question": "How should cantilevered floor joists be attached to interior joists?",
            "reference_answer": "",  # No reference answer - let the system find it
            "relevant_sections": ["9.23.11.3"],
            "category": "structural"
        },
        {
            "question": "What are the ventilation requirements for bathrooms without windows?",
            "reference_answer": "",
            "relevant_sections": ["9.32.3.5"],
            "category": "ventilation"
        }
    ]
    
    with open("evaluation_questions.json", 'w', encoding='utf-8') as f:
        json.dump(sample_questions, f, indent=2, ensure_ascii=False)
    
    print("Created sample evaluation_questions.json file")


if __name__ == "__main__":
    # First time setup: create sample questions file
    if not os.path.exists("evaluation_questions.json"):
        print("Creating sample evaluation questions file...")
        create_sample_questions_file()
        print("\nPlease edit 'evaluation_questions.json' with your actual evaluation questions.")
        print("Then run this script again to perform the evaluation.\n")
    else:
        # Run evaluation
        evaluator = RAGEvaluator()
        results = evaluator.run_evaluation(experiment_name="graphrag-v1")
        
        # Example of how to compare systems later:
        # results = evaluator.compare_systems([
        #     {
        #         "name": "graphrag",
        #         "predictor": evaluator.create_rag_predictor("GraphRAG"),
        #         "metadata": {"system": "graph", "top_k": 3}
        #     },
        #     {
        #         "name": "naive-rag",
        #         "predictor": create_naive_rag_predictor(),  # You'll create this later
        #         "metadata": {"system": "naive", "top_k": 5}
        #     }
        # ])