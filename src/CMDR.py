import cohere
import os
import json
import numpy as np
from datasets import load_dataset
from dotenv import load_dotenv
from tqdm import tqdm
from helpers import query_model, get_text_from_response, get_logprobs_from_response

# Load .env file
load_dotenv()
COHERE_API = os.getenv("COHERE_API")

# Initialize Cohere Client
co = cohere.ClientV2(api_key=COHERE_API)

print("check-1")
## ==========================================
# PIPELINE CLASS
# ==========================================
class GenerativeCrossLingualPipeline:
    def __init__(self, model_name="tiny-aya-water"):
        """Initializes the pipeline with the specified model."""
        self.model_name = model_name
        print(f"Initialized pipeline for API model: {self.model_name}")

    def load_parallel_data(self, num_samples=1000):
        """Step 1: Curates a parallel evaluation corpus using XNLI."""
        print("Loading XNLI dataset for French and vietnamese...")
        en_data = load_dataset("xnli", "fr", split="test")
        hi_data = load_dataset("xnli", "vi", split="test")
        
        # Ensure perfect alignment by zipping the premises and hypotheses
        parallel_pairs = list(zip(
            en_data['premise'][:num_samples], en_data['hypothesis'][:num_samples],
            hi_data['premise'][:num_samples], hi_data['hypothesis'][:num_samples]
        ))
        return parallel_pairs

    def calculate_sequence_probability(self, logprobs_data):
        """Helper to convert token logprobs into a single probability score."""
        if not logprobs_data:
            return 0.0

        # Cohere LogprobItem returns a list of logprobs per chunk.
        # Flatten the per-chunk values into one sequence-level score.
        log_probs = []
        for token in logprobs_data:
            if hasattr(token, "logprobs") and token.logprobs:
                log_probs.extend(token.logprobs)
            if hasattr(token, "logprob"):
                log_probs.append(token.logprob)
            elif hasattr(token, "log_probability"):
                log_probs.append(token.log_probability)

        if not log_probs:
            return 0.0

        mean_logprob = np.mean(log_probs)
        return np.exp(mean_logprob)

    def calculate_label_disagreement(self, label_e, label_h):
        """Calculates if the model changed its mind (1 for disagree, 0 for agree)."""
        return 0.0 if label_e.strip().lower() == label_h.strip().lower() else 1.0

    def calculate_confidence_distance(self, prob_e, prob_h):
        """Calculates the absolute difference in model confidence."""
        return abs(prob_e - prob_h)

    def run_pipeline(self, num_samples=100):
        """Executes the data loading, querying, and metric calculation."""
        parallel_pairs = self.load_parallel_data(num_samples)
        
        all_label_disagreements = []
        all_conf_disagreements = []
        
        prompt_template = """Analyze the premise and hypothesis. 
Determine if the relationship is 'entailment', 'neutral', or 'contradiction'. 
You MUST output a valid JSON object with a single key 'label'.

Premise: {premise}
Hypothesis: {hypothesis}"""

        print("Running dual-path inference via API and calculating metrics...")
        
        for en_prem, en_hyp, hi_prem, hi_hyp in tqdm(parallel_pairs):
            en_query = prompt_template.format(premise=en_prem, hypothesis=en_hyp)
            hi_query = prompt_template.format(premise=hi_prem, hypothesis=hi_hyp)
            
            try:
                # Path A (English)
                en_resp = query_model(en_query, model=self.model_name, logprobs=True)
                en_text = get_text_from_response(en_resp)
                en_label = json.loads(en_text).get("label", "")
                en_prob = self.calculate_sequence_probability(get_logprobs_from_response(en_resp))
                
                # Path B (Hindi)
                hi_resp = query_model(hi_query, model=self.model_name, logprobs=True)
                hi_text = get_text_from_response(hi_resp)
                hi_label = json.loads(hi_text).get("label", "")
                hi_prob = self.calculate_sequence_probability(get_logprobs_from_response(hi_resp))
                
                # Metric Calculation
                label_dist = self.calculate_label_disagreement(en_label, hi_label)
                conf_dist = self.calculate_confidence_distance(en_prob, hi_prob)
                
                all_label_disagreements.append(label_dist)
                all_conf_disagreements.append(conf_dist)
                
            except Exception as e:
                # Catch JSON decoding errors or API timeouts
                print(f"Error processing pair: {e}")
                continue

        # Aggregation and Statistical Analysis
        label_disagreement_array = np.array(all_label_disagreements)
        conf_disagreement_array = np.array(all_conf_disagreements)
        
        results = {
            "Label Disagreement": {
                "Mean": np.mean(label_disagreement_array) if len(label_disagreement_array) > 0 else 0,
                "Variance": np.var(label_disagreement_array) if len(label_disagreement_array) > 0 else 0
            },
            "Confidence Disagreement": {
                "Mean": np.mean(conf_disagreement_array) if len(conf_disagreement_array) > 0 else 0,
                "Variance": np.var(conf_disagreement_array) if len(conf_disagreement_array) > 0 else 0
            }
        }
        
        self.print_results(results)
        return results

    def print_results(self, results):
        """Formats and prints the interpretation of the results."""
        print("\n" + "="*50)
        print("GENERATIVE CROSS-LINGUAL DISAGREEMENT RESULTS")
        print("="*50)
        for metric, stats in results.items():
            print(f"{metric}:")
            print(f"  Mean:     {stats['Mean']:.4f}")
            print(f"  Variance: {stats['Variance']:.4f}\n")
            
        print("Interpretation Guide:")
        print("* A Label Disagreement mean of 0.30 means the model changes its prediction 30% of the time based purely on language.")
        print("* A higher Confidence Variance indicates the model is highly unstable in how certain it is across languages.")

if __name__ == "__main__":
    # Initialize and run the pipeline
    # Note: Keep num_samples low initially to avoid hitting API rate limits
    pipeline = GenerativeCrossLingualPipeline(model_name="tiny-aya-global")
    final_results = pipeline.run_pipeline(num_samples=100)