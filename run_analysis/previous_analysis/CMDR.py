import cohere
import os
import json
import csv
from itertools import combinations
from pathlib import Path
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
    def __init__(self, model_name="tiny-aya-global"):
        """Initializes the pipeline with the specified model."""
        self.model_name = model_name
        print(f"Initialized pipeline for API model: {self.model_name}")

    def load_multilingual_data(self, languages, num_samples=300):
        """Loads aligned XNLI premise/hypothesis text for each language."""
        multilingual_data = {}
        print(f"Loading XNLI dataset for languages: {', '.join(languages)}")
        for lang in languages:
            lang_data = load_dataset("xnli", lang, split="test")
            multilingual_data[lang] = {
                "premise": lang_data["premise"][:num_samples],
                "hypothesis": lang_data["hypothesis"][:num_samples],
            }
        return multilingual_data

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
            elif hasattr(token, "logprob"):
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

    def _build_pairs(self, languages, english_only_pairs):
        if english_only_pairs:
            return [("en", lang) for lang in languages if lang != "en"]
        return list(combinations(languages, 2))

    def _safe_extract_label(self, text):
        """Returns label from JSON output or raw fallback text."""
        try:
            return json.loads(text).get("label", "").strip().lower()
        except Exception:
            return text.strip().lower()

    def _write_csv(self, path, fieldnames, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def run_pipeline(
        self,
        num_samples=300,
        languages=None,
        english_only_pairs=False,
        output_dir="outputs",
    ):
        """
        Runs multilingual inference and computes pairwise disagreement metrics.
        For efficiency, each (sample, language) is queried once, then reused for all pairs.
        """
        if languages is None:
            languages = ["en", "hi", "zh", "fr", "de", "ar", "es", "th", "tr", "vi", "sw"]
        if "en" not in languages:
            raise ValueError("`languages` must include 'en' for English comparison.")

        multilingual_data = self.load_multilingual_data(languages, num_samples)
        language_pairs = self._build_pairs(languages, english_only_pairs)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions = {}
        prediction_rows = []

        prompt_template = """Analyze the premise and hypothesis. 
Determine if the relationship is 'entailment', 'neutral', or 'contradiction'. 
You MUST output a valid JSON object with a single key 'label'.

Premise: {premise}
Hypothesis: {hypothesis}"""

        print("Running multilingual inference...")
        for sample_id in tqdm(range(num_samples), desc="Querying model"):
            for lang in languages:
                premise = multilingual_data[lang]["premise"][sample_id]
                hypothesis = multilingual_data[lang]["hypothesis"][sample_id]
                query = prompt_template.format(premise=premise, hypothesis=hypothesis)
                try:
                    resp = query_model(query, model=self.model_name, logprobs=True)
                    text = get_text_from_response(resp)
                    label = self._safe_extract_label(text)
                    prob = self.calculate_sequence_probability(get_logprobs_from_response(resp))
                    predictions[(sample_id, lang)] = {"label": label, "prob": float(prob)}
                    prediction_rows.append(
                        {
                            "sample_id": sample_id,
                            "language": lang,
                            "label": label,
                            "probability": float(prob),
                        }
                    )
                except Exception as e:
                    print(f"Error processing sample {sample_id}, language {lang}: {e}")
                    predictions[(sample_id, lang)] = None

        sample_metric_rows = []
        pairwise_label = {}
        pairwise_conf = {}
        print("Computing pairwise disagreement metrics...")
        for sample_id in range(num_samples):
            for lang_a, lang_b in language_pairs:
                pred_a = predictions.get((sample_id, lang_a))
                pred_b = predictions.get((sample_id, lang_b))
                if not pred_a or not pred_b:
                    continue

                label_dist = self.calculate_label_disagreement(pred_a["label"], pred_b["label"])
                conf_dist = self.calculate_confidence_distance(pred_a["prob"], pred_b["prob"])
                pair_name = f"{lang_a}-{lang_b}"

                pairwise_label.setdefault(pair_name, []).append(label_dist)
                pairwise_conf.setdefault(pair_name, []).append(conf_dist)
                sample_metric_rows.append(
                    {
                        "sample_id": sample_id,
                        "lang_a": lang_a,
                        "lang_b": lang_b,
                        "pair": pair_name,
                        "label_a": pred_a["label"],
                        "label_b": pred_b["label"],
                        "prob_a": pred_a["prob"],
                        "prob_b": pred_b["prob"],
                        "label_disagreement": float(label_dist),
                        "confidence_distance": float(conf_dist),
                    }
                )

        label_all = [v for values in pairwise_label.values() for v in values]
        conf_all = [v for values in pairwise_conf.values() for v in values]
        results = {
            "Label Disagreement": {
                "Mean": float(np.mean(label_all)) if label_all else 0.0,
                "Variance": float(np.var(label_all)) if label_all else 0.0,
            },
            "Confidence Disagreement": {
                "Mean": float(np.mean(conf_all)) if conf_all else 0.0,
                "Variance": float(np.var(conf_all)) if conf_all else 0.0,
            },
            "Pairwise": {},
        }

        for pair_name in sorted(pairwise_label.keys()):
            labels = pairwise_label[pair_name]
            confs = pairwise_conf[pair_name]
            results["Pairwise"][pair_name] = {
                "n": len(labels),
                "label_disagreement_mean": float(np.mean(labels)) if labels else 0.0,
                "label_disagreement_variance": float(np.var(labels)) if labels else 0.0,
                "confidence_distance_mean": float(np.mean(confs)) if confs else 0.0,
                "confidence_distance_variance": float(np.var(confs)) if confs else 0.0,
            }

        self._write_csv(
            output_path / "cmdr_predictions.csv",
            ["sample_id", "language", "label", "probability"],
            prediction_rows,
        )
        self._write_csv(
            output_path / "cmdr_sample_metrics.csv",
            [
                "sample_id",
                "lang_a",
                "lang_b",
                "pair",
                "label_a",
                "label_b",
                "prob_a",
                "prob_b",
                "label_disagreement",
                "confidence_distance",
            ],
            sample_metric_rows,
        )
        with open(output_path / "cmdr_summary.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"Wrote metrics to: {output_path.resolve()}")
        self.print_results(results)
        return results

    def print_results(self, results):
        """Formats and prints the interpretation of the results."""
        print("\n" + "="*50)
        print("GENERATIVE CROSS-LINGUAL DISAGREEMENT RESULTS")
        print("="*50)
        for metric, stats in results.items():
            if metric == "Pairwise":
                continue
            print(f"{metric}:")
            print(f"  Mean:     {stats['Mean']:.4f}")
            print(f"  Variance: {stats['Variance']:.4f}\n")
            
        print("Interpretation Guide:")
        print("* A Label Disagreement mean of 0.30 means the model changes its prediction 30% of the time based purely on language.")
        print("* A higher Confidence Variance indicates the model is highly unstable in how certain it is across languages.")
        print("* Pairwise metrics are saved in cmdr_summary.json for per-language analysis.")

if __name__ == "__main__":
    # Initialize and run the pipeline
    # Increase num_samples for comprehensive plots (API cost/time will increase).
    pipeline = GenerativeCrossLingualPipeline(model_name="tiny-aya-global")
    final_results = pipeline.run_pipeline(
        num_samples=300,
        languages=["en", "hi", "zh", "fr", "de", "ar", "es", "th", "tr", "vi", "sw"],
        english_only_pairs=True,
        output_dir="data",
    )