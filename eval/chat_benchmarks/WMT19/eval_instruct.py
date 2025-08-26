import logging
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy
except ImportError:
    numpy = None

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from lm_eval.api.instance import Instance
    from lm_eval.api.model import LM
except ImportError:
    Instance = None
    LM = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    BaseBenchmark = None

try:
    import sacrebleu
except ImportError:
    sacrebleu = None


class WMT19Benchmark(BaseBenchmark if BaseBenchmark is not None else object):
    """
    WMT19 Machine Translation Benchmark for evaluating translation capabilities of LLMs.

    This benchmark evaluates models on machine translation tasks using the WMT19 dataset.
    It supports multiple language pairs and uses BLEU scores for evaluation.

    Link: https://huggingface.co/datasets/wmt/wmt19
    """

    def __init__(
        self,
        language_pair: str = "de-en",
        max_examples: Optional[int] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 1024,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize WMT19 benchmark.

        Args:
            language_pair: Language pair to evaluate (e.g., "de-en", "fr-de", "zh-en")
            max_examples: Maximum number of examples to evaluate (None for all)
            debug: If set, only evaluate on 10 examples
            seed: Random seed for reproducibility
            max_tokens: Maximum number of tokens to generate
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
        """
        # Check for required dependencies
        missing_deps = []
        if load_dataset is None:
            missing_deps.append("datasets")
        if sacrebleu is None:
            missing_deps.append("sacrebleu")
        if numpy is None:
            missing_deps.append("numpy")
        if BaseBenchmark is None:
            missing_deps.append("lm_eval")

        if missing_deps:
            raise ImportError(
                f"Missing required dependencies for WMT19 evaluation: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )

        super().__init__(logger=logger, system_instruction=system_instruction)

        self.language_pair = language_pair
        self.max_examples = max_examples
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens

        # Parse language pair
        self.source_lang, self.target_lang = language_pair.split("-")

        # Available language pairs in WMT19
        self.available_pairs = [
            "cs-en", "de-en", "fi-en", "fr-de", "gu-en",
            "kk-en", "lt-en", "ru-en", "zh-en"
        ]

        if language_pair not in self.available_pairs:
            raise ValueError(f"Language pair {language_pair} not available. Choose from: {self.available_pairs}")

        # Load dataset
        self.dataset = load_dataset("wmt/wmt19", language_pair)

        # Language name mapping for prompts
        self.lang_names = {
            "cs": "Czech", "de": "German", "en": "English", "fi": "Finnish",
            "fr": "French", "gu": "Gujarati", "kk": "Kazakh",
            "lt": "Lithuanian", "ru": "Russian", "zh": "Chinese"
        }

    def _create_translation_prompt(self, source_text: str) -> str:
        """Create a translation prompt for the given source text."""
        source_lang_name = self.lang_names[self.source_lang]
        target_lang_name = self.lang_names[self.target_lang]

        prompt = f"Translate the following {source_lang_name} text to {target_lang_name}:\n\n"
        prompt += f"{source_text}\n\n"
        prompt += f"Translation:"

        return prompt

    def generate_responses(self, model) -> Dict[str, Any]:
        """
        Generate translation completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and evaluation data,
            or None for non-primary ranks
        """
        # Use validation split for evaluation
        examples = list(self.dataset["validation"])

        if self.debug:
            examples = examples[:10]
        elif self.max_examples:
            examples = examples[:self.max_examples]

        self.logger.info(f"Evaluating {len(examples)} examples for {self.language_pair} translation")

        # Prepare instances for model
        all_instances = []
        for idx, example in enumerate(examples):
            translation_dict = example["translation"]
            source_text = translation_dict[self.source_lang]
            target_text = translation_dict[self.target_lang]

            prompt = self._create_translation_prompt(source_text)

            messages = [
                {"role": "user", "content": prompt},
            ]

            templated_messages = self._prepare_messages(messages, model)

            instance = Instance(
                "generate_until",
                example,
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "seed": self.seed,
                    },
                ),
                idx,
            )

            # Add metadata
            instance.metadata = {
                "source_text": source_text,
                "reference_translation": target_text,
                "language_pair": self.language_pair,
            }

            all_instances.append(instance)

        # Generate model responses
        self.logger.info(f"Generating responses for WMT19 {self.language_pair}...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Process outputs
        processed_examples = []
        for example, output in zip(examples, outputs):
            translation_dict = example["translation"]
            source_text = translation_dict[self.source_lang]
            reference_translation = translation_dict[self.target_lang]

            # Extract translation from model output
            model_translation = self._extract_translation(output)

            processed_examples.append({
                "source_text": source_text,
                "reference_translation": reference_translation,
                "model_translation": model_translation,
                "model_output": output,
                "language_pair": self.language_pair,
            })

        return {"examples": processed_examples, "language_pair": self.language_pair}

    def _extract_translation(self, output: str) -> str:
        """
        Extract the translation from model output.

        Args:
            output: Raw model output

        Returns:
            Extracted translation text
        """
        # Handle different output formats
        if isinstance(output, str):
            text = output
        elif hasattr(output, 'outputs') and output.outputs:
            text = output.outputs[0].text
        elif hasattr(output, 'text'):
            text = output.text
        else:
            text = str(output)

        # Clean up the translation
        # Look for text after "Translation:" or similar patterns
        patterns = [
            r"Translation:\s*(.*?)(?:\n|$)",
            r"Answer:\s*(.*?)(?:\n|$)",
            r"Output:\s*(.*?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                translation = match.group(1).strip()
                if translation:
                    return translation

        # If no pattern matches, return the entire text cleaned up
        lines = text.strip().split('\n')
        # Take the first non-empty line that doesn't look like a prompt
        for line in lines:
            line = line.strip()
            if line and not any(keyword in line.lower() for keyword in ['translate', 'translation:', 'answer:', 'output:']):
                return line

        # Fallback: return the entire text
        return text.strip()

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate the generated translations using BLEU scores."""

        # Handle None result from non-primary ranks
        if results is None:
            return None

        examples = results["examples"]
        language_pair = results["language_pair"]

        if not examples:
            return {"error": "No examples to evaluate"}

        # Prepare references and hypotheses for BLEU calculation
        references = []
        hypotheses = []

        for example in examples:
            ref = example["reference_translation"].strip()
            hyp = example["model_translation"].strip()

            if ref and hyp:  # Only include non-empty translations
                references.append([ref])  # sacrebleu expects list of references
                hypotheses.append(hyp)

        if not references or not hypotheses:
            return {"error": "No valid translation pairs found"}

        # Calculate BLEU score using sacrebleu
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, references)
            bleu_score = bleu.score
        except Exception as e:
            self.logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = 0.0

        # Calculate additional metrics
        num_examples = len(examples)
        num_valid_translations = len(references)

        # Calculate average translation length
        if references:
            avg_ref_length = sum(len(ref[0].split()) for ref in references) / len(references)
        else:
            avg_ref_length = 0

        if hypotheses:
            avg_hyp_length = sum(len(hyp.split()) for hyp in hypotheses) / len(hypotheses)
        else:
            avg_hyp_length = 0

        results_dict = {
            "bleu_score": bleu_score,
            "num_examples": num_examples,
            "num_valid_translations": num_valid_translations,
            "avg_reference_length": avg_ref_length,
            "avg_hypothesis_length": avg_hyp_length,
            "language_pair": language_pair,
            "coverage": num_valid_translations / num_examples if num_examples > 0 else 0.0,
        }

        self.logger.info(f"WMT19 {language_pair} Results:")
        self.logger.info(f"  BLEU Score: {bleu_score:.2f}")
        self.logger.info(f"  Valid Translations: {num_valid_translations}/{num_examples}")
        self.logger.info(f"  Coverage: {results_dict['coverage']:.2%}")

        return results_dict