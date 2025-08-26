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


class WMT24Benchmark(BaseBenchmark if BaseBenchmark is not None else object):
    """
    WMT24 Machine Translation Benchmark for evaluating translation capabilities of LLMs.

    This benchmark evaluates models on machine translation tasks using the WMT24++ dataset.
    It supports multiple language pairs and uses BLEU scores for evaluation.

    Link: https://huggingface.co/datasets/google/wmt24pp
    """

    def __init__(
        self,
        language_pair: str = "en-ko_KR",
        max_examples: Optional[int] = None,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        max_tokens: int = 1024,
        filter_bad_source: bool = True,
        domain_filter: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
    ):
        """
        Initialize WMT24 benchmark.

        Args:
            language_pair: Language pair to evaluate (e.g., "en-de_DE", "en-fr_FR", "en-zh_CN")
            max_examples: Maximum number of examples to evaluate (None for all)
            debug: If set, only evaluate on 10 examples
            seed: Random seed for reproducibility
            max_tokens: Maximum number of tokens to generate
            filter_bad_source: If True, filter out examples marked as bad source
            domain_filter: Optional domain filter ("news", "social", "canary", "speech", "literary")
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
                f"Missing required dependencies for WMT24 evaluation: {', '.join(missing_deps)}. "
                f"Install with: pip install {' '.join(missing_deps)}"
            )

        super().__init__(logger=logger, system_instruction=system_instruction)

        self.language_pair = language_pair
        self.max_examples = max_examples
        self.debug = debug
        self.seed = seed
        self.max_new_tokens = max_tokens
        self.filter_bad_source = filter_bad_source
        self.domain_filter = domain_filter

        # Parse language pair (format: en-de_DE)
        parts = language_pair.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid language pair format: {language_pair}. Expected format: en-de_DE")

        self.source_lang = parts[0]
        self.target_lang_code = parts[1]

        # Extract base language from locale code (e.g., de_DE -> de)
        self.target_lang = self.target_lang_code.split("_")[0]

        # Available language pairs in WMT24++
        self.available_pairs = [
            "en-ar_EG", "en-ar_SA", "en-bg_BG", "en-bn_IN", "en-ca_ES", "en-cs_CZ",
            "en-da_DK", "en-de_DE", "en-el_GR", "en-es_MX", "en-et_EE", "en-fa_IR",
            "en-fi_FI", "en-fil_PH", "en-fr_CA", "en-fr_FR", "en-gu_IN", "en-he_IL",
            "en-hi_IN", "en-hr_HR", "en-hu_HU", "en-id_ID", "en-is_IS", "en-it_IT",
            "en-ja_JP", "en-kn_IN", "en-ko_KR", "en-lt_LT", "en-lv_LV", "en-ml_IN",
            "en-mr_IN", "en-nl_NL", "en-no_NO", "en-pa_IN", "en-pl_PL", "en-pt_BR",
            "en-pt_PT", "en-ro_RO", "en-ru_RU", "en-sk_SK", "en-sl_SI", "en-sr_RS",
            "en-sv_SE", "en-sw_KE", "en-sw_TZ", "en-ta_IN", "en-te_IN", "en-th_TH",
            "en-tr_TR", "en-uk_UA", "en-ur_PK", "en-vi_VN", "en-zh_CN", "en-zh_TW",
            "en-zu_ZA"
        ]

        if language_pair not in self.available_pairs:
            raise ValueError(f"Language pair {language_pair} not available. Choose from: {self.available_pairs}")

        # Language name mapping for prompts (using the provided constants from the dataset)
        self.lang_names = {
            "en": "English",
            "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali", "ca": "Catalan", "cs": "Czech",
            "da": "Danish", "de": "German", "el": "Greek", "es": "Spanish", "et": "Estonian",
            "fa": "Farsi", "fi": "Finnish", "fil": "Filipino", "fr": "French", "gu": "Gujarati",
            "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
            "is": "Icelandic", "it": "Italian", "ja": "Japanese", "kn": "Kannada", "ko": "Korean",
            "lt": "Lithuanian", "lv": "Latvian", "ml": "Malayalam", "mr": "Marathi", "nl": "Dutch",
            "no": "Norwegian", "pa": "Punjabi", "pl": "Polish", "pt": "Portuguese", "ro": "Romanian",
            "ru": "Russian", "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "sv": "Swedish",
            "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "th": "Thai", "tr": "Turkish",
            "uk": "Ukrainian", "ur": "Urdu", "vi": "Vietnamese", "zh": "Mandarin", "zu": "Zulu"
        }

        # Load dataset
        try:
            self.dataset = load_dataset("google/wmt24pp", language_pair)
        except Exception as e:
            raise RuntimeError(f"Failed to load WMT24++ dataset for {language_pair}: {e}")

    def _create_translation_prompt(self, source_text: str) -> str:
        """Create a translation prompt for the given source text."""
        source_lang_name = self.lang_names.get(self.source_lang, self.source_lang)
        target_lang_name = self.lang_names.get(self.target_lang, self.target_lang)

        prompt = f"Translate the following {source_lang_name} text to {target_lang_name}:\n\n"
        prompt += f"{source_text}\n\n"
        prompt += f"Translation:"

        return prompt

    def _filter_examples(self, examples: List[Dict]) -> List[Dict]:
        """Filter examples based on quality and domain criteria."""
        filtered = []

        for example in examples:
            # Filter out bad source examples if requested
            if self.filter_bad_source and example.get("is_bad_source", False):
                continue

            # Filter by domain if specified
            if self.domain_filter and example.get("domain") != self.domain_filter:
                continue

            # Ensure we have valid source and target text
            if not example.get("source") or not example.get("target"):
                continue

            filtered.append(example)

        return filtered

    def generate_responses(self, model) -> Dict[str, Any]:
        """
        Generate translation completions using the provided model.

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses and evaluation data,
            or None for non-primary ranks
        """
        # Use train split for evaluation (WMT24++ only has train split)
        examples = list(self.dataset["train"])

        # Apply filtering
        examples = self._filter_examples(examples)

        if self.debug:
            examples = examples[:10]
        elif self.max_examples:
            examples = examples[:self.max_examples]

        self.logger.info(f"Evaluating {len(examples)} examples for {self.language_pair} translation")
        if self.filter_bad_source:
            self.logger.info("Filtering out bad source examples")
        if self.domain_filter:
            self.logger.info(f"Filtering for domain: {self.domain_filter}")

        # Prepare instances for model
        all_instances = []
        for idx, example in enumerate(examples):
            source_text = example["source"]
            target_text = example["target"]  # Use post-edited target as recommended

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
                "original_target": example.get("original_target", target_text),
                "language_pair": self.language_pair,
                "domain": example.get("domain", "unknown"),
                "document_id": example.get("document_id", "unknown"),
                "segment_id": example.get("segment_id", idx),
            }

            all_instances.append(instance)

        # Generate model responses
        self.logger.info(f"Generating responses for WMT24 {self.language_pair}...")
        outputs = self.compute(model, all_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Process outputs
        processed_examples = []
        for example, output in zip(examples, outputs):
            source_text = example["source"]
            reference_translation = example["target"]  # Use post-edited target
            original_target = example.get("original_target", reference_translation)

            # Extract translation from model output
            model_translation = self._extract_translation(output)

            processed_examples.append({
                "source_text": source_text,
                "reference_translation": reference_translation,
                "original_target": original_target,
                "model_translation": model_translation,
                "model_output": output,
                "language_pair": self.language_pair,
                "domain": example.get("domain", "unknown"),
                "document_id": example.get("document_id", "unknown"),
                "segment_id": example.get("segment_id", 0),
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

        # Track domain-specific performance
        domain_stats = {}

        for example in examples:
            ref = example["reference_translation"].strip()
            hyp = example["model_translation"].strip()
            domain = example.get("domain", "unknown")

            if ref and hyp:  # Only include non-empty translations
                references.append([ref])  # sacrebleu expects list of references
                hypotheses.append(hyp)

                # Track domain statistics
                if domain not in domain_stats:
                    domain_stats[domain] = {"count": 0, "refs": [], "hyps": []}
                domain_stats[domain]["count"] += 1
                domain_stats[domain]["refs"].append([ref])
                domain_stats[domain]["hyps"].append(hyp)

        if not references or not hypotheses:
            return {"error": "No valid translation pairs found"}

        # Calculate overall BLEU score using sacrebleu
        try:
            bleu = sacrebleu.corpus_bleu(hypotheses, references)
            bleu_score = bleu.score
        except Exception as e:
            self.logger.error(f"Error calculating BLEU score: {e}")
            bleu_score = 0.0

        # Calculate domain-specific BLEU scores
        domain_bleu_scores = {}
        for domain, stats in domain_stats.items():
            if stats["refs"] and stats["hyps"]:
                try:
                    domain_bleu = sacrebleu.corpus_bleu(stats["hyps"], stats["refs"])
                    domain_bleu_scores[f"bleu_{domain}"] = domain_bleu.score
                except Exception as e:
                    self.logger.warning(f"Error calculating BLEU for domain {domain}: {e}")
                    domain_bleu_scores[f"bleu_{domain}"] = 0.0

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

        # Add domain-specific scores
        results_dict.update(domain_bleu_scores)

        # Add domain counts
        for domain, stats in domain_stats.items():
            results_dict[f"count_{domain}"] = stats["count"]

        self.logger.info(f"WMT24 {language_pair} Results:")
        self.logger.info(f"  BLEU Score: {bleu_score:.2f}")
        self.logger.info(f"  Valid Translations: {num_valid_translations}/{num_examples}")
        self.logger.info(f"  Coverage: {results_dict['coverage']:.2%}")

        # Log domain-specific results
        for domain, score in domain_bleu_scores.items():
            domain_name = domain.replace("bleu_", "")
            count = domain_stats[domain_name]["count"]
            self.logger.info(f"  {domain_name.title()} BLEU: {score:.2f} ({count} examples)")

        return results_dict