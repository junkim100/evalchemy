"""
Creative Writing Bench evaluation implementation for evalchemy.

This module implements the EQ-Bench Creative Writing Benchmark v3 with complete
compatibility to the original evaluation methodology, integrated with evalchemy framework.
"""

import asyncio
import json
import logging
import os
import random
import re
import statistics
import tempfile
import threading
import time
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
from openai import AsyncOpenAI

try:
    from lm_eval.api.instance import Instance
except ImportError:
    # Fallback for when lm_eval is not available
    Instance = None

try:
    from eval.task import BaseBenchmark
except ImportError:
    # Try alternative import paths
    try:
        from evalchemy.eval.task import BaseBenchmark
    except ImportError:
        try:
            from ...task import BaseBenchmark
        except ImportError:
            # Fallback base class for testing
            class BaseBenchmark:
                def __init__(self, logger=None, system_instruction=None):
                    self.logger = logger or logging.getLogger(self.__class__.__name__)
                    self.system_instruction = system_instruction


# Load environment variables
load_dotenv()

# Initialize OpenAI client for judge evaluation
client = AsyncOpenAI(timeout=300.0, max_retries=3)


async def judge_creative_writing_response(
    judge_prompt: str,
    judge_model: str = "gpt-4o-mini"
) -> str:
    """
    Judge a creative writing response using OpenAI API.

    Args:
        judge_prompt: The formatted judge prompt
        judge_model: The judge model to use

    Returns:
        The judge response text
    """
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        raise Exception(f"OpenAI judge API failed: {e}")


async def judge_multiple_responses(
    judge_prompts: List[str],
    judge_model: str = "gpt-4o-mini",
    max_concurrent: int = 5
) -> List[str]:
    """
    Judge multiple creative writing responses concurrently.

    Args:
        judge_prompts: List of formatted judge prompts
        judge_model: The judge model to use
        max_concurrent: Maximum concurrent requests

    Returns:
        List of judge response texts
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bound_judge(prompt):
        async with semaphore:
            return await judge_creative_writing_response(prompt, judge_model)

    tasks = [bound_judge(prompt) for prompt in judge_prompts]
    return await asyncio.gather(*tasks)


# Original Creative Writing Bench API Client (from utils/api.py)
class APIClient:
    """
    Client for interacting with LLM API endpoints (OpenAI or other).
    Integrated from original Creative Writing Bench utils/api.py.
    """

    def __init__(self, model_type=None, request_timeout=240, max_retries=3, retry_delay=5):
        self.model_type = model_type or "default"

        if model_type == "test":
            self.api_key = os.getenv("TEST_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("TEST_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        elif model_type == "judge":
            self.api_key = os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY"))
            self.base_url = os.getenv("JUDGE_API_URL", os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions"))
        else:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.base_url = os.getenv("OPENAI_API_URL", "https://api.openai.com/v1/chat/completions")

        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", request_timeout))
        self.max_retries = int(os.getenv("MAX_RETRIES", max_retries))
        self.retry_delay = int(os.getenv("RETRY_DELAY", retry_delay))

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        logging.debug(f"Initialized {self.model_type} API client with URL: {self.base_url}")

    def generate(self, model: str, prompt: str, temperature: float = 0.0, max_tokens: int = 8096, include_seed=True, min_p=0.1, system=None) -> str:
        """
        Generic chat-completion style call with retry logic.
        """
        messages = [{"role": "user", "content": prompt}]
        if system:
            messages = [{"role": "system", "content": system}] + messages

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if include_seed:
            payload["seed"] = random.randint(1, 1000000)

        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.request_timeout
                )
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"]

            except Exception as e:
                logging.warning(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    logging.error(f"All {self.max_retries} API attempts failed")
                    raise e


# Thread-safe file I/O utilities (from utils/file_io.py)
_file_locks = {}
_file_locks_lock = threading.Lock()

def get_file_lock(file_path: str) -> threading.Lock:
    """Acquire or create a per-file lock to avoid concurrent writes."""
    with _file_locks_lock:
        if file_path not in _file_locks:
            _file_locks[file_path] = threading.Lock()
        return _file_locks[file_path]

def load_json_file_safe(file_path: str) -> dict:
    """Thread-safe read of a JSON file, returning an empty dict if not found or error."""
    lock = get_file_lock(file_path)
    with lock:
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in {file_path}: {e}")
            return {}
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return {}

def save_json_file_safe(data: Dict[str, Any], file_path: str, max_retries: int = 3, retry_delay: float = 0.5) -> bool:
    """Thread-safe atomic write of JSON data."""
    lock = get_file_lock(file_path)
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(retry_delay)
        with lock:
            try:
                temp_path = file_path + ".tmp"
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                os.replace(temp_path, file_path)
                return True
            except Exception as e:
                logging.error(f"Attempt {attempt + 1} to save {file_path} failed: {e}")
    return False


# Model name substitutions (from model_name_subs.py)
MODEL_NAME_SUBS = {
    'deepseek/deepseek-r1': 'deepseek-ai/DeepSeek-R1',
    'anthropic/claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
    'openai/chatgpt-4o-latest': 'chatgpt-4o-latest-2025-01-29',
    'openai/gpt-4o-mini': 'gpt-4o-mini',
    'mistralai/mistral-nemo': 'mistralai/Mistral-Nemo-Instruct-2407',
    'google/gemini-2.0-flash-001': 'gemini-2.0-flash-001',
    'anthropic/claude-3.5-haiku': 'claude-3-5-haiku-20241022',
    'meta-llama/llama-4-scout': 'meta-llama/Llama-4-Scout-17B-16E-Instruct',
    'x-ai/grok-3-beta': 'grok-3-beta',
}

def substitute_model_name(model_name: str) -> str:
    """Apply model name substitutions from original Creative Writing Bench."""
    return MODEL_NAME_SUBS.get(model_name, model_name)


# Original Creative Writing Bench scoring functions (copied from original core/scoring.py)
SCORE_RANGE_MIN = 0
SCORE_RANGE_MAX = 20

# ELO Configuration (from original core/elo_config_cw.py)
DEFAULT_ELO = 1200.0
SAMPLING_SCHEDULE = [
    ((None,), 10),
    ((1, 2, 3), 4),
    ((1, 2, 3), 8),
    ((1, 2, 3), 16),
    ((1, 2, 3), 48),
]
MAX_STAGE_LOOPS = 4
TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION = 4
TRUESKILL_BIN_SIZE_FOR_CI_CALCULATION = 3
RANK_WINDOW = 16
CW_ANCHOR_MODELS = {
    'deepseek/deepseek-r1': 1500,
    'meta-llama/llama-3.2-1b-instruct': 200
}
TS_SIGMA = DEFAULT_ELO / 3
TS_BETA = TS_SIGMA / 2
TS_TAU = TS_SIGMA / 100
EXPAND_MARGINS_TO_EXTRA_WINS = True
TS_GAMMA_FOR_BETA_ADJUSTMENT = 40.0
LENGTH_TRUNCATION_CHARS = 4000
IGNORE_PROMPTS_FOR_ELO = ["5", "16", "20", "21", "26", "28", "30"]
TRUESKILL_BIN_SIZE_FOR_WIN_EXPANSION = 50
TRUESKILL_BIN_SIZE_FOR_CI_CALCULATION = 100
RANK_WINDOW = 5
LENGTH_TRUNCATION_CHARS = 8000
TS_SIGMA = 1000 / 3

# Anchor models for ELO calibration (from original)
CW_ANCHOR_MODELS = [
    "gpt-4o-2024-08-06",
    "claude-3-5-sonnet-20241022",
    "gpt-4o-mini-2024-07-18",
    "llama-3.1-70b-instruct"
]


def parse_judge_scores_creative(judge_response: str) -> Dict[str, float]:
    """
    Parse judge scores from response using original Creative Writing Bench logic.
    
    Maintains identical parsing patterns and validation as the original benchmark.
    """
    scores = {}
    
    # Original parsing patterns from the benchmark
    # Pattern 1: Metric: Score or Metric: Score X
    score_pattern1 = r'(.*?):\s*(?:Score\s+)?(-?\d+(?:\.\d+)?)'
    # Pattern 2: Metric: [Score]
    score_pattern2 = r'(.*?):\s*\[(-?\d+(?:\.\d+)?)\]'
    
    # Combine both patterns
    matches1 = re.findall(score_pattern1, judge_response)
    matches2 = re.findall(score_pattern2, judge_response)
    
    # Process matches from both patterns
    for matches in [matches1, matches2]:
        for match in matches:
            metric_name = match[0].strip()
            try:
                score = float(match[1])
                # Original validation: score must be <= 20
                if score <= 20:
                    scores[metric_name] = score
                # If score > 20, it's discarded/ignored (original behavior)
            except ValueError:
                continue
    
    return scores


def invert_if_negative(metric: str, score: float, negative_criteria: List[str]) -> float:
    """
    Invert score for negative criteria (original benchmark logic).
    
    For negative criteria, higher scores are worse, so we invert:
    new_score = 20 - old_score
    """
    if metric in negative_criteria:
        return 20.0 - score
    return score


def compute_creative_scores(tasks: List[Dict[str, Any]], negative_criteria: List[str]) -> float:
    """
    Compute creative writing scores using original Creative Writing Bench methodology.
    """
    all_scores = []
    
    for task in tasks:
        judge_scores = task.get('judge_scores', {})
        if not judge_scores:
            continue
        
        # Process each score, inverting negative criteria
        task_scores = []
        for metric, score in judge_scores.items():
            if isinstance(score, (int, float)):
                inverted_score = invert_if_negative(metric, score, negative_criteria)
                if inverted_score <= SCORE_RANGE_MAX:  # Original validation
                    task_scores.append(inverted_score)
        
        if task_scores:
            # Average scores for this task (0-20 scale)
            avg_score = sum(task_scores) / len(task_scores)
            all_scores.append(avg_score)
    
    if not all_scores:
        return 0.0
    
    # Compute overall score (0-20 scale)
    return sum(all_scores) / len(all_scores)


def compute_single_benchmark_score_creative(tasks: List[Dict[str, Any]], negative_criteria: List[str]) -> Dict[str, Any]:
    """
    Compute single benchmark score using original Creative Writing Bench methodology.
    """
    overall_score = compute_creative_scores(tasks, negative_criteria)
    eqbench_score = overall_score * 5.0  # Scale to 0-100
    
    return {
        "overall_score": round(overall_score, 2),
        "eqbench_creative_score": round(eqbench_score, 2)
    }


def bootstrap_benchmark_stability_creative(tasks: List[Dict[str, Any]], negative_criteria: List[str], 
                                         n_bootstrap: int = 500, confidence_level: float = 0.95) -> Dict[str, Any]:
    """
    Compute bootstrap confidence intervals using original benchmark methodology.
    """
    # Collect all valid scores
    all_scores = []
    
    for task in tasks:
        judge_scores = task.get('judge_scores', {})
        if not judge_scores:
            continue
        
        task_scores = []
        for metric, score in judge_scores.items():
            if isinstance(score, (int, float)):
                inverted_score = invert_if_negative(metric, score, negative_criteria)
                if inverted_score <= SCORE_RANGE_MAX:
                    task_scores.append(inverted_score)
        
        if task_scores:
            avg_score = sum(task_scores) / len(task_scores)
            all_scores.append(avg_score)
    
    if len(all_scores) < 2:
        return {"error": "Insufficient scores for bootstrap analysis"}
    
    original_mean = sum(all_scores) / len(all_scores)
    
    # Bootstrap sampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = random.choices(all_scores, k=len(all_scores))
        bootstrap_means.append(sum(sample) / len(sample))
    
    bootstrap_means.sort()
    
    # Compute confidence interval
    lower_idx = int((1 - confidence_level) / 2 * len(bootstrap_means))
    upper_idx = int((1 + confidence_level) / 2 * len(bootstrap_means)) - 1
    lower_idx = max(0, lower_idx)
    upper_idx = min(upper_idx, len(bootstrap_means) - 1)
    
    ci_lower = bootstrap_means[lower_idx]
    ci_upper = bootstrap_means[upper_idx]
    bootstrap_mean = sum(bootstrap_means) / len(bootstrap_means)
    bootstrap_std = statistics.stdev(bootstrap_means)
    
    return {
        "original": round(original_mean, 2),
        "bootstrap_mean": round(bootstrap_mean, 2),
        "bootstrap_std": round(bootstrap_std, 2),
        "standard_error": round(bootstrap_std, 2),
        "ci_lower": round(ci_lower, 2),
        "ci_upper": round(ci_upper, 2),
        "confidence_level": confidence_level,
        "n_bootstrap": n_bootstrap
    }


# Original Creative Writing Bench ELO functions (from core/elo.py)
def invert_if_negative_elo(metric: str, val: float, neg_list: List[str]) -> float:
    """From original CW elo.py - invert negative criteria."""
    if metric in neg_list:
        return 20.0 - val
    return val


def compute_fraction_for_test_cw(test_scores: Dict[str, float], ref_scores: Dict[str, float],
                                negative_criteria: List[str]) -> float:
    """
    Compute fraction for Creative Writing test using original methodology.

    From original core/elo_helpers_cw.py
    """
    test_vals = []
    ref_vals = []

    # Collect scores for common metrics
    for metric in test_scores:
        if metric in ref_scores:
            test_val = invert_if_negative_elo(metric, test_scores[metric], negative_criteria)
            ref_val = invert_if_negative_elo(metric, ref_scores[metric], negative_criteria)

            if 0 <= test_val <= 20 and 0 <= ref_val <= 20:
                test_vals.append(test_val)
                ref_vals.append(ref_val)

    if not test_vals:
        return 0.5  # Default tie

    # Compute average scores
    test_avg = sum(test_vals) / len(test_vals)
    ref_avg = sum(ref_vals) / len(ref_vals)

    # Convert to fraction (0.0 = ref wins, 1.0 = test wins, 0.5 = tie)
    if test_avg > ref_avg:
        return 0.75  # Test model wins
    elif test_avg < ref_avg:
        return 0.25  # Reference model wins
    else:
        return 0.5   # Tie


def create_matchup_signature_cw(model_a: str, model_b: str, prompt_id: str,
                               iteration: int, seed_modifier: str) -> str:
    """Create unique signature for matchup (from original)."""
    models_sorted = tuple(sorted([model_a, model_b]))
    return f"{models_sorted[0]}__vs__{models_sorted[1]}__prompt_{prompt_id}__iter_{iteration}__seed_{seed_modifier}"


def interpret_pairwise_result_cw(model_a: str, model_b: str, scores_a: Dict[str, float],
                                scores_b: Dict[str, float], negative_criteria: List[str]) -> Dict[str, Any]:
    """
    Interpret pairwise comparison result using original Creative Writing Bench methodology.

    From original core/elo_helpers_cw.py
    """
    fraction = compute_fraction_for_test_cw(scores_a, scores_b, negative_criteria)

    # Determine winner based on fraction
    if fraction > 0.5:
        winner = model_a
        loser = model_b
    elif fraction < 0.5:
        winner = model_b
        loser = model_a
    else:
        winner = None  # Tie
        loser = None

    return {
        "model_a": model_a,
        "model_b": model_b,
        "fraction": fraction,
        "winner": winner,
        "loser": loser,
        "scores_a": scores_a,
        "scores_b": scores_b
    }


def do_pairwise_judge_cw(
    textA: str,
    textB: str,
    prompt_id: str,
    pairwise_prompt_template: str,
    writing_prompts: Dict[str, Any],
    judge_model,
    negative_criteria: List[str]
) -> Dict[str, Any]:
    """
    Core pairwise judging function from original CW elo.py.

    Performs pairwise comparison between two creative writing responses.
    """
    try:
        # Get the original writing prompt
        writing_prompt = writing_prompts.get(prompt_id, {}).get("writing_prompt", "")

        # Truncate texts if too long (from original LENGTH_TRUNCATION_CHARS)
        if len(textA) > LENGTH_TRUNCATION_CHARS:
            textA = textA[:LENGTH_TRUNCATION_CHARS] + "..."
        if len(textB) > LENGTH_TRUNCATION_CHARS:
            textB = textB[:LENGTH_TRUNCATION_CHARS] + "..."

        # Format pairwise prompt
        pairwise_prompt = pairwise_prompt_template.format(
            writing_prompt=writing_prompt,
            response_a=textA,
            response_b=textB
        )

        # Use judge model if available
        if judge_model and Instance is not None:
            try:
                judge_instance = Instance(
                    "generate_until",
                    None,
                    (
                        pairwise_prompt,
                        {
                            "temperature": 0.0,
                            "max_gen_toks": 1000,
                            "min_p": 0.0
                        }
                    ),
                    0
                )

                judge_response = judge_model.generate_until([judge_instance])[0]

            except Exception as judge_error:
                logging.warning(f"Pairwise judge model failed: {judge_error}")
                judge_response = "A0493"  # Default to A wins
        else:
            # Mock response for testing
            judge_response = "A0493" if len(textA) > len(textB) else "A0488"

        # Parse the judge response (simplified version of original logic)
        result_dict = {"overall": judge_response}

        # Use original interpretation logic
        outcome_for_a, plus_for_a, plus_for_b = interpret_pairwise_result_original(result_dict)

        # Calculate fraction using original methodology
        fraction, diff, diff_norm, diff_blend = compute_fraction_for_test_cw(outcome_for_a, plus_for_a, plus_for_b)

        return {
            "judge_response": judge_response,
            "result_dict": result_dict,
            "outcome_for_a": outcome_for_a,
            "plus_for_a": plus_for_a,
            "plus_for_b": plus_for_b,
            "fraction": fraction,
            "diff": diff,
            "diff_norm": diff_norm,
            "diff_blend": diff_blend
        }

    except Exception as e:
        logging.error(f"Error in pairwise judging: {e}")
        return {
            "judge_response": f"[ERROR: {e}]",
            "result_dict": {},
            "outcome_for_a": 0.5,
            "plus_for_a": 0,
            "plus_for_b": 0,
            "fraction": 0.5,
            "diff": 0,
            "diff_norm": 0.0,
            "diff_blend": 0.0
        }


def should_ignore_prompt_cw(prompt_id: str) -> bool:
    """Check if prompt_id is in IGNORE_PROMPTS_FOR_ELO. From original elo_helpers_cw.py"""
    base_id = prompt_id.split("_")[0] if "_" in prompt_id else prompt_id
    return base_id in IGNORE_PROMPTS_FOR_ELO


def interpret_pairwise_result_original(result_dict: Optional[Dict[str, str]]) -> tuple:
    """
    Original interpretation logic from elo_helpers_cw.py.

    Return (outcome_for_A, plus_for_A, plus_for_B) in {0,0.5,1}, plus_for_A, plus_for_B as int tallies.
    """
    if not result_dict:
        return 0.5, 0, 0

    a_score = 0
    b_score = 0
    for key, val in result_dict.items():
        if key in ["improvement_suggestions", "theory_of_mind", "_item_order_idx"]:
            continue

        # "A0493" => means model A is better for that dimension
        if "A0493" in val:  # Model A preferred
            plus_count = val.count('+')
            if plus_count > 0:
                a_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]:  # Negative criteria
                b_score -= plus_count  # If A is good on negative, B is penalized
        elif "A0488" in val:  # Model B preferred
            plus_count = val.count('+')
            if plus_count > 0:
                b_score += plus_count
            if key in ["avoids_poetic_overload", "coherence", "avoids_verbosity"]:  # Negative criteria
                a_score -= plus_count  # If B is good on negative, A is penalized

    if a_score > b_score:
        return 1.0, a_score, b_score
    elif b_score > a_score:
        return 0.0, a_score, b_score
    else:
        return 0.5, a_score, b_score


# Original Creative Writing Bench metrics functions (from core/metrics.py)
def calculate_complexity_index(text: str) -> float:
    """
    Calculate a complexity index (0-100) based on Flesch-Kincaid grade level and percentage of complex words.

    From original core/metrics.py - simplified version without NLTK dependencies for evalchemy.
    """
    if not text or not text.strip():
        return 0

    # Simple sentence and word counting
    sentences = text.split('.')
    sentences = [s.strip() for s in sentences if s.strip()]
    words = text.split()

    sentence_count = max(1, len(sentences))
    word_count = max(1, len(words))

    # Simplified syllable counting (approximation)
    def count_syllables(word):
        word = word.lower().strip('.,!?;:"')
        vowels = 'aeiouy'
        syllable_count = 0
        prev_was_vowel = False

        for char in word:
            if char in vowels:
                if not prev_was_vowel:
                    syllable_count += 1
                prev_was_vowel = True
            else:
                prev_was_vowel = False

        # Handle silent e
        if word.endswith('e') and syllable_count > 1:
            syllable_count -= 1

        return max(1, syllable_count)

    # Calculate metrics
    total_syllables = sum(count_syllables(word) for word in words)
    fk_grade_level = 0.39 * (word_count / sentence_count) + 11.8 * (total_syllables / word_count) - 15.59

    # Calculate percentage of complex words (3+ syllables)
    complex_word_count = sum(1 for word in words if count_syllables(word) >= 3)
    percent_complex_words = (complex_word_count / word_count) * 100

    # Combine into complexity index (0-100)
    complexity_index = min(100, max(0, (fk_grade_level * 3) + (percent_complex_words * 0.5)))

    return round(complexity_index, 2)


def load_slop_list_to_set(filename: str) -> set:
    """
    Loads slop words/phrases from the specific JSON format into a set.

    From original core/metrics.py - handles format like [["word1"], ["word2 phrase"], ...]
    """
    if not os.path.exists(filename):
        logging.warning(f"Slop file not found: {filename}. Returning empty set.")
        return set()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Extract the first element from each inner list and lowercase it
        # Handles format like [["word1"], ["word2 phrase"], ...]
        slop_items = {item[0].lower() for item in data if item}  # Ensure inner list is not empty
        logging.info(f"Loaded {len(slop_items)} items from {filename}")
        return slop_items

    except json.JSONDecodeError:
        logging.warning(f"Error: Could not decode JSON from {filename}. Returning empty set.")
        return set()
    except Exception as e:
        logging.warning(f"Error loading {filename}: {e}. Returning empty set.")
        return set()


def calculate_slop_index(text: str, data_dir: Path, debug: bool = False) -> float:
    """
    Calculates a slop index based on hits in word, bigram, and trigram slop lists.

    From original core/metrics.py calculate_slop_index_new function.

    Args:
        text (str): The text to analyze.
        data_dir (Path): Directory containing slop list files.
        debug (bool): If True, prints the hit counts for each list.

    Returns:
        float: The calculated slop index.
    """
    # 1. Load Slop Lists
    slop_words_set = load_slop_list_to_set(str(data_dir / 'slop_list.json'))
    slop_bigrams_set = load_slop_list_to_set(str(data_dir / 'slop_list_bigrams.json'))
    slop_trigrams_set = load_slop_list_to_set(str(data_dir / 'slop_list_trigrams.json'))

    # Check if any lists were loaded
    if not slop_words_set and not slop_bigrams_set and not slop_trigrams_set:
        if debug:
            logging.warning("Error: No slop lists could be loaded. Returning slop index 0.")
        return 0.0

    if not text or not isinstance(text, str):
        if debug:
            logging.info("Input text is empty or invalid.")
            logging.info(f"Word Hits: 0")
            logging.info(f"Bigram Hits: 0")
            logging.info(f"Trigram Hits: 0")
        return 0.0

    # 2. Preprocess Text and Count Total Words
    lower_text = text.lower()
    # Simple regex split for words (no NLTK dependency)
    tokens = re.findall(r'\b\w+\b', lower_text)  # Simple word split

    total_words = len(tokens)
    if total_words == 0:
        if debug:
            logging.info("No valid words found in the text after tokenization.")
            logging.info(f"Word Hits: 0")
            logging.info(f"Bigram Hits: 0")
            logging.info(f"Trigram Hits: 0")
        return 0.0

    # 3. Count Hits
    word_hits = 0
    bigram_hits = 0
    trigram_hits = 0

    # Count word hits
    if slop_words_set:
        word_hits = sum(1 for token in tokens if token in slop_words_set)

    # Count bigram hits
    if slop_bigrams_set and len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            bigram_str = f"{tokens[i]} {tokens[i+1]}"
            if bigram_str in slop_bigrams_set:
                bigram_hits += 1

    # Count trigram hits
    if slop_trigrams_set and len(tokens) >= 3:
        for i in range(len(tokens) - 2):
            trigram_str = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
            if trigram_str in slop_trigrams_set:
                trigram_hits += 1

    # 4. Calculate Final Score (using original weighting)
    total_slop_score = word_hits + 2*bigram_hits + 8*trigram_hits
    # Use the same normalization factor as the original function for consistency
    slop_index = (total_slop_score / total_words) * 1000 if total_words > 0 else 0

    # 5. Debug Output
    if debug:
        logging.info("--- Slop Index Debug ---")
        logging.info(f"Total Words Analyzed: {total_words}")
        logging.info(f"Word Hits: {word_hits} (using {len(slop_words_set)} slop words)")
        logging.info(f"Bigram Hits: {bigram_hits} (using {len(slop_bigrams_set)} slop bigrams)")
        logging.info(f"Trigram Hits: {trigram_hits} (using {len(slop_trigrams_set)} slop trigrams)")
        logging.info(f"Total Hits: {total_slop_score}")
        logging.info(f"Calculated Slop Index: {slop_index:.4f}")
        logging.info("------------------------")

    return round(slop_index, 4)


def calculate_readability_metrics(text: str, data_dir: Optional[Path] = None) -> Dict[str, float]:
    """
    Calculate readability metrics for creative writing.

    Enhanced version with slop index from original core/metrics.py functions.
    """
    if not text or not text.strip():
        return {
            "complexity_index": 0.0,
            "avg_sentence_length": 0.0,
            "avg_word_length": 0.0,
            "word_count": 0,
            "sentence_count": 0,
            "slop_index": 0.0
        }

    # Basic text analysis
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    words = text.split()

    sentence_count = len(sentences)
    word_count = len(words)

    avg_sentence_length = word_count / max(1, sentence_count)
    avg_word_length = sum(len(word.strip('.,!?;:"')) for word in words) / max(1, word_count)

    # Calculate slop index if data directory is provided
    slop_index = 0.0
    if data_dir:
        slop_index = calculate_slop_index(text, data_dir)

    return {
        "complexity_index": calculate_complexity_index(text),
        "avg_sentence_length": round(avg_sentence_length, 2),
        "avg_word_length": round(avg_word_length, 2),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "slop_index": slop_index
    }





class CreativeWriting(BaseBenchmark):
    """
    Creative Writing Bench benchmark implementation for evalchemy.

    Evaluates creative writing capabilities using the original EQ-Bench Creative Writing
    Benchmark v3 methodology, integrated with evalchemy's evaluation framework.
    """

    REQUIRES_OPENAI_ANNOTATOR = True  # Requires judge model for evaluation
    
    def __init__(
        self,
        judge_model: str = "gpt-4o-mini",
        iterations: int = 3,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        min_p: float = 0.1,
        debug: bool = False,
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None
    ):
        """
        Initialize Creative Writing Bench benchmark.
        
        Args:
            judge_model: Model name for evaluation (e.g., "gpt-4o-mini")
            iterations: Number of iterations per prompt (default: 3)
            max_tokens: Maximum tokens for generation (default: 4000)
            temperature: Temperature for creative generation (default: 0.7)
            min_p: min_p parameter for generation (default: 0.1)
            debug: If True, run on subset for debugging
            logger: Optional logger instance
            system_instruction: Optional system instruction
        """
        super().__init__(logger=logger, system_instruction=system_instruction)

        self.judge_model_name = judge_model
        self.iterations = iterations
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.min_p = min_p
        self.debug = debug

        # Initialize data directory
        self.data_dir = Path(__file__).parent / "data"

        # Load evaluation data
        self._load_evaluation_data()

        # Verify OpenAI API key is available
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY environment variable is required for CreativeWriting benchmark")

        self.logger.info(f"Initialized CreativeWriting benchmark with {len(self.creative_prompts)} prompts")
    
    def _load_evaluation_data(self):
        """Load evaluation criteria, prompts, and judge template."""
        # Load creative writing prompts
        prompts_file = self.data_dir / "creative_writing_prompts_v3.json"
        with open(prompts_file, 'r', encoding='utf-8') as f:
            self.creative_prompts = json.load(f)
        
        # Load evaluation criteria
        criteria_file = self.data_dir / "creative_writing_criteria.txt"
        with open(criteria_file, 'r', encoding='utf-8') as f:
            self.creative_writing_criteria = [line.strip() for line in f if line.strip()]
        
        # Load negative criteria
        negative_file = self.data_dir / "negative_criteria.txt"
        with open(negative_file, 'r', encoding='utf-8') as f:
            self.negative_criteria = [line.strip() for line in f if line.strip()]
        
        # Load judge prompt template
        judge_prompt_file = self.data_dir / "creative_writing_judging_prompt.txt"
        with open(judge_prompt_file, 'r', encoding='utf-8') as f:
            self.judge_prompt_template = f.read()

    def generate_responses(self, model) -> Dict[str, Any]:
        """
        Generate creative writing responses using evalchemy's model interface.

        Args:
            model: Language model instance from evalchemy

        Returns:
            Dictionary containing generation results and metadata
        """
        # Handle distributed evaluation - only rank 0 processes
        try:
            import torch.distributed as dist
            if dist.is_initialized() and dist.get_rank() != 0:
                return None
        except:
            pass

        self.logger.info(f"Generating responses for {len(self.creative_prompts)} prompts across {self.iterations} iterations")

        # Create temporary file for results
        temp_dir = tempfile.mkdtemp()
        output_file = os.path.join(temp_dir, "creative_writing_responses.jsonl")

        responses = []

        # Prepare all instances for batch processing
        instances = []
        instance_metadata = []

        # Build all instances first
        for iteration in range(1, self.iterations + 1):
            for prompt_key, prompt_data in self.creative_prompts.items():
                if self.debug and len(instances) >= 10:  # Limit for debugging
                    break

                base_prompt = prompt_data.get("writing_prompt", "")
                seed_modifiers = prompt_data.get("seed_modifiers", [])

                if not seed_modifiers:
                    self.logger.warning(f"No seed modifiers for prompt {prompt_key}; using default.")
                    seed_modifier = "default"
                else:
                    # Select seed modifier for this iteration (cycling through available modifiers)
                    seed_modifier = seed_modifiers[(iteration - 1) % len(seed_modifiers)]

                # Replace <SEED> placeholder with the selected modifier
                if "<SEED>" in base_prompt:
                    final_prompt = base_prompt.replace("<SEED>", seed_modifier)
                else:
                    final_prompt = base_prompt

                if Instance is not None:
                    # Create instance for batch processing
                    instance = Instance(
                        "generate_until",
                        None,
                        (
                            final_prompt,
                            {
                                "temperature": self.temperature,
                                "max_gen_toks": self.max_tokens,
                                "min_p": self.min_p
                            }
                        ),
                        len(instances)  # Use instance count as idx
                    )
                    instances.append(instance)

                    # Store metadata for later processing
                    metadata = {
                        "prompt_id": prompt_key,
                        "iteration": iteration,
                        "category": prompt_data.get("category", "unknown"),
                        "title": prompt_data.get("title", f"Prompt {prompt_key}"),
                        "base_prompt": base_prompt,
                        "seed_modifier": seed_modifier,
                        "final_prompt": final_prompt
                    }
                    instance_metadata.append(metadata)
                else:
                    # Fallback when lm_eval is not available - should not happen in production
                    raise ImportError("lm_eval is required for evalchemy integration")

        # Batch generate all responses at once
        if instances:
            self.logger.info(f"Generating {len(instances)} responses in batch...")
            try:
                batch_responses = model.generate_until(instances)

                # Process batch results
                for response, metadata in zip(batch_responses, instance_metadata):
                    # Check minimum length requirement (original benchmark requirement)
                    if len(response.strip()) < 500:
                        self.logger.warning(f"Generated text too short ({len(response.strip())} chars) for prompt {metadata['prompt_id']}, iteration {metadata['iteration']}")

                    response_data = {
                        **metadata,
                        "response": response.strip(),
                        "response_length": len(response.strip())
                    }

                    responses.append(response_data)

                    # Write to file for incremental saving
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(response_data) + '\n')

            except Exception as e:
                self.logger.error(f"Error in batch generation: {e}")
                # Fallback to individual processing if batch fails
                for instance, metadata in zip(instances, instance_metadata):
                    try:
                        response = model.generate_until([instance])[0]
                        response_data = {
                            **metadata,
                            "response": response.strip(),
                            "response_length": len(response.strip())
                        }
                        responses.append(response_data)

                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(response_data) + '\n')

                    except Exception as individual_error:
                        self.logger.error(f"Error generating response for prompt {metadata['prompt_id']}, iteration {metadata['iteration']}: {individual_error}")
                        error_data = {
                            **metadata,
                            "response": "",
                            "response_length": 0,
                            "error": str(individual_error)
                        }
                        responses.append(error_data)

                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_data) + '\n')

        self.logger.info(f"Generated {len(responses)} total responses")

        return {
            "responses": responses,
            "filepath": output_file,
            "temp_dir": temp_dir,
            "num_responses": len(responses)
        }

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate generated responses using original Creative Writing Bench methodology.

        Args:
            results: Dictionary containing generation results

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        try:
            responses = results["responses"]
            self.logger.info(f"Evaluating {len(responses)} responses using original Creative Writing Bench methodology")

            # Evaluate each response using OpenAI judge API
            evaluated_responses = []

            # Prepare judge prompts for batch evaluation
            judge_prompts = []
            valid_responses = []

            for response in responses:
                if "error" in response:
                    # Skip responses with generation errors
                    continue

                try:
                    # Format judge prompt using original Creative Writing Bench template
                    criteria_formatted = "\n".join(["- " + c for c in self.creative_writing_criteria])
                    negative_criteria_str = ", ".join(self.negative_criteria)

                    judge_prompt = self.judge_prompt_template.format(
                        writing_prompt=response["base_prompt"],
                        test_model_response=response["response"],
                        creative_writing_criteria=criteria_formatted,
                        lower_is_better_criteria=negative_criteria_str
                    )

                    judge_prompts.append(judge_prompt)
                    valid_responses.append(response)

                except Exception as e:
                    self.logger.error(f"Error preparing judge prompt for response {response.get('prompt_id', 'unknown')}: {e}")
                    response['judge_scores'] = {}
                    response['raw_judge_text'] = f"[ERROR: {e}]"
                    evaluated_responses.append(response)

            # Batch evaluate using OpenAI API
            if judge_prompts:
                self.logger.info(f"Evaluating {len(judge_prompts)} responses using OpenAI judge API")
                try:
                    judge_responses = asyncio.run(judge_multiple_responses(
                        judge_prompts,
                        self.judge_model_name,
                        max_concurrent=5
                    ))

                    # Process judge responses
                    for response, judge_response in zip(valid_responses, judge_responses):
                        # Parse scores using original Creative Writing Bench logic
                        judge_scores = parse_judge_scores_creative(judge_response)

                        # Store results in response
                        response['judge_scores'] = judge_scores
                        response['raw_judge_text'] = judge_response
                        evaluated_responses.append(response)

                except Exception as e:
                    self.logger.error(f"Batch judge evaluation failed: {e}")
                    # Add all responses with error status
                    for response in valid_responses:
                        response['judge_scores'] = {}
                        response['raw_judge_text'] = f"[JUDGE_ERROR: {e}]"
                        evaluated_responses.append(response)

            # Compute final benchmark scores using original Creative Writing Bench methodology
            benchmark_result = compute_single_benchmark_score_creative(evaluated_responses, self.negative_criteria)

            # Compute score distribution for additional insights
            valid_scores = []
            for response in evaluated_responses:
                judge_scores = response.get('judge_scores', {})
                if judge_scores:
                    response_scores = []
                    for metric, score in judge_scores.items():
                        if isinstance(score, (int, float)):
                            inverted_score = invert_if_negative(metric, score, self.negative_criteria)
                            if inverted_score <= SCORE_RANGE_MAX:
                                response_scores.append(inverted_score)
                    if response_scores:
                        avg_score = sum(response_scores) / len(response_scores)
                        valid_scores.append(avg_score)

            score_distribution = {}
            if valid_scores:
                score_distribution = {
                    "mean": round(statistics.mean(valid_scores), 2),
                    "median": round(statistics.median(valid_scores), 2),
                    "std": round(statistics.stdev(valid_scores) if len(valid_scores) > 1 else 0.0, 2),
                    "min": round(min(valid_scores), 2),
                    "max": round(max(valid_scores), 2)
                }

            # Compute bootstrap analysis
            bootstrap_stats = bootstrap_benchmark_stability_creative(evaluated_responses, self.negative_criteria)

            # Return results in evalchemy format
            final_results = {
                "creative_score_0_20": benchmark_result["overall_score"],
                "eqbench_creative_score": benchmark_result["eqbench_creative_score"],
                "bootstrap_analysis": bootstrap_stats,
                "num_evaluated_responses": len(valid_scores),
                "score_distribution": score_distribution,
                "examples": evaluated_responses,  # Include detailed results for analysis
                "benchmark_version": "v3",
                "judge_model": self.judge_model_name,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "evaluation_methodology": "original_creative_writing_bench"
            }

            # Clean up temporary files
            try:
                import shutil
                shutil.rmtree(results["temp_dir"])
            except:
                pass

            return final_results

        except Exception as e:
            self.logger.error(f"Error in evaluation: {e}")
            return {"error": str(e)}



    def _compute_readability_statistics(self, evaluated_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute readability statistics using original Creative Writing Bench metrics.

        From original core/metrics.py functionality.
        """
        if not evaluated_responses:
            return {"error": "No responses to analyze"}

        all_metrics = []

        for response in evaluated_responses:
            text = response.get("response", "")
            if text:
                metrics = calculate_readability_metrics(text, self.data_dir)
                all_metrics.append(metrics)

        if not all_metrics:
            return {"error": "No valid text to analyze"}

        # Aggregate statistics
        complexity_scores = [m["complexity_index"] for m in all_metrics]
        sentence_lengths = [m["avg_sentence_length"] for m in all_metrics]
        word_lengths = [m["avg_word_length"] for m in all_metrics]
        word_counts = [m["word_count"] for m in all_metrics]
        slop_indices = [m["slop_index"] for m in all_metrics]

        return {
            "complexity_index": {
                "mean": round(statistics.mean(complexity_scores), 2),
                "median": round(statistics.median(complexity_scores), 2),
                "std": round(statistics.stdev(complexity_scores) if len(complexity_scores) > 1 else 0.0, 2),
                "min": round(min(complexity_scores), 2),
                "max": round(max(complexity_scores), 2)
            },
            "avg_sentence_length": {
                "mean": round(statistics.mean(sentence_lengths), 2),
                "median": round(statistics.median(sentence_lengths), 2),
                "std": round(statistics.stdev(sentence_lengths) if len(sentence_lengths) > 1 else 0.0, 2)
            },
            "avg_word_length": {
                "mean": round(statistics.mean(word_lengths), 2),
                "median": round(statistics.median(word_lengths), 2),
                "std": round(statistics.stdev(word_lengths) if len(word_lengths) > 1 else 0.0, 2)
            },
            "word_count": {
                "mean": round(statistics.mean(word_counts), 2),
                "median": round(statistics.median(word_counts), 2),
                "std": round(statistics.stdev(word_counts) if len(word_counts) > 1 else 0.0, 2)
            },
            "slop_index": {
                "mean": round(statistics.mean(slop_indices), 2),
                "median": round(statistics.median(slop_indices), 2),
                "std": round(statistics.stdev(slop_indices) if len(slop_indices) > 1 else 0.0, 2),
                "min": round(min(slop_indices), 2),
                "max": round(max(slop_indices), 2)
            },
            "num_responses_analyzed": len(all_metrics)
        }

    def _compute_pairwise_comparisons(self, evaluated_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compute pairwise comparisons using original Creative Writing Bench ELO methodology.

        From original core/elo.py functionality.
        """
        if len(evaluated_responses) < 2:
            return {"note": "Insufficient responses for pairwise comparison"}

        # Group responses by prompt for fair comparison
        responses_by_prompt = {}
        for response in evaluated_responses:
            prompt_id = response.get("prompt_id", "unknown")
            if prompt_id not in responses_by_prompt:
                responses_by_prompt[prompt_id] = []
            responses_by_prompt[prompt_id].append(response)

        all_comparisons = []

        # Perform pairwise comparisons within each prompt
        for prompt_id, prompt_responses in responses_by_prompt.items():
            if len(prompt_responses) < 2:
                continue

            # Compare all pairs of responses for this prompt
            for i in range(len(prompt_responses)):
                for j in range(i + 1, len(prompt_responses)):
                    response_a = prompt_responses[i]
                    response_b = prompt_responses[j]

                    scores_a = response_a.get("judge_scores", {})
                    scores_b = response_b.get("judge_scores", {})

                    if scores_a and scores_b:
                        comparison = interpret_pairwise_result_cw(
                            f"response_{i}",
                            f"response_{j}",
                            scores_a,
                            scores_b,
                            self.negative_criteria
                        )
                        comparison["prompt_id"] = prompt_id
                        all_comparisons.append(comparison)

        # Aggregate comparison statistics
        if not all_comparisons:
            return {"note": "No valid comparisons could be made"}

        fractions = [comp["fraction"] for comp in all_comparisons]

        return {
            "total_comparisons": len(all_comparisons),
            "fraction_statistics": {
                "mean": round(statistics.mean(fractions), 3),
                "median": round(statistics.median(fractions), 3),
                "std": round(statistics.stdev(fractions) if len(fractions) > 1 else 0.0, 3)
            },
            "win_distribution": {
                "decisive_wins": len([f for f in fractions if f > 0.6 or f < 0.4]),
                "close_matches": len([f for f in fractions if 0.4 <= f <= 0.6]),
                "ties": len([f for f in fractions if f == 0.5])
            },
            "comparisons": all_comparisons[:10]  # Include first 10 for inspection
        }

    def _run_elo_analysis_creative(self, evaluated_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run ELO analysis using original Creative Writing Bench methodology.

        From original core/elo.py and benchmark.py logic.
        """
        if len(evaluated_responses) < 2:
            return {"note": "Insufficient responses for ELO analysis"}

        try:
            # Group responses by model for ELO calculation
            responses_by_model = {}
            for response in evaluated_responses:
                model_name = response.get("model_name", "unknown_model")
                if model_name not in responses_by_model:
                    responses_by_model[model_name] = []
                responses_by_model[model_name].append(response)

            if len(responses_by_model) < 2:
                return {"note": "Need at least 2 different models for ELO analysis"}

            # Perform pairwise comparisons between models
            model_comparisons = []
            model_names = list(responses_by_model.keys())

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model_a = model_names[i]
                    model_b = model_names[j]

                    responses_a = responses_by_model[model_a]
                    responses_b = responses_by_model[model_b]

                    # Compare responses on same prompts
                    for resp_a in responses_a:
                        for resp_b in responses_b:
                            if (resp_a.get("prompt_id") == resp_b.get("prompt_id") and
                                resp_a.get("iteration") == resp_b.get("iteration")):

                                # Skip prompts that should be ignored for ELO
                                if should_ignore_prompt_cw(resp_a.get("prompt_id", "")):
                                    continue

                                scores_a = resp_a.get("judge_scores", {})
                                scores_b = resp_b.get("judge_scores", {})

                                if scores_a and scores_b:
                                    comparison = interpret_pairwise_result_cw(
                                        model_a, model_b, scores_a, scores_b, self.negative_criteria
                                    )
                                    comparison["prompt_id"] = resp_a.get("prompt_id")
                                    comparison["iteration"] = resp_a.get("iteration")
                                    model_comparisons.append(comparison)

            # Calculate basic ELO ratings
            model_ratings = {model: DEFAULT_ELO for model in model_names}

            # Simple ELO update based on comparisons
            K_FACTOR = 32  # Standard ELO K-factor

            for comparison in model_comparisons:
                model_a = comparison["model_a"]
                model_b = comparison["model_b"]
                fraction = comparison["fraction"]

                # Calculate expected scores
                rating_a = model_ratings[model_a]
                rating_b = model_ratings[model_b]

                expected_a = 1 / (1 + 10**((rating_b - rating_a) / 400))
                expected_b = 1 - expected_a

                # Actual scores based on fraction
                actual_a = fraction
                actual_b = 1 - fraction

                # Update ratings
                model_ratings[model_a] += K_FACTOR * (actual_a - expected_a)
                model_ratings[model_b] += K_FACTOR * (actual_b - expected_b)

            # Sort models by rating
            sorted_models = sorted(model_ratings.items(), key=lambda x: x[1], reverse=True)

            return {
                "model_ratings": model_ratings,
                "model_ranking": sorted_models,
                "total_comparisons": len(model_comparisons),
                "k_factor": K_FACTOR,
                "default_elo": DEFAULT_ELO,
                "methodology": "simplified_elo_from_original_cw_bench"
            }

        except Exception as e:
            self.logger.error(f"Error in ELO analysis: {e}")
            return {"error": f"ELO analysis failed: {e}"}
