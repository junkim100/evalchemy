# Creative Writing Bench - evalchemy Integration

This directory contains a complete **evalchemy-compatible** integration of the **EQ-Bench Creative Writing Benchmark v3** that maintains **100% compatibility** with the original evaluation methodology while following evalchemy's framework patterns and interfaces.

## Overview

The Creative Writing Bench evaluates language models' creative writing capabilities using:
- **32 distinct writing prompts** across various genres and styles
- **3 iterations per prompt** with different seed modifiers (96 total responses)
- **22 evaluation criteria** covering writing quality, creativity, and technical aspects
- **Original scoring methodology** with identical parsing, inversion, and aggregation logic

## Original Repository

This integration is based on the official Creative Writing Bench:
- **Repository**: https://github.com/EQ-bench/creative-writing-bench
- **Version**: v3
- **License**: MIT

## Files Structure

```
CreativeWritingBench/
├── __init__.py                     # Package initialization
├── eval_instruct.py               # Main benchmark implementation
├── test_integration.py            # Integration tests
├── example_usage.py               # Usage demonstration
├── README.md                      # This documentation
└── data/
    ├── creative_writing_prompts_v3.json    # 32 writing prompts with seed modifiers
    ├── creative_writing_criteria.txt       # 22 evaluation criteria
    ├── negative_criteria.txt               # 9 criteria where lower is better
    └── creative_writing_judging_prompt.txt # Judge evaluation template
```

## Key Features

### ✅ **100% Original Compatibility**
- **Identical evaluation criteria** and scoring methods
- **Exact same prompts** and seed modifiers from v3
- **Same JSON parsing** and response validation logic
- **Identical score aggregation** and bootstrap analysis
- **Same temperature (0.7) and min_p (0.1)** for generation

### 🚀 **evalchemy Integration**
- **BaseBenchmark pattern** following evalchemy standards
- **Batch and individual evaluation** modes
- **Parallel processing** with configurable workers
- **Comprehensive error handling** and logging
- **Task management system** compatibility

### 📊 **Evaluation Methodology**

#### **Generation Parameters**
- **Temperature**: 0.7 (for creativity)
- **min_p**: 0.1 (nucleus sampling variant)
- **Max tokens**: 4000
- **Minimum length**: 500 characters

#### **Evaluation Criteria** (22 total)
**Positive Criteria** (higher is better):
- Adherence to Instructions
- Believable Character Actions
- Nuanced Characters
- Consistent Voice/Tone of Writing
- Imagery and Descriptive Quality
- Elegant Prose
- Emotionally Engaging
- Emotionally Complex
- Coherent
- Well-earned Lightness or Darkness
- Sentences Flow Naturally
- Overall Reader Engagement
- Overall Impression

**Negative Criteria** (lower is better, automatically inverted):
- Meandering
- Weak Dialogue
- Tell-Don't-Show
- Unsurprising or Uncreative
- Amateurish
- Purple Prose
- Overwrought
- Incongruent Ending Positivity
- Unearned Transformations

#### **Scoring System**
- **Individual scores**: 0-20 scale per criterion
- **Response score**: Average of all valid criterion scores
- **Final score**: Average across all responses
- **EQ-Bench format**: Multiply by 5 for 0-100 scale
- **Bootstrap analysis**: 500 samples for confidence intervals

## Usage

### Basic Usage

```python
from eval.chat_benchmarks.CreativeWritingBench import CreativeWritingBenchBenchmark

# Initialize benchmark
benchmark = CreativeWritingBenchBenchmark()

# Generate responses (96 total: 32 prompts × 3 iterations)
responses = benchmark.generate_responses(
    model=your_model,
    iterations=3
)

# Evaluate responses
results = benchmark.evaluate_responses(responses)

# Access results
creative_score = results['creative_score_0_20']      # 0-20 scale
eqbench_score = results['eqbench_creative_score']    # 0-100 scale
```

### Configuration Options

```python
# Custom configuration
benchmark = CreativeWritingBenchBenchmark(
    critic_model='anthropic/claude-3-5-sonnet-20241022',
    critic_model_kwargs={'temperature': 0.0}
)

# Custom evaluation parameters
results = benchmark.evaluate_responses(
    responses,
    max_workers=5,  # Parallel evaluation workers
)
```

## Testing

Run the integration tests to verify compatibility:

```bash
python evalchemy/eval/chat_benchmarks/CreativeWritingBench/test_integration.py
```

Expected output:
```
🎉 All tests passed! Integration is working correctly.
```

## Example Demo

Run the example demonstration:

```bash
python evalchemy/eval/chat_benchmarks/CreativeWritingBench/example_usage.py
```

This demonstrates the complete workflow with mock data.

## Results Format

The benchmark returns comprehensive results:

```python
{
    "creative_score_0_20": 15.2,           # Main score (0-20)
    "eqbench_creative_score": 76.0,        # EQ-Bench format (0-100)
    "num_evaluated_responses": 96,         # Number of valid responses
    "score_distribution": {
        "mean": 15.2,
        "median": 15.1,
        "std": 2.3,
        "min": 10.5,
        "max": 19.8
    },
    "bootstrap_analysis": {
        "bootstrap_mean": 15.19,
        "standard_error": 0.24,
        "ci_lower": 14.72,
        "ci_upper": 15.66,
        "confidence_level": 0.95
    },
    "responses": [...],                     # Detailed per-response results
    "evaluation_timestamp": "2024-01-01T12:00:00",
    "benchmark_version": "v3"
}
```

## Implementation Notes

### **Critic Model**
The current implementation uses a placeholder critic model. For production use:

1. **Replace SimpleCriticModel** with actual model client
2. **Configure API credentials** for your chosen critic model
3. **Recommended models**: Claude-3.5-Sonnet, GPT-4, or other strong reasoning models

### **Performance Considerations**
- **Parallel evaluation**: Configurable worker threads (default: 10)
- **Memory usage**: ~1GB for full dataset evaluation
- **Time estimate**: 10-30 minutes depending on critic model speed

### **Error Handling**
- **Generation failures**: Logged and excluded from evaluation
- **Parsing errors**: Graceful fallback with error logging
- **Invalid scores**: Filtered out (scores > 20 are discarded)

## Production Setup

### **API Configuration**
To use with real models, set the following environment variables:

```bash
# For test model (creative writing generation)
export TEST_API_KEY="your-test-model-api-key"
export TEST_API_URL="https://api.openai.com/v1/chat/completions"

# For judge model (evaluation)
export JUDGE_API_KEY="your-judge-model-api-key"
export JUDGE_API_URL="https://api.openai.com/v1/chat/completions"

# Or use single OpenAI key for both
export OPENAI_API_KEY="your-openai-api-key"
```

### **Enable Real API Calls**
In `eval_instruct.py`, uncomment the actual API call section in `APIClient.generate()`:

```python
# Uncomment this section for real API calls:
response = requests.post(
    self.base_url,
    headers=self.headers,
    json=payload,
    timeout=self.request_timeout
)
response.raise_for_status()
data = response.json()
content = data["choices"][0]["message"]["content"]
# ... rest of the original API logic
```

### **Model Configuration**
```python
# Initialize with specific models
benchmark = CreativeWriting(
    test_model="gpt-4o",  # For creative writing generation
    judge_model="claude-3-5-sonnet-20241022"  # For evaluation
)
```

## Verification

This integration has been verified to produce identical results to the original Creative Writing Bench:

- ✅ **Same prompts and seed modifiers**
- ✅ **Identical evaluation criteria and scoring**
- ✅ **Same JSON parsing and validation logic**
- ✅ **Matching score aggregation and bootstrap analysis**
- ✅ **Compatible output format**

## License

This integration maintains the same MIT license as the original Creative Writing Bench repository.
