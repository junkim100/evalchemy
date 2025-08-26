# LogicKor Benchmark

LogicKor is a Korean logical reasoning benchmark designed to evaluate the logical reasoning capabilities of large language models in Korean across multiple domains.

## Overview

- **Repository**: [LogicKor](https://github.com/instructkr/LogicKor)
- **Website**: [https://lk.instruct.kr](https://lk.instruct.kr)
- **Dataset**: GitHub JSONL file (42 examples across 6 categories)
- **Task Type**: Multi-turn Korean logical reasoning
- **Evaluation**: Rule-based scoring (1-10 scale)
- **Language**: Korean (한국어)

## Key Features

- **Korean Language Focus**: Specifically designed for evaluating Korean language models
- **Multi-Domain Coverage**: Six major categories covering diverse reasoning tasks
- **Multi-Turn Conversations**: Each example contains follow-up questions
- **Comprehensive Evaluation**: Rule-based scoring with category-specific metrics
- **Real-World Reasoning**: Practical logical reasoning scenarios

## Task Categories

1. **추론(Reasoning)**: Logical deduction and inference problems
2. **수학(Math)**: Mathematical problem solving and calculations
3. **글쓰기(Writing)**: Creative and analytical writing tasks
4. **코딩(Coding)**: Programming and algorithm problems
5. **이해(Understanding)**: Reading comprehension and analysis
6. **문법(Grammar)**: Korean language grammar and usage

## Dataset Structure

Each example contains:
- `id`: Unique identifier
- `category`: Task category (e.g., "추론(Reasoning)")
- `questions`: List of questions (typically 2 questions per example)
- `references`: Reference answers (may be null for open-ended questions)

## Usage

### Basic Usage

```bash
python -m eval.eval --model vllm --tasks LogicKor --model_args "pretrained=MODEL_NAME" --batch_size auto --output_path logs
```

### Configuration Options

The `LogicKorBenchmark` class supports the following parameters:

#### Dataset Options
- `dataset_url` (str): URL to LogicKor questions.jsonl file (default: GitHub raw URL)

#### Generation Options
- `max_tokens` (int): Maximum tokens for model generation (default: `32768`)
- `seed` (List[int]): Random seed for reproducibility (default: `[0, 1234, 1234, 1234]`)
- `enable_cot` (bool): Enable Chain-of-Thought prompting (default: `False`)

#### Filtering Options
- `filter_category` (str): Filter by specific category (optional)

#### Evaluation Options
- `debug` (bool): If True, only evaluate on 5 examples (default: `False`)
- `logger` (logging.Logger): Optional logger instance
- `system_instruction` (str): Optional system instruction for the model
- `judge_model` (str): Model for judge evaluation (default: `"gpt-4"`)
- `openai_api_key` (str): OpenAI API key for judge evaluation (optional)

### Example Configurations

#### Full Dataset
```python
benchmark = LogicKorBenchmark(
    debug=False,
    max_tokens=32768
)
```

#### Specific Category
```python
benchmark = LogicKorBenchmark(
    filter_category="수학(Math)",
    debug=False
)
```

#### Chain-of-Thought Mode
```python
benchmark = LogicKorBenchmark(
    enable_cot=True,
    max_tokens=32768
)
```

#### Debug Mode
```python
benchmark = LogicKorBenchmark(
    debug=True,  # Only 5 examples
    filter_category="추론(Reasoning)"
)
```

## Available Categories

Based on the dataset, the categories include:
- `추론(Reasoning)` - Logical reasoning and deduction
- `수학(Math)` - Mathematical problem solving
- `글쓰기(Writing)` - Writing and composition tasks
- `코딩(Coding)` - Programming and coding problems
- `이해(Understanding)` - Reading comprehension
- `문법(Grammar)` - Korean grammar and language usage

## Prompting Modes

### Direct Prompting (Default)
```
다음 문제에 대해 정확하고 자세한 답변을 한국어로 제공해주세요.

문제: [QUESTION]
```

### Chain-of-Thought Prompting
```
다음 문제를 단계별로 차근차근 생각해보고 답변해주세요.

문제: [QUESTION]

단계별로 생각해보겠습니다:
```

## Evaluation Metrics

- **Overall Score**: Average score across all examples (0-10 scale)
- **Category-specific Scores**: Average scores per category
- **Question Type Scores**: Scores for first questions vs follow-up questions
- **Standard Error**: Statistical confidence measure

## Evaluation Criteria

### Korean Language Requirement
- **Mandatory**: All responses must be primarily in Korean
- **Zero Score**: Non-Korean responses receive 0 points
- **Threshold**: At least 70% Korean characters required

### Rule-Based Scoring Components

1. **Base Score**: 5.0 points
2. **Length Scoring**: 
   - Too short (<10 chars): -2.0 points
   - Too long (>2000 chars): -1.0 points
   - Appropriate length: +1.0 points
3. **Reference Matching**: +3.0 points for correct answers
4. **Heuristic Scoring**: +1.0-2.0 points for structured responses
5. **Category Adjustments**: Stricter scoring for Math and Grammar

### Category-Specific Features

- **Math**: Number extraction and mathematical symbol detection
- **Coding**: Programming language and syntax recognition
- **Grammar**: Strict reference matching for correctness
- **Reasoning**: Logical structure and reasoning indicator detection

## Performance Considerations

- **Korean Processing**: Requires proper Korean text handling
- **Multi-Turn**: Processes multiple questions per example
- **Rule-Based**: Fast evaluation without external API dependencies
- **Scalable**: Efficient processing for the 42-example dataset

## Notes

- The benchmark automatically downloads data from GitHub
- Rule-based evaluation provides consistent scoring without API dependencies
- Korean language detection ensures responses meet language requirements
- Multi-turn structure captures conversational reasoning abilities
- Category filtering allows focused evaluation on specific domains

## Future Enhancements

- **Judge Evaluation**: Integration with GPT-4 for more sophisticated scoring
- **Extended Dataset**: Support for larger LogicKor datasets
- **Custom Metrics**: Domain-specific evaluation criteria
- **Multilingual**: Potential extension to other languages
