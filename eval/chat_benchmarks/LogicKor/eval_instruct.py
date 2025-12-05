import logging
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from eval.task import BaseBenchmark

# Judge evaluation template
JUDGE_TEMPLATE = {
    "single_turn": """너는 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.

# 기본 규칙
1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.
2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.
3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.
4. Additional Reference가 제공된다면 평가 시 해당 정보를 참고할 것.

# 언어 요구사항
- 모델은 반드시 한국어로 답변해야 하며, 다른 언어로의 답변은 절대 허용되지 않는다.
- 예외적으로 질문이 영어로 답변할 것을 요구할 때에만 영어 답변이 허용된다.
- 한국어로 답변하지 않을 경우, 점수는 0점 처리된다.
- 언어 요구사항을 충족하는 것은 필수적이나, 이 요구사항의 충족이 답변의 질적 평가에 추가 점수로 이어지지는 않는다.

# 평가 출력 방식
**주어진 Question에 집중하여** Model's Response에 대한 평가와 1~10의 점수를 부여한다. 답변에 대한 평가는 4~5 문장으로 규칙을 참고하여 상세히 작성한다.

# 출력 형식
평가: 평가 내용
점수: 숫자""",
    "multi_turn": """너는 대화 후 이어지는 후속 질문에 대한 한국어 언어 모델의 답변을 매우 꼼꼼히 평가할 것이다. 공정한 평가를 위해 아래의 규칙을 준수한다.

# 기본 규칙
1. 질문의 요구사항을 충분히 반영하였는지 상세히 분석할 것.
2. 답변 과정에서 누락되었거나 포함되지 못하여 아쉬운 부분에 대하여 상세히 분석할 것.
3. 답변의 길이가 평가 결과에 영향을 미치지 않도록 할 것.
4. Additional Reference가 제공된다면 평가 시 해당 정보를 참고할 것.
5. 후속 질문에 대한 답변이 이전 대화 맥락과 일치하는지 확인할 것.

# 언어 요구사항
- 모델은 반드시 한국어로 답변해야 하며, 다른 언어로의 답변은 절대 허용되지 않는다.
- 예외적으로 질문이 영어로 답변할 것을 요구할 때에만 영어 답변이 허용된다.
- 한국어로 답변하지 않을 경우, 점수는 0점 처리된다.
- 언어 요구사항을 충족하는 것은 필수적이나, 이 요구사항의 충족이 답변의 질적 평가에 추가 점수로 이어지지는 않는다.

# 평가 출력 방식
**주어진 Question에 집중하여** Model's Response에 대한 평가와 1~10의 점수를 부여한다. 답변에 대한 평가는 4~5 문장으로 규칙을 참고하여 상세히 작성한다.

# 출력 형식
평가: 평가 내용
점수: 숫자""",
}


class LogicKorBenchmark(BaseBenchmark):
    """
    LogicKor Benchmark for evaluating Korean logical reasoning capabilities of LLMs.

    LogicKor is a Korean logical reasoning benchmark designed to evaluate the logical
    reasoning capabilities of large language models in Korean across multiple domains.

    Features:
    - Korean language logical reasoning tasks
    - Multiple categories: Reasoning, Math, Writing, Coding, Understanding, Grammar
    - Multi-turn conversations with follow-up questions
    - Judge-based evaluation using GPT-4 or similar models
    - Comprehensive scoring from 1-10 scale

    Link: https://github.com/instructkr/LogicKor
    """

    def __init__(
        self,
        dataset_url: str = "https://raw.githubusercontent.com/instructkr/LogicKor/main/questions.jsonl",
        max_tokens: int = 32768,
        debug: bool = False,
        seed: List[int] = [0, 1234, 1234, 1234],
        logger: Optional[logging.Logger] = None,
        system_instruction: Optional[str] = None,
        filter_category: Optional[str] = None,
        prompt_strategy: str = "default",
        judge_model: str = "gpt-4",
        openai_api_key: Optional[str] = None,
        use_azure: bool = False,
    ):
        """
        Initialize LogicKor benchmark.

        Args:
            dataset_url: URL to the LogicKor questions.jsonl file
            max_tokens: Maximum tokens for model generation
            debug: If True, only evaluate on 10 examples
            seed: Random seed for reproducibility
            logger: Optional logger instance
            system_instruction: Optional system instruction for the model
            filter_category: Filter by specific category (optional)
            prompt_strategy: Prompt strategy (default, 1-shot, cot-1-shot)
            judge_model: Model to use for evaluation (requires OpenAI API)
            openai_api_key: OpenAI API key for judge model evaluation
            use_azure: Use Azure OpenAI API
        """
        super().__init__(logger=logger, system_instruction=system_instruction)
        
        self.dataset_url = dataset_url
        self.max_new_tokens = max_tokens
        self.debug = debug
        self.seed = seed
        self.filter_category = filter_category
        self.prompt_strategy = prompt_strategy
        self.judge_model = judge_model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.use_azure = use_azure

        # Load dataset
        self.dataset = self._load_dataset()

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load LogicKor dataset from URL."""
        try:
            import requests
            
            self.logger.info(f"Downloading dataset from {self.dataset_url}")
            response = requests.get(self.dataset_url, timeout=30)
            response.raise_for_status()
            
            # Parse JSONL data
            data = []
            for line in response.text.strip().split("\n"):
                if line.strip():
                    data.append(json.loads(line.strip()))

            # Apply category filter if specified
            if self.filter_category:
                original_count = len(data)
                data = [item for item in data if item.get("category") == self.filter_category]
                self.logger.info(
                    f"Filtered by category '{self.filter_category}': {len(data)}/{original_count} examples"
                )

            # Apply debug mode
            if self.debug:
                data = data[:10]
                self.logger.info(f"Debug mode: using {len(data)} examples")

            self.logger.info(f"Loaded {len(data)} LogicKor examples")
            return data

        except Exception as e:
            self.logger.error(f"Error loading LogicKor dataset from {self.dataset_url}: {str(e)}")
            raise

    def generate_responses(self, model: LM) -> Dict[str, Any]:
        """
        Generate solution completions using the provided model.

        This follows the LogicKor evaluation protocol:
        1. Generate response to first question (single-turn)
        2. Generate response to second question with conversation history (multi-turn)

        Args:
            model: Language model

        Returns:
            Dictionary containing generated responses in LogicKor format,
            or None for non-primary ranks
        """
        # Step 1: Generate single-turn responses (first question only)
        self.logger.info("Step 1: Generating single-turn responses...")
        single_turn_instances = []
        
        for idx, example in enumerate(self.dataset):
            questions = example.get("questions", [])
            if not questions:
                continue
                
            # First question only
            question = questions[0]
            prompt_messages = self._get_prompt_messages(question, is_first=True)
            templated_messages = self._prepare_messages(prompt_messages, model)

            instance = Instance(
                "generate_until",
                {
                    "example_id": example.get("id", idx),
                    "question": question,
                    "category": example.get("category", "unknown"),
                },
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "seed": self.seed,
                    },
                ),
                f"{idx}_single",
            )
            single_turn_instances.append(instance)

        single_turn_outputs = self.compute(model, single_turn_instances)

        # Return None early for non-primary ranks
        if model.rank != 0:
            return None

        # Step 2: Generate multi-turn responses (second question with history)
        self.logger.info("Step 2: Generating multi-turn responses...")
        multi_turn_instances = []
        
        for idx, example in enumerate(self.dataset):
            questions = example.get("questions", [])
            if len(questions) < 2:
                continue
                
            # Get the single-turn output for this example
            single_output = self._extract_text(single_turn_outputs[idx])
            
            # Second question with conversation history
            question_1 = questions[0]
            question_2 = questions[1]
            
            # Build conversation history
            prompt_messages = self._get_prompt_messages(question_1, is_first=True)
            prompt_messages.append({"role": "assistant", "content": single_output})
            prompt_messages.append({"role": "user", "content": question_2})
            
            templated_messages = self._prepare_messages(prompt_messages, model)

            instance = Instance(
                "generate_until",
                {
                    "example_id": example.get("id", idx),
                    "question": question_2,
                    "category": example.get("category", "unknown"),
                },
                (
                    templated_messages,
                    {
                        "do_sample": False,
                        "max_new_tokens": self.max_new_tokens,
                        "temperature": 0.0,
                        "seed": self.seed,
                    },
                ),
                f"{idx}_multi",
            )
            multi_turn_instances.append(instance)

        multi_turn_outputs = self.compute(model, multi_turn_instances)

        # Step 3: Combine results in LogicKor format
        self.logger.info("Step 3: Combining results...")
        examples = []
        
        for idx, example in enumerate(self.dataset):
            questions = example.get("questions", [])
            references = example.get("references", [])
            
            single_output = self._extract_text(single_turn_outputs[idx])
            multi_output = self._extract_text(multi_turn_outputs[idx]) if idx < len(multi_turn_outputs) else ""
            
            examples.append({
                "id": example.get("id", idx),
                "category": example.get("category", "unknown"),
                "questions": questions,
                "outputs": [single_output, multi_output],
                "references": references,
            })

        return {"examples": examples}

    def _extract_text(self, output: Any) -> str:
        """Extract text from different output types and remove think tokens."""
        if isinstance(output, str):
            text = output
        elif hasattr(output, "outputs") and output.outputs:
            text = output.outputs[0].text
        elif hasattr(output, "text"):
            text = output.text
        else:
            text = str(output)
        
        # Remove think tokens: <think>...</think> (removes both the tags and content between them)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        
        return text.strip()


    def _get_prompt_messages(self, question: str, is_first: bool = True) -> List[Dict[str, str]]:
        """
        Get prompt messages based on the prompt strategy.

        Args:
            question: The question text
            is_first: Whether this is the first question

        Returns:
            List of message dictionaries
        """
        messages = []
        
        # Add prompt strategy prefix if specified
        if self.prompt_strategy == "cot-1-shot":
            messages.extend([
                {
                    "role": "system",
                    "content": "문제를 해결하는 과정을 단계별로 명확하게 설명한 다음, 최종 답변을 제시해 주세요.",
                },
                {
                    "role": "user",
                    "content": "한 도시에서 매년 나무를 심고 있습니다. 첫해에 100그루를 심었고, 매년 전년도에 심은 나무의 수보다 20%씩 더 많은 나무를 심고 있습니다. 2년 후에는 총 몇 그루의 나무를 심게 될까요?",
                },
                {
                    "role": "assistant",
                    "content": """이 문제를 해결하기 위해 단계별로 접근해 보겠습니다.

1단계: 주어진 정보 정리
- 첫해에 100그루를 심음
- 매년 전년도보다 20% 더 많은 나무를 심음
- 2년 후의 총 나무 수를 구해야 함

2단계: 두 번째 해에 심은 나무 수 계산
- 첫해보다 20% 증가
- 100 + (100의 20%) = 100 + 20 = 120그루

3단계: 세 번째 해에 심은 나무 수 계산
- 두 번째 해보다 20% 증가
- 120 + (120의 20%) = 120 + 24 = 144그루

4단계: 3년 동안 심은 총 나무 수 계산
- 첫해: 100그루
- 둘째 해: 120그루
- 셋째 해: 144그루
- 총합: 100 + 120 + 144 = 364그루

따라서, 2년 후(즉, 3년째)에는 총 364그루의 나무를 심게 됩니다.""",
                },
            ])
        elif self.prompt_strategy == "1-shot":
            messages.extend([
                {
                    "role": "user",
                    "content": "Git의 rebase와 merge의 차이점을 설명하고, 각각 어떤 상황에서 사용하는 것이 좋은지 예를 들어 설명해주세요.",
                },
                {
                    "role": "assistant",
                    "content": """Git에서 `rebase`와 `merge`는 두 개의 브랜치를 통합하는 데 사용되는 두 가지 주요 명령어입니다. 이들의 차이점과 사용 상황을 설명하겠습니다.

### Rebase

#### 개념
- `rebase`는 하나의 브랜치의 커밋을 다른 브랜치의 끝에 다시 적용하는 것입니다. 즉, 커밋 히스토리를 새로 만든다고 할 수 있습니다.
- 결과적으로 커밋 히스토리가 깔끔하게 정리됩니다.

#### 사용 상황
- **깔끔한 히스토리가 필요할 때**: 브랜치 히스토리가 병합 없이 일직선으로 이어지도록 하고 싶을 때 유용합니다.
- **작업 중인 브랜치를 최신 상태로 유지하고 싶을 때**: 예를 들어, `feature` 브랜치에서 작업 중인데 `main` 브랜치에서 업데이트가 있을 경우, `feature` 브랜치를 `rebase`하여 최신 상태로 유지할 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업 중입니다.
2. `main` 브랜치에 새로운 커밋이 추가되었습니다.
3. `feature` 브랜치에서 `git rebase main`을 실행합니다.
4. `feature` 브랜치의 커밋들이 `main` 브랜치의 최신 커밋들 뒤에 재배치됩니다.

```bash
git checkout feature
git rebase main
```

### Merge

#### 개념
- `merge`는 두 개의 브랜치를 합치는 방법으로, 두 브랜치의 히스토리를 유지하며 새로운 병합 커밋을 생성합니다.
- `merge`는 기존의 히스토리를 그대로 보존합니다.

#### 사용 상황
- **히스토리를 유지하고 싶을 때**: 각 브랜치의 작업 기록을 그대로 보존하면서 병합할 때 사용합니다.
- **협업 중 충돌을 명확히 해결하고 기록을 남기고 싶을 때**: 여러 사람이 같은 저장소에서 작업할 때, `merge`는 충돌 해결 과정과 그 기록을 명확히 남길 수 있습니다.

#### 예제
1. `feature` 브랜치에서 작업을 완료했습니다.
2. `main` 브랜치에 병합하고 싶습니다.
3. `main` 브랜치로 체크아웃한 후 `feature` 브랜치를 병합합니다.

```bash
git checkout main
git merge feature
```

### 요약

- **Rebase**는 히스토리를 깔끔하게 정리하는 데 유리하며, 주로 개인 작업이나 `feature` 브랜치를 최신 상태로 유지할 때 사용됩니다.
- **Merge**는 두 브랜치의 작업 히스토리를 보존하면서 병합하며, 협업 과정에서 충돌 해결과 기록을 명확히 남기는 데 유리합니다.

각 방법의 장단점과 사용 상황을 고려하여 적절히 선택하는 것이 중요합니다.""",
                },
            ])
        
        # Add custom system instruction if provided
        if self.system_instruction and not messages:
            messages.append({
                "role": "system",
                "content": self.system_instruction,
            })
        
        # Add the actual question
        messages.append({
            "role": "user",
            "content": question,
        })
        
        return messages

    def evaluate_responses(self, results: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate the generated responses using GPT-4 judge.

        Args:
            results: Dictionary containing examples with model outputs

        Returns:
            Dictionary containing evaluation metrics
        """
        # Handle None result from non-primary ranks
        if results is None:
            return None

        if not self.openai_api_key:
            self.logger.warning("No OpenAI API key provided. Skipping evaluation.")
            # Update results with default scores (like AIME24 pattern)
            results.update({
                "num_total": len(results["examples"]),
                "score_single_turn": 0.0,
                "score_multi_turn": 0.0,
                "score_avg": 0.0,
            })
            return results

        examples = results["examples"]
        
        # Initialize OpenAI client
        try:
            if self.use_azure:
                from openai import AzureOpenAI
                client = AzureOpenAI(
                    azure_endpoint=os.environ.get("AZURE_ENDPOINT"),
                    api_key=self.openai_api_key,
                    api_version=os.environ.get("AZURE_API_VERSION", "2024-02-15-preview"),
                )
                model_name = os.environ.get("AZURE_DEPLOYMENT_NAME", "gpt-4")
            else:
                from openai import OpenAI
                client = OpenAI(api_key=self.openai_api_key)
                model_name = self.judge_model
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {str(e)}")
            # Update results with default scores
            results.update({
                "num_total": len(examples),
                "score_single_turn": 0.0,
                "score_multi_turn": 0.0,
                "score_avg": 0.0,
            })
            return results

        # Evaluate each example
        self.logger.info(f"Evaluating {len(examples)} examples with {model_name}...")
        
        # Track detailed results for individual samples
        detailed_results = []
        
        for idx, example in enumerate(examples):
            example_id = example.get("id", idx)
            category = example.get("category", "unknown")
            questions = example.get("questions", [])
            outputs = example.get("outputs", [])
            references = example.get("references", [])
            
            # Evaluate single-turn for this example
            single_result = self._evaluate_with_judge(
                client, model_name, example, is_multi_turn=False
            )
            example["query_single"] = single_result
            single_score = single_result["judge_score"]
            
            # Store detailed result for single-turn
            if len(questions) > 0 and len(outputs) > 0:
                detailed_results.append({
                    "example_id": example_id,
                    "question_index": 0,
                    "category": category,
                    "question": questions[0],
                    "output": outputs[0],
                    "reference": references[0] if len(references) > 0 else None,
                    "judge_score": single_score,
                    "final_score": single_score,
                    "question_type": "first_question",
                })
            
            # Evaluate multi-turn for the same example
            multi_result = self._evaluate_with_judge(
                client, model_name, example, is_multi_turn=True
            )
            example["query_multi"] = multi_result
            multi_score = multi_result["judge_score"]
            
            # Store detailed result for multi-turn
            if len(questions) > 1 and len(outputs) > 1:
                detailed_results.append({
                    "example_id": example_id,
                    "question_index": 1,
                    "category": category,
                    "question": questions[1],
                    "output": outputs[1],
                    "reference": references[1] if len(references) > 1 else None,
                    "judge_score": multi_score,
                    "final_score": multi_score,
                    "question_type": "follow_up_question",
                })
            
            # Log both scores together for this sample
            self.logger.info(
                f"ID {example_id} [{category}]: Turn 1 (Single)={single_score:.2f}점, Turn 2 (Multi)={multi_score:.2f}점"
            )
            
            # Progress logging every 10 examples
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Progress: {idx + 1}/{len(examples)} examples evaluated")
            
            # Small delay to avoid rate limiting
            time.sleep(0.5)

        # Calculate metrics
        single_scores = [ex["query_single"]["judge_score"] for ex in examples]
        multi_scores = [ex["query_multi"]["judge_score"] for ex in examples]
        
        avg_single = sum(single_scores) / len(single_scores) if single_scores else 0.0
        avg_multi = sum(multi_scores) / len(multi_scores) if multi_scores else 0.0
        avg_overall = (avg_single + avg_multi) / 2.0
        
        # Calculate category-specific scores
        category_stats = defaultdict(lambda: {"single_scores": [], "multi_scores": []})
        for example in examples:
            category = example["category"]
            category_stats[category]["single_scores"].append(example["query_single"]["judge_score"])
            category_stats[category]["multi_scores"].append(example["query_multi"]["judge_score"])
        
        category_scores = {}
        for category, scores in category_stats.items():
            avg_cat_single = sum(scores["single_scores"]) / len(scores["single_scores"])
            avg_cat_multi = sum(scores["multi_scores"]) / len(scores["multi_scores"])
            safe_category = category.replace("(", "_").replace(")", "_").replace(" ", "_")
            category_scores[f"score_{safe_category}_single"] = avg_cat_single
            category_scores[f"score_{safe_category}_multi"] = avg_cat_multi
        
        # Update results dict (like AIME24 does)
        # Ensure examples are included with query_single and query_multi
        results.update({
            "examples": examples,  # Include examples with query_single and query_multi
            "num_total": len(examples),
            "score_single_turn": avg_single,
            "score_multi_turn": avg_multi,
            "score_avg": avg_overall,
            "detailed_results": detailed_results,
            **category_scores,
        })
        
        self.logger.info(f"LogicKor Evaluation Results:")
        self.logger.info(f"  Total examples: {len(examples)}")
        self.logger.info(f"  Single-turn score: {avg_single:.2f}/10")
        self.logger.info(f"  Multi-turn score: {avg_multi:.2f}/10")
        self.logger.info(f"  Overall score: {avg_overall:.2f}/10")
        
        # Log category-specific results
        for category, scores in category_stats.items():
            avg_cat_single = sum(scores["single_scores"]) / len(scores["single_scores"])
            avg_cat_multi = sum(scores["multi_scores"]) / len(scores["multi_scores"])
            self.logger.info(
                f"  {category}: Single={avg_cat_single:.2f}, Multi={avg_cat_multi:.2f}"
            )
        
        return results

    def _evaluate_with_judge(
        self, client, model_name: str, example: Dict[str, Any], is_multi_turn: bool
    ) -> Dict[str, Any]:
        """Evaluate a single example using the judge model."""
        questions = example["questions"]
        outputs = example["outputs"]
        references = example["references"]
        
        # Build prompt
        prompt = (
            "아래의 내용을 주어진 평가 기준들을 충실히 반영하여 평가해라. "
            "특히 모델 답변이 언어 요구사항을 준수하는지 반드시 확인해야 한다.\n\n"
            f"**Question**\n{questions[0]}"
        )
        
        if references and references[0]:
            prompt += f"\n\n**Additional Reference**\n{references[0]}"
        
        prompt += f"\n\n**Model's Response**\n{outputs[0]}"
        
        if is_multi_turn and len(questions) > 1:
            prompt += f"\n\n**Follow-up Question.**\n{questions[1]}"
            if len(references) > 1 and references[1]:
                prompt += f"\n\n**Additional Reference**\n{references[1]}"
            if len(outputs) > 1:
                prompt += f"\n\n**Model's Response**\n{outputs[1]}"
        
        prompt += "\n\n[[대화 종료. 평가 시작.]]"
        
        # Call judge model
        template_key = "multi_turn" if is_multi_turn else "single_turn"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    temperature=0.0,
                    n=1,
                    messages=[
                        {"role": "system", "content": JUDGE_TEMPLATE[template_key]},
                        {"role": "user", "content": prompt},
                    ],
                )
                
                content = response.choices[0].message.content
                
                # Parse response
                judge_message_match = re.search(r"평가:(.*?)점수:", content.replace("*", ""), re.DOTALL)
                judge_message = judge_message_match.group(1).strip() if judge_message_match else "No judge message found"
                
                judge_score_match = re.search(r"점수:\s*(\d+(?:\.\d+)?)", content.replace("*", ""))
                if judge_score_match:
                    judge_score = float(judge_score_match.group(1))
                else:
                    raise ValueError("No score found in response")
                
                return {"judge_message": judge_message, "judge_score": judge_score}
                
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Error in judge evaluation (attempt {attempt + 1}): {str(e)}. Retrying...")
                    time.sleep(5)
                else:
                    self.logger.error(f"Failed to evaluate after {max_retries} attempts: {str(e)}")
                    return {"judge_message": "Evaluation failed", "judge_score": 0.0}

