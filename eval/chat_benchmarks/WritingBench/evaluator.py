import json
import logging
import time
import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import requests
from openai import AsyncOpenAI


class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def evaluate(self, query: str, response: str, criteria: Dict[str, Any],
                top_p: float = 0.95, temperature: float = 1.0,
                max_tokens: int = 2048) -> Dict[str, Any]:
        """Evaluate a single response."""
        pass
    
    def basic_success_check(self, response: str) -> bool:
        """Basic check for successful response."""
        if not response or len(response.strip()) == 0:
            self.logger.warning("Empty response received")
            return False
        return True


class GPTEvaluator(BaseEvaluator):
    """GPT model-based evaluator for WritingBench using OpenAI API."""
    
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None,
                 api_base: str = None, system_prompt: str = None,
                 logger: Optional[logging.Logger] = None):
        super().__init__(logger)
        self.model_name = model_name
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Initialize OpenAI client
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=api_base,
            timeout=300.0,
            max_retries=3
        )
        
        self.logger.info(f"Initialized GPT evaluator with model: {model_name}")
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for evaluation."""
        return """You are an expert evaluator with extensive experience in evaluating writing quality and task completion."""
    
    async def _call_gpt_async(self, messages: List[Dict[str, str]], 
                             top_p: float = 0.95, temperature: float = 1.0, 
                             max_tokens: int = 2048) -> str:
        """Make async call to GPT API."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"GPT API call failed: {str(e)}")
            raise
    
    def _call_gpt(self, messages: List[Dict[str, str]], 
                  top_p: float = 0.95, temperature: float = 1.0, 
                  max_tokens: int = 2048) -> str:
        """Synchronous wrapper for GPT API call."""
        try:
            # Run async function in event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._call_gpt_async(messages, top_p, temperature, max_tokens)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in GPT call: {str(e)}")
            raise
    
    def _format_evaluation_prompt(self, query: str, response: str, criteria: Dict[str, Any]) -> str:
        """Format evaluation prompt for GPT using the original WritingBench template."""
        from .prompt import evaluate_prompt

        # Format criteria as string (exactly like original CriticEvaluator)
        if isinstance(criteria, dict):
            criteria_str = "\n".join([f"- {k}: {v}" for k, v in criteria.items()])
        elif isinstance(criteria, list):
            criteria_str = "\n".join([f"- {item}" for item in criteria])
        else:
            criteria_str = str(criteria)

        return evaluate_prompt.format(
            query=query,
            response=response,
            criteria=criteria_str
        )
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse evaluation response using original WritingBench logic."""
        try:
            # 원본 WritingBench와 동일한 파싱 로직
            result = json.loads(response.strip('json|```'))
        except json.JSONDecodeError as e:
            self.logger.warning(f"JSON decode error: {e}, trying eval()")
            try:
                # 원본 WritingBench의 fallback 방식
                result = eval(response.strip('json|```'))
            except Exception as eval_error:
                self.logger.error(f"Both json.loads and eval failed: {eval_error}")
                return {"score": 1, "reason": f"Failed to parse evaluation: {str(eval_error)}"}

        # 원본 WritingBench와 동일한 검증
        valid_score_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        if "score" not in result or "reason" not in result:
            self.logger.error("Missing 'score' or 'reason' in the result")
            return {"score": 1, "reason": "Missing required fields"}

        if result["score"] not in valid_score_values:
            self.logger.error(f"Invalid score value: {result['score']}")
            return {"score": 1, "reason": f"Invalid score: {result['score']}"}

        if not isinstance(result["reason"], str):
            self.logger.error("Reason is not a string")
            return {"score": 1, "reason": "Invalid reason format"}

        return result
    
    def evaluate(self, query: str, response: str, criteria: Dict[str, Any],
                top_p: float = 0.95, temperature: float = 1.0,
                max_tokens: int = 2048) -> Dict[str, Any]:
        """Evaluate a single response using GPT."""
        try:
            # Format evaluation prompt
            eval_prompt = self._format_evaluation_prompt(query, response, criteria)
            
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": eval_prompt}
            ]
            
            eval_response = self._call_gpt(messages, top_p, temperature, max_tokens)
            
            # Parse evaluation response
            return self._parse_evaluation_response(eval_response)
            
        except Exception as e:
            self.logger.error(f"Error in GPT evaluation: {str(e)}")
            return {"score": 1, "reason": f"GPT evaluation failed: {str(e)}"}
    
    def evaluate_batch(self, evaluations: List[Dict[str, Any]],
                      top_p: float = 0.95, temperature: float = 1.0,
                      max_tokens: int = 2048) -> List[Dict[str, Any]]:
        """Evaluate multiple responses using GPT in batch."""
        try:
            # Run batch evaluation asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(
                    self._evaluate_batch_async(evaluations, top_p, temperature, max_tokens)
                )
                return result
            finally:
                loop.close()
        except Exception as e:
            self.logger.error(f"Error in GPT batch evaluation: {str(e)}")
            # Return error results for all evaluations
            return [{"score": 1, "reason": f"GPT batch evaluation failed: {str(e)}"} 
                   for _ in evaluations]
    
    async def _evaluate_batch_async(self, evaluations: List[Dict[str, Any]],
                                   top_p: float = 0.95, temperature: float = 1.0,
                                   max_tokens: int = 2048) -> List[Dict[str, Any]]:
        """Async batch evaluation using GPT."""
        async def evaluate_single_async(eval_item):
            try:
                eval_prompt = self._format_evaluation_prompt(
                    eval_item["query"],
                    eval_item["response"],
                    eval_item["criteria"]
                )
                
                messages = [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": eval_prompt}
                ]
                
                eval_response = await self._call_gpt_async(messages, top_p, temperature, max_tokens)
                
                # Parse evaluation response
                return self._parse_evaluation_response(eval_response)
                
            except Exception as e:
                self.logger.error(f"Error in single GPT evaluation: {str(e)}")
                return {"score": 1, "reason": f"GPT evaluation failed: {str(e)}"}
        
        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(8)  # Limit to 8 concurrent requests
        
        async def bounded_evaluate(eval_item):
            async with semaphore:
                return await evaluate_single_async(eval_item)
        
        # Execute all evaluations concurrently
        tasks = [bounded_evaluate(eval_item) for eval_item in evaluations]
        results = await asyncio.gather(*tasks)
        
        return results
