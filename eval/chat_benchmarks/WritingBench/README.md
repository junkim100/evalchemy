# WritingBench for evalchemy

이 디렉토리는 evalchemy 프레임워크에 통합된 WritingBench 벤치마크 구현을 포함합니다.

## 개요

WritingBench는 LLM의 글쓰기 능력을 평가하는 포괄적인 벤치마크입니다. 1,000개의 실제 글쓰기 쿼리를 통해 6개 주요 도메인과 100개 세부 도메인에 걸쳐 모델을 평가합니다.

### 주요 특징

- **포괄적 평가**: 1,000개의 다양한 글쓰기 작업
- **도메인별 분석**: 6개 주요 도메인 (학술/공학, 금융/비즈니스, 정치/법률, 문학/예술, 교육, 광고/마케팅)
- **인스턴스별 기준**: 각 쿼리마다 5개의 특화된 평가 기준
- **Critic 모델 평가**: Claude 대신 critic 모델을 사용한 효율적인 평가
- **evalchemy 통합**: evalchemy 프레임워크와 완전 호환

## 파일 구조

```
WritingBench/
├── __init__.py              # 모듈 초기화
├── eval_instruct.py         # 메인 벤치마크 클래스
├── evaluator.py             # 평가자 클래스들
├── prompt.py                # 평가 프롬프트 템플릿
├── test_writing_bench.py    # 테스트 스크립트
├── README.md               # 이 파일
└── data/                   # 데이터셋 디렉토리
    ├── benchmark_all.jsonl  # 메인 데이터셋 (1,000개 쿼리)
    └── requirement/         # 요구사항별 서브셋
```

## 사용법

### 기본 사용법

```python
from evalchemy.eval.chat_benchmarks.WritingBench import WritingBenchBenchmark, WritingBenchConfig

# 설정 생성
config = WritingBenchConfig(
    critic_model_path="/path/to/critic/model",
    temperature=0.7,
    max_new_tokens=16000,
    evaluator_model="critic"
)

# 벤치마크 초기화
benchmark = WritingBenchBenchmark(
    config=config,
    debug=False,  # True로 설정하면 10개 샘플만 사용
    logger=your_logger
)

# 평가 실행
results = benchmark.run_benchmark(model)
```

### 설정 옵션

#### WritingBenchConfig 주요 매개변수

- `critic_model_path`: Critic 모델 경로 (필수)
- `do_sample`: 샘플링 활성화 여부 (기본값: False - 결정적 생성)
- `temperature`: 생성 온도 (기본값: 0.7, do_sample=True일 때만 사용)
- `top_p`: Top-p 샘플링 (기본값: 0.8, do_sample=True일 때만 사용)
- `top_k`: Top-k 샘플링 (기본값: 20, do_sample=True일 때만 사용)
- `max_new_tokens`: 최대 토큰 수 (기본값: 2048)
- `evaluator_model`: 평가자 유형 ("critic" 또는 "claude")
- `eval_temperature`: 평가 온도 (기본값: 1.0)
- `eval_top_p`: 평가 top_p (기본값: 0.95)

#### 기본 설정 (결정적 생성)

```python
config = WritingBenchConfig(
    critic_model_path="AQuarterMile/WritingBench-Critic-Model-Qwen-7B",
    do_sample=False,  # 결정적 생성 (기본값)
    evaluator_model="critic"
)
```

#### 샘플링 활성화 설정

```python
config = WritingBenchConfig(
    critic_model_path="AQuarterMile/WritingBench-Critic-Model-Qwen-7B",
    do_sample=True,  # 샘플링 활성화
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    evaluator_model="critic"
)
```

#### Claude 평가자 설정 (대안)

```python
config = WritingBenchConfig(
    evaluator_model="claude",
    claude_api_key="your_api_key",
    claude_url="your_api_url",
    claude_model="claude-3-sonnet-20240229"
)
```

## 평가 메트릭

벤치마크는 다음 메트릭을 반환합니다:

- `overall_score`: 전체 평균 점수
- `num_evaluated`: 평가된 응답 수
- `domain1_*`: 주요 도메인별 점수
- `domain2_*`: 세부 도메인별 점수
- `benchmark_version`: 벤치마크 버전

## 요구사항

### 필수 의존성

- `vllm`: Critic 모델 실행용
- `requests`: Claude API 호출용 (Claude 사용 시)
- `datasets`: 데이터셋 로딩용
- `lm_eval`: evalchemy 프레임워크 통합용

### 데이터셋

WritingBench 데이터셋은 구현 디렉토리 내에 포함되어 있습니다:
```
evalchemy/eval/chat_benchmarks/WritingBench/data/
├── benchmark_all.jsonl      # 1,000개의 평가 쿼리
└── requirement/             # 요구사항별 서브셋
    ├── format/
    ├── length/
    └── style/
```

데이터셋은 자동으로 로드되므로 별도 설정이 필요하지 않습니다.

## 테스트

테스트 스크립트를 실행하여 구현을 검증할 수 있습니다:

```bash
cd /mnt/raid6/hst/wbl
python evalchemy/eval/chat_benchmarks/WritingBench/test_writing_bench.py
```

## 구현 세부사항

### 평가 프로세스

1. **데이터 로딩**: JSONL 파일에서 쿼리와 평가 기준 로드
2. **응답 생성**: 모델을 사용하여 각 쿼리에 대한 응답 생성
3. **평가**: Critic 모델을 사용하여 각 기준별로 응답 평가
4. **점수 집계**: 도메인별 및 전체 점수 계산

### 에러 처리

- 평가 실패 시 기본 낮은 점수 (1점) 할당
- 재시도 로직으로 일시적 오류 처리
- 상세한 로깅으로 디버깅 지원

### 분산 처리

- 다중 GPU 환경에서 자동 작업 분할
- 주 프로세스(rank 0)에서만 결과 집계
- 비주 프로세스에서는 None 반환

## 원본 WritingBench와의 차이점

1. **평가자**: Claude 대신 critic 모델 사용
2. **프레임워크**: evalchemy BaseBenchmark 상속
3. **설정 관리**: dataclass 기반 구조화된 설정
4. **에러 처리**: 강화된 예외 처리 및 로깅
5. **분산 지원**: 다중 GPU 환경 지원

## 문제 해결

### 일반적인 문제

1. **모델 로딩 실패**: critic_model_path가 올바른지 확인
2. **메모리 부족**: tensor_parallel_size 조정
3. **데이터셋 없음**: WritingBench 데이터셋 경로 확인

### 로그 확인

상세한 로그를 위해 로깅 레벨을 DEBUG로 설정:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 기여

이 구현은 원본 WritingBench 논문과 코드베이스를 기반으로 합니다:

```bibtex
@misc{wu2025writingbench,
      title={WritingBench: A Comprehensive Benchmark for Generative Writing}, 
      author={Yuning Wu and Jiahao Mei and Ming Yan and Chenliang Li and Shaopeng Lai and Yuran Ren and Zijia Wang and Ji Zhang and Mengyue Wu and Qin Jin and Fei Huang},
      year={2025},
      url={https://arxiv.org/abs/2503.05244}, 
}
```
