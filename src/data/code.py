"""Code datasets for training: CodeAlpaca, Magicoder, etc."""

import re
from typing import Iterator, Optional, Any

from .base import BaseDataset, SFTExample, RLPrompt
from .templates import get_template


class CodeAlpacaDataset(BaseDataset):
    """
    CodeAlpaca: Code instruction following dataset.
    
    ~20K code generation examples.
    Good for basic code generation SFT.
    """
    
    def __init__(
        self,
        split: str = "train",
        template: str = "code",
        max_samples: Optional[int] = None,
    ):
        """
        Initialize CodeAlpaca dataset.
        
        Args:
            split: Dataset split
            template: Prompt template
            max_samples: Maximum samples
        """
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        
        dataset = load_dataset("sahil2801/CodeAlpaca-20k", split=split)
        self._data = list(dataset)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "codealpaca"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for i, item in enumerate(self._data):
            instruction = item.get("instruction", item.get("prompt", ""))
            inp = item.get("input", "")
            output = item.get("output", item.get("completion", ""))
            
            # Combine instruction and input
            if inp:
                full_instruction = f"{instruction}\n\nInput: {inp}"
            else:
                full_instruction = instruction
            
            prompt = self.template.format_prompt(full_instruction)
            response = self.template.format_response(output)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"codealpaca_{i}",
                metadata={"instruction": instruction},
            )


class MagicoderDataset(BaseDataset):
    """
    Magicoder-OSS-Instruct: High-quality code generation dataset.
    
    ~75K examples with code problems and solutions.
    """
    
    def __init__(
        self,
        split: str = "train",
        template: str = "code",
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        
        dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split=split)
        self._data = list(dataset)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "magicoder"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for i, item in enumerate(self._data):
            problem = item.get("problem", "")
            solution = item.get("solution", "")
            
            prompt = self.template.format_prompt(problem)
            response = self.template.format_response(solution)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"magicoder_{i}",
                metadata={},
            )


class HumanEvalDataset(BaseDataset):
    """
    HumanEval: Hand-written code generation benchmark.
    
    164 Python problems with test cases.
    Used for evaluation, not training.
    """
    
    def __init__(
        self,
        template: str = "code",
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset
        
        self.template = get_template(template)
        
        dataset = load_dataset("openai_humaneval", split="test")
        self._data = list(dataset)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "humaneval"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for item in self._data:
            task_id = item["task_id"]
            func_prompt = item["prompt"]  # Function signature + docstring
            canonical = item.get("canonical_solution", "")
            
            prompt = self.template.format_prompt(func_prompt)
            response = self.template.format_response(canonical)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=task_id,
                metadata={
                    "entry_point": item["entry_point"],
                    "test": item["test"],
                },
            )
    
    def get_rl_prompts(self, limit: Optional[int] = None) -> list[RLPrompt]:
        """Get prompts with test cases for RL."""
        examples = list(self)
        if limit:
            examples = examples[:limit]
        
        return [
            RLPrompt(
                prompt=ex.prompt,
                id=ex.id,
                metadata={
                    "entry_point": ex.metadata["entry_point"],
                    "tests": [ex.metadata["test"]],
                },
            )
            for ex in examples
        ]


class MBPPDataset(BaseDataset):
    """
    MBPP: Mostly Basic Python Programming.
    
    ~1000 Python problems with test cases.
    """
    
    def __init__(
        self,
        split: str = "test",
        template: str = "code",
        max_samples: Optional[int] = None,
    ):
        from datasets import load_dataset
        
        self.split = split
        self.template = get_template(template)
        
        try:
            dataset = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
        except Exception:
            dataset = load_dataset("google-research-datasets/mbpp", "full", split=split)
        
        self._data = list(dataset)
        
        if max_samples:
            self._data = self._data[:max_samples]
    
    @property
    def name(self) -> str:
        return "mbpp"
    
    def __len__(self) -> int:
        return len(self._data)
    
    def __iter__(self) -> Iterator[SFTExample]:
        for item in self._data:
            task_id = item.get("task_id", 0)
            text = item.get("text", item.get("prompt", ""))
            code = item.get("code", "")
            
            prompt = self.template.format_prompt(text)
            response = self.template.format_response(code)
            
            yield SFTExample(
                prompt=prompt,
                response=response,
                id=f"mbpp_{task_id}",
                metadata={
                    "test_list": item.get("test_list", []),
                },
            )
    
    def get_rl_prompts(self, limit: Optional[int] = None) -> list[RLPrompt]:
        """Get prompts with test cases for RL."""
        examples = list(self)
        if limit:
            examples = examples[:limit]
        
        return [
            RLPrompt(
                prompt=ex.prompt,
                id=ex.id,
                metadata={"tests": ex.metadata["test_list"]},
            )
            for ex in examples
        ]


# ============================================================================
# Code extraction utilities
# ============================================================================

def extract_code(response: str) -> str:
    """
    Extract code from model response.
    
    Handles:
    - Markdown code blocks
    - Function definitions
    - Raw code
    """
    # Try markdown code block
    code_match = re.search(r'```(?:python)?\n(.+?)```', response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    
    # Try to find function definition
    func_match = re.search(r'(def .+?)(?:\n\n|\Z)', response, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()
    
    # Return as-is
    return response.strip()


def execute_code_with_tests(
    code: str,
    tests: list[str],
    timeout: float = 5.0,
) -> dict:
    """
    Execute code against test cases.
    
    Args:
        code: Code to execute
        tests: List of test assertions
        timeout: Execution timeout
        
    Returns:
        Dict with passed, total, error fields
    """
    from multiprocess import Process, Queue
    
    def run_tests(code: str, tests: list, queue: Queue):
        passed = 0
        total = len(tests)
        errors = []
        
        try:
            namespace = {}
            exec(code, namespace)
            
            for test in tests:
                try:
                    exec(test, namespace)
                    passed += 1
                except AssertionError as e:
                    errors.append(f"AssertionError: {e}")
                except Exception as e:
                    errors.append(f"{type(e).__name__}: {e}")
            
            queue.put({
                "passed": passed,
                "total": total,
                "all_passed": passed == total,
                "errors": errors,
            })
        except Exception as e:
            queue.put({
                "passed": 0,
                "total": total,
                "all_passed": False,
                "error": f"Execution error: {e}",
            })
    
    queue = Queue()
    process = Process(target=run_tests, args=(code, tests, queue))
    process.start()
    process.join(timeout=timeout)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return {
            "passed": 0,
            "total": len(tests),
            "all_passed": False,
            "error": "timeout",
        }
    
    try:
        return queue.get_nowait()
    except Exception:
        return {
            "passed": 0,
            "total": len(tests),
            "all_passed": False,
            "error": "queue_error",
        }
