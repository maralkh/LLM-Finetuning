"""Outcome-based rewards for math and code tasks."""

from typing import Optional, Callable, Any
import re

from .base import BaseReward, BinaryReward, RewardOutput


class MathCorrectnessReward(BinaryReward):
    """
    Reward based on final answer correctness for math problems.
    
    Extracts answer from response and compares to gold.
    """
    
    def __init__(
        self,
        positive: float = 1.0,
        negative: float = 0.0,
        partial_credit: bool = False,
    ):
        """
        Initialize math reward.
        
        Args:
            positive: Reward for correct answer
            negative: Reward for incorrect answer
            partial_credit: If True, give partial credit for close answers
        """
        super().__init__(positive, negative)
        self.partial_credit = partial_credit
    
    @property
    def name(self) -> str:
        return "math_correctness"
    
    def is_correct(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Check if math answer is correct."""
        if metadata is None or "gold_answer" not in metadata:
            return False
        
        gold = metadata["gold_answer"]
        predicted = self.extract_answer(response)
        
        return self.check_answer(predicted, gold)
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute reward with optional partial credit."""
        if metadata is None or "gold_answer" not in metadata:
            return RewardOutput(reward=self.negative, info={"error": "no_gold_answer"})
        
        gold = metadata["gold_answer"]
        predicted = self.extract_answer(response)
        
        if predicted is None:
            return RewardOutput(
                reward=self.negative,
                info={"correct": False, "predicted": None, "gold": gold},
            )
        
        correct = self.check_answer(predicted, gold)
        
        if correct:
            reward = self.positive
        elif self.partial_credit:
            reward = self._compute_partial_credit(predicted, gold)
        else:
            reward = self.negative
        
        return RewardOutput(
            reward=reward,
            info={"correct": correct, "predicted": predicted, "gold": gold},
        )
    
    def extract_answer(self, response: str) -> Optional[Any]:
        """Extract numerical answer from response."""
        patterns = [
            r"[Aa]nswer[:\s]*\$?(-?\d+(?:\.\d+)?)",
            r"####\s*(-?\d+(?:\.\d+)?)",
            r"\\boxed\{(-?\d+(?:\.\d+)?)\}",
            r"=\s*\$?(-?\d+(?:\.\d+)?)\s*$",
            r"(?:is|equals?)\s*\$?(-?\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.MULTILINE)
            if match:
                try:
                    val = match.group(1)
                    return float(val) if "." in val else int(val)
                except ValueError:
                    continue
        
        # Last number in response
        numbers = re.findall(r"(-?\d+(?:\.\d+)?)", response)
        if numbers:
            try:
                val = numbers[-1]
                return float(val) if "." in val else int(val)
            except ValueError:
                pass
        
        return None
    
    def check_answer(self, predicted: Any, gold: Any, tolerance: float = 1e-6) -> bool:
        """Check if answers match."""
        if predicted is None or gold is None:
            return False
        
        try:
            pred_num = float(predicted)
            gold_num = float(gold)
            return abs(pred_num - gold_num) < tolerance
        except (ValueError, TypeError):
            return str(predicted).strip() == str(gold).strip()
    
    def _compute_partial_credit(self, predicted: Any, gold: Any) -> float:
        """Compute partial credit for close answers."""
        try:
            pred_num = float(predicted)
            gold_num = float(gold)
            
            if gold_num == 0:
                return self.negative
            
            # Relative error
            rel_error = abs(pred_num - gold_num) / abs(gold_num)
            
            if rel_error < 0.01:  # Within 1%
                return self.positive * 0.8
            elif rel_error < 0.1:  # Within 10%
                return self.positive * 0.5
            else:
                return self.negative
        except (ValueError, TypeError):
            return self.negative


class CodeExecutionReward(BinaryReward):
    """
    Reward based on code execution against test cases.
    
    Executes generated code and checks if tests pass.
    """
    
    def __init__(
        self,
        positive: float = 1.0,
        negative: float = 0.0,
        timeout: float = 5.0,
        partial_credit: bool = True,  # Credit for passing some tests
    ):
        """
        Initialize code reward.
        
        Args:
            positive: Reward for all tests passing
            negative: Reward for all tests failing
            timeout: Execution timeout in seconds
            partial_credit: Give credit proportional to tests passed
        """
        super().__init__(positive, negative)
        self.timeout = timeout
        self.partial_credit = partial_credit
    
    @property
    def name(self) -> str:
        return "code_execution"
    
    def is_correct(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> bool:
        """Check if all tests pass."""
        result = self._execute_tests(response, metadata)
        return result["all_passed"]
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute reward based on test execution."""
        result = self._execute_tests(response, metadata)
        
        if result["all_passed"]:
            reward = self.positive
        elif self.partial_credit and result["total"] > 0:
            # Proportional reward
            pass_rate = result["passed"] / result["total"]
            reward = self.negative + (self.positive - self.negative) * pass_rate
        else:
            reward = self.negative
        
        return RewardOutput(
            reward=reward,
            info={
                "correct": result["all_passed"],
                "passed": result["passed"],
                "total": result["total"],
                "error": result.get("error"),
            },
        )
    
    def _execute_tests(self, code: str, metadata: Optional[dict]) -> dict:
        """Execute code against tests."""
        from multiprocess import Process, Queue
        
        if metadata is None:
            return {"all_passed": False, "passed": 0, "total": 0, "error": "no_metadata"}
        
        tests = metadata.get("tests", metadata.get("test_list", []))
        if not tests:
            return {"all_passed": False, "passed": 0, "total": 0, "error": "no_tests"}
        
        # Extract code from response
        code = self._extract_code(code)
        
        def run_tests(code: str, tests: list, queue: Queue):
            passed = 0
            total = len(tests)
            
            try:
                namespace = {}
                exec(code, namespace)
                
                for test in tests:
                    try:
                        exec(test, namespace)
                        passed += 1
                    except Exception:
                        pass
                
                queue.put({"passed": passed, "total": total, "all_passed": passed == total})
            except Exception as e:
                queue.put({"passed": 0, "total": total, "all_passed": False, "error": str(e)})
        
        queue = Queue()
        process = Process(target=run_tests, args=(code, tests, queue))
        process.start()
        process.join(timeout=self.timeout)
        
        if process.is_alive():
            process.terminate()
            process.join()
            return {"all_passed": False, "passed": 0, "total": len(tests), "error": "timeout"}
        
        try:
            return queue.get_nowait()
        except Exception:
            return {"all_passed": False, "passed": 0, "total": len(tests), "error": "queue_error"}
    
    def _extract_code(self, response: str) -> str:
        """Extract code from response."""
        # Try to find code block
        code_match = re.search(r'```(?:python)?\n(.+?)```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Try to find function definition
        func_match = re.search(r'(def .+?)(?:\n\n|\Z)', response, re.DOTALL)
        if func_match:
            return func_match.group(1).strip()
        
        return response.strip()


class CustomReward(BaseReward):
    """
    Wrapper for custom reward functions.
    
    Allows using any callable as a reward function.
    """
    
    def __init__(
        self,
        reward_fn: Callable[[str, str, Optional[dict]], float],
        name: str = "custom",
    ):
        """
        Initialize custom reward.
        
        Args:
            reward_fn: Function (prompt, response, metadata) -> reward
            name: Reward function name
        """
        self._reward_fn = reward_fn
        self._name = name
    
    @property
    def name(self) -> str:
        return self._name
    
    def compute(
        self,
        prompt: str,
        response: str,
        metadata: Optional[dict] = None,
    ) -> RewardOutput:
        """Compute reward using custom function."""
        reward = self._reward_fn(prompt, response, metadata)
        return RewardOutput(reward=reward)
