"""Evaluation utilities for quick assessment during training."""

from typing import Optional, Any
from dataclasses import dataclass
import torch
from tqdm import tqdm


@dataclass
class EvalMetrics:
    """Evaluation metrics."""
    accuracy: float
    total: int
    correct: int
    details: Optional[list] = None


def evaluate_math_accuracy(
    model,
    tokenizer,
    prompts: list[str],
    gold_answers: list[Any],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    batch_size: int = 4,
    show_progress: bool = True,
    extract_fn=None,
    check_fn=None,
) -> EvalMetrics:
    """
    Evaluate math accuracy by generating and checking answers.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer
        prompts: List of prompts
        gold_answers: List of gold answers
        max_new_tokens: Max tokens to generate
        temperature: Generation temperature (0 = greedy)
        batch_size: Batch size for generation
        show_progress: Show progress bar
        extract_fn: Function to extract answer from response
        check_fn: Function to check if answer is correct
        
    Returns:
        EvalMetrics with accuracy
    """
    from ..data.math import extract_math_answer, check_math_answer
    
    if extract_fn is None:
        extract_fn = extract_math_answer
    if check_fn is None:
        check_fn = check_math_answer
    
    model.eval()
    correct = 0
    total = len(prompts)
    details = []
    
    # Process in batches
    iterator = range(0, total, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Evaluating")
    
    for i in iterator:
        batch_prompts = prompts[i:i + batch_size]
        batch_golds = gold_answers[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode and check
        input_len = inputs["input_ids"].shape[1]
        
        for j, (output, gold) in enumerate(zip(outputs, batch_golds)):
            response = tokenizer.decode(output[input_len:], skip_special_tokens=True)
            predicted = extract_fn(response)
            is_correct = check_fn(predicted, gold)
            
            if is_correct:
                correct += 1
            
            details.append({
                "prompt": batch_prompts[j][:100] + "...",
                "predicted": predicted,
                "gold": gold,
                "correct": is_correct,
            })
    
    model.train()
    
    return EvalMetrics(
        accuracy=correct / total if total > 0 else 0.0,
        total=total,
        correct=correct,
        details=details,
    )


def evaluate_from_dataset(
    model,
    tokenizer,
    dataset,
    max_samples: int = 200,
    **kwargs,
) -> EvalMetrics:
    """
    Evaluate on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset: Dataset with get_rl_prompts() method
        max_samples: Maximum samples to evaluate
        **kwargs: Additional args for evaluate_math_accuracy
        
    Returns:
        EvalMetrics
    """
    rl_prompts = dataset.get_rl_prompts(limit=max_samples)
    
    prompts = [p.prompt for p in rl_prompts]
    gold_answers = [p.metadata.get("gold_answer") for p in rl_prompts]
    
    return evaluate_math_accuracy(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        gold_answers=gold_answers,
        **kwargs,
    )


def compute_response_stats(
    responses: list[str],
    tokenizer,
) -> dict:
    """
    Compute statistics about generated responses.
    
    Args:
        responses: List of generated responses
        tokenizer: Tokenizer for token counting
        
    Returns:
        Dict with stats
    """
    if not responses:
        return {}
    
    # Token lengths
    token_lengths = [
        len(tokenizer.encode(r, add_special_tokens=False))
        for r in responses
    ]
    
    # Word lengths
    word_lengths = [len(r.split()) for r in responses]
    
    import numpy as np
    
    return {
        "num_responses": len(responses),
        "avg_tokens": np.mean(token_lengths),
        "max_tokens": max(token_lengths),
        "min_tokens": min(token_lengths),
        "avg_words": np.mean(word_lengths),
        "empty_responses": sum(1 for r in responses if not r.strip()),
    }
