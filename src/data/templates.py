"""Prompt templates for different tasks and models."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PromptTemplate:
    """Template for formatting prompts and responses."""
    system: Optional[str] = None
    user_prefix: str = ""
    user_suffix: str = ""
    assistant_prefix: str = ""
    assistant_suffix: str = ""
    
    def format_prompt(self, user_content: str) -> str:
        """Format user content into a prompt."""
        parts = []
        if self.system:
            parts.append(self.system)
        parts.append(f"{self.user_prefix}{user_content}{self.user_suffix}")
        parts.append(self.assistant_prefix)
        return "".join(parts)
    
    def format_response(self, response: str) -> str:
        """Format assistant response."""
        return f"{response}{self.assistant_suffix}"
    
    def format_full(self, user_content: str, response: str) -> str:
        """Format full prompt + response for training."""
        return self.format_prompt(user_content) + self.format_response(response)


# ============================================================================
# Chat Templates (for instruction-tuned models)
# ============================================================================

CHATML_TEMPLATE = PromptTemplate(
    system="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n",
    user_prefix="<|im_start|>user\n",
    user_suffix="<|im_end|>\n",
    assistant_prefix="<|im_start|>assistant\n",
    assistant_suffix="<|im_end|>\n",
)

LLAMA_CHAT_TEMPLATE = PromptTemplate(
    system="<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>",
    user_prefix="<|start_header_id|>user<|end_header_id|>\n\n",
    user_suffix="<|eot_id|>",
    assistant_prefix="<|start_header_id|>assistant<|end_header_id|>\n\n",
    assistant_suffix="<|eot_id|>",
)

# Simple template (no special tokens, for base models)
SIMPLE_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="### Question:\n",
    user_suffix="\n\n",
    assistant_prefix="### Answer:\n",
    assistant_suffix="\n",
)


# ============================================================================
# Task-Specific Templates
# ============================================================================

# Math with Chain-of-Thought
MATH_COT_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="Solve this math problem step by step. Show your reasoning, then provide the final numerical answer.\n\nProblem: ",
    user_suffix="\n\n",
    assistant_prefix="Solution:\n",
    assistant_suffix="\n",
)

# Math without CoT (direct answer)
MATH_DIRECT_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="Solve this math problem and provide only the numerical answer.\n\nProblem: ",
    user_suffix="\n\n",
    assistant_prefix="Answer: ",
    assistant_suffix="\n",
)

# Code generation
CODE_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="Complete the following Python function. Only output the function body, no explanations.\n\n",
    user_suffix="\n",
    assistant_prefix="",
    assistant_suffix="\n",
)

# Code with explanation
CODE_EXPLAINED_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="Complete the following Python function. First explain your approach, then provide the code.\n\n",
    user_suffix="\n\n",
    assistant_prefix="",
    assistant_suffix="\n",
)

# Generic instruction following
INSTRUCTION_TEMPLATE = PromptTemplate(
    system=None,
    user_prefix="### Instruction:\n",
    user_suffix="\n\n",
    assistant_prefix="### Response:\n",
    assistant_suffix="\n",
)


# ============================================================================
# Template Registry
# ============================================================================

TEMPLATES = {
    # Chat templates
    "chatml": CHATML_TEMPLATE,
    "llama_chat": LLAMA_CHAT_TEMPLATE,
    "simple": SIMPLE_TEMPLATE,
    
    # Task templates
    "math_cot": MATH_COT_TEMPLATE,
    "math_direct": MATH_DIRECT_TEMPLATE,
    "code": CODE_TEMPLATE,
    "code_explained": CODE_EXPLAINED_TEMPLATE,
    "instruction": INSTRUCTION_TEMPLATE,
    
    # Default
    "default": SIMPLE_TEMPLATE,
}


def get_template(name: str) -> PromptTemplate:
    """
    Get prompt template by name.
    
    Args:
        name: Template name
        
    Returns:
        PromptTemplate instance
    """
    if name not in TEMPLATES:
        available = ", ".join(TEMPLATES.keys())
        raise ValueError(f"Unknown template: {name}. Available: {available}")
    return TEMPLATES[name]


def detect_template_for_model(model_name: str) -> PromptTemplate:
    """
    Auto-detect appropriate template based on model name.
    
    Args:
        model_name: HuggingFace model name
        
    Returns:
        Appropriate PromptTemplate
    """
    model_lower = model_name.lower()
    
    if "qwen" in model_lower:
        return CHATML_TEMPLATE
    elif "llama" in model_lower and ("instruct" in model_lower or "chat" in model_lower):
        return LLAMA_CHAT_TEMPLATE
    elif "deepseek" in model_lower:
        return CHATML_TEMPLATE
    elif "mistral" in model_lower and "instruct" in model_lower:
        return INSTRUCTION_TEMPLATE
    else:
        return SIMPLE_TEMPLATE
