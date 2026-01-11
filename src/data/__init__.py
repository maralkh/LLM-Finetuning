"""Training datasets and data utilities."""

from .base import (
    BaseDataset,
    BasePreferenceDataset,
    SFTExample,
    PreferenceExample,
    RLPrompt,
    DatasetConfig,
)
from .templates import (
    PromptTemplate,
    get_template,
    detect_template_for_model,
    TEMPLATES,
)
from .math import (
    GSM8KDataset,
    MATHDataset,
    MetaMathQADataset,
    extract_math_answer,
    check_math_answer,
)
from .code import (
    CodeAlpacaDataset,
    MagicoderDataset,
    HumanEvalDataset,
    MBPPDataset,
    extract_code,
    execute_code_with_tests,
)
from .preference import (
    PreferenceDataset,
    HFPreferenceDataset,
    generate_preferences_rejection_sampling,
    generate_preferences_best_of_n,
    filter_preferences,
)
from .collators import (
    DataCollatorForSFT,
    DataCollatorForDPO,
    DataCollatorForRL,
)


def load_dataset(config: dict) -> BaseDataset:
    """
    Load dataset from configuration.
    
    Args:
        config: Dict with 'name' and dataset-specific params
        
    Returns:
        Initialized dataset
    """
    name = config.get("name", "").lower()
    
    # Math datasets
    if name == "gsm8k":
        return GSM8KDataset(
            split=config.get("split", "train"),
            template=config.get("template", "math_cot"),
            max_samples=config.get("max_samples"),
        )
    elif name == "math":
        return MATHDataset(
            split=config.get("split", "train"),
            template=config.get("template", "math_cot"),
            max_samples=config.get("max_samples"),
            levels=config.get("levels"),
            subjects=config.get("subjects"),
        )
    elif name == "metamathqa" or name == "metamath":
        return MetaMathQADataset(
            split=config.get("split", "train"),
            template=config.get("template", "math_cot"),
            max_samples=config.get("max_samples"),
            source_filter=config.get("source_filter"),
        )
    
    # Code datasets
    elif name == "codealpaca":
        return CodeAlpacaDataset(
            split=config.get("split", "train"),
            template=config.get("template", "code"),
            max_samples=config.get("max_samples"),
        )
    elif name == "magicoder":
        return MagicoderDataset(
            split=config.get("split", "train"),
            template=config.get("template", "code"),
            max_samples=config.get("max_samples"),
        )
    elif name == "humaneval":
        return HumanEvalDataset(
            template=config.get("template", "code"),
            max_samples=config.get("max_samples"),
        )
    elif name == "mbpp":
        return MBPPDataset(
            split=config.get("split", "test"),
            template=config.get("template", "code"),
            max_samples=config.get("max_samples"),
        )
    
    # Preference datasets
    elif name == "hf_preference":
        return HFPreferenceDataset(
            dataset_name=config["dataset_name"],
            split=config.get("split", "train"),
            max_samples=config.get("max_samples"),
            prompt_column=config.get("prompt_column", "prompt"),
            chosen_column=config.get("chosen_column", "chosen"),
            rejected_column=config.get("rejected_column", "rejected"),
        )
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


__all__ = [
    # Base
    "BaseDataset",
    "BasePreferenceDataset",
    "SFTExample",
    "PreferenceExample",
    "RLPrompt",
    "DatasetConfig",
    # Templates
    "PromptTemplate",
    "get_template",
    "detect_template_for_model",
    "TEMPLATES",
    # Math datasets
    "GSM8KDataset",
    "MATHDataset",
    "MetaMathQADataset",
    "extract_math_answer",
    "check_math_answer",
    # Code datasets
    "CodeAlpacaDataset",
    "MagicoderDataset",
    "HumanEvalDataset",
    "MBPPDataset",
    "extract_code",
    "execute_code_with_tests",
    # Preference
    "PreferenceDataset",
    "HFPreferenceDataset",
    "generate_preferences_rejection_sampling",
    "generate_preferences_best_of_n",
    "filter_preferences",
    # Collators
    "DataCollatorForSFT",
    "DataCollatorForDPO",
    "DataCollatorForRL",
    # Factory
    "load_dataset",
]
