"""
Typed Pydantic configuration for Parkinson's Disease reproducible research pipeline.
Conforms to Phase 2.1 of the Consolidated Roadmap.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ParkinsonsPipelineConfig(BaseModel):
    """
    Pydantic configuration for Parkinson's research pipeline.
    Guarantees strict schema validation, type safety, and reproducibility.
    """

    data_path: Optional[str] = Field(
        default=None,
        description="Path to CSV dataset or directory containing PPMI/SBR data",
    )
    target_column: str = Field(
        default="motor_progression_label",
        description="Independent clinical endpoint (e.g. delta MDS-UPDRS Part III >= 5)",
    )
    group_column: str = Field(
        default="PATNO",
        description="Participant ID grouping column to prevent longitudinal visit leakage",
    )
    output_dir: str = Field(
        default="results/ppmi_reproducibility_benchmark",
        description="Destination directory for metrics, model checkpoints, and reports",
    )
    n_folds: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of StratifiedGroupKFold cross-validation folds",
    )
    random_state: int = Field(
        default=42,
        description="Pseudo-random seed across NumPy, PyTorch, and scikit-learn; hardware-level reproducibility requires separate verification",
    )
    n_bootstraps: int = Field(
        default=1000,
        ge=100,
        description="Number of non-parametric bootstrap iterations for confidence intervals",
    )
    alpha: float = Field(
        default=0.05,
        ge=0.001,
        le=0.5,
        description="Significance level for confidence interval (alpha=0.05 corresponds to 95% CI)",
    )
    epochs: int = Field(
        default=25,
        ge=1,
        description="Number of training epochs for PyTorch MLP classifier",
    )
    batch_size: int = Field(
        default=16,
        ge=1,
        description="Batch size for neural network training",
    )
    learning_rate: float = Field(
        default=0.001,
        gt=0.0,
        description="Optimizer initial learning rate",
    )
    dataset_choice: str = Field(
        default="vikasukani",
        description="Kaggle or LONI dataset variant fallback",
    )
    track_experiments: bool = Field(
        default=True,
        description="Enable SQLite/JSON experiment tracking and model lineage",
    )
    experiment_name: str = Field(
        default="parkinsons_progression",
        description="Experiment group label for lineage tracking",
    )

    class Config:
        arbitrary_types_allowed = True
