```python
"""
EconoCausal Core Package

Double Machine Learning (DML) for:
- Causal Inference
- Uplift Modeling
- Dynamic Pricing
- Price Elasticity Estimation
"""

from .data_generator import (
    generate_observational_pricing_dataset,
    CustomerDataGenerator,
)

from .dml_engine import DoubleMachineLearningEngine

from .benchmark import ModelComparator

from .optimizer import DynamicPricingOptimizer

from .refutation import CausalRefutationSuite


__all__ = [
    "generate_observational_pricing_dataset",
    "CustomerDataGenerator",
    "DoubleMachineLearningEngine",
    "ModelComparator",
    "DynamicPricingOptimizer",
    "CausalRefutationSuite",
]

__version__ = "1.0.0"
__author__ = "EconoCausal Team"
```
