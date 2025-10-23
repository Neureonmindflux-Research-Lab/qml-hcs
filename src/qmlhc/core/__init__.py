
# Public re-exports for core contracts and orchestration.


from .types import (
    Scalar,
    Array,
    TensorLike,
    GradientKind,
    Capabilities,
    RunInfo,
    State,
    FutureSet,
    SelectionPolicy,
    RiskFunctional,
    QuantumBackend as QuantumBackendProtocol,
    ProjectionPolicy,
    HypercausalNode,
    LossFn,
)
from .backend import BackendConfig, QuantumBackend
from .model import HCModel, ModelConfig
from .registry import (
    BackendRegistry,
    register_backend,
    create_backend,
    backend_exists,
    list_backends,
)

__all__ = [
    "Scalar",
    "Array",
    "TensorLike",
    "GradientKind",
    "Capabilities",
    "RunInfo",
    "State",
    "FutureSet",
    "SelectionPolicy",
    "RiskFunctional",
    "QuantumBackendProtocol",
    "ProjectionPolicy",
    "HypercausalNode",
    "LossFn",
    "BackendConfig",
    "QuantumBackend",
    "HCModel",
    "ModelConfig",
    "BackendRegistry",
    "register_backend",
    "create_backend",
    "backend_exists",
    "list_backends",
]
