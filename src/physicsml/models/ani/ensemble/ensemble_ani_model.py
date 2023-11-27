from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.ani.ani_datamodule import ANIDataModule
from physicsml.models.ani.ensemble.default_configs import EnsembleANIModelConfig
from physicsml.models.ani.ensemble.ensemble_ani_module import PooledEnsembleANIModule
from physicsml.utils import OptionalDependencyImportError


class EnsembleANIModel(PhysicsMLUncertaintyModelBase[EnsembleANIModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[EnsembleANIModelConfig]:
        return EnsembleANIModelConfig

    def _instantiate_module(self) -> Any:
        return PooledEnsembleANIModule(
            model_config=self.model_config,
        )

    @property
    def _datamodule_builder(self) -> Type[ANIDataModule]:  # type: ignore
        return ANIDataModule

    def to_openmm(self, **kwargs: Any) -> Any:
        from physicsml.plugins.openmm.openmm_ani import OpenMMANI

        return OpenMMANI(**kwargs)

    def to_ase(self, **kwargs: Any) -> Any:
        try:
            from physicsml.plugins.ase.ase_ani import ANIASECalculator
        except ImportError as err:
            raise OptionalDependencyImportError("ASE", "ase") from err
        else:
            return ANIASECalculator(**kwargs)
