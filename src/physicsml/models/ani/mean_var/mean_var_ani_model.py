from typing import Any

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model_uncertainty import PhysicsMLUncertaintyModelBase
from physicsml.models.ani.ani_datamodule import ANIDataModule
from physicsml.models.ani.mean_var.default_configs import MeanVarANIModelConfig
from physicsml.models.ani.mean_var.mean_var_ani_module import PooledMeanVarANIModule
from physicsml.utils import OptionalDependencyImportError


class MeanVarANIModel(PhysicsMLUncertaintyModelBase[MeanVarANIModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> type[MeanVarANIModelConfig]:
        return MeanVarANIModelConfig

    def _instantiate_module(self) -> Any:
        return PooledMeanVarANIModule(
            model_config=self.model_config,
        )

    @property
    def _datamodule_builder(self) -> type[ANIDataModule]:  # type: ignore
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
