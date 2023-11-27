from typing import Any, Type

from molflux.modelzoo.info import ModelInfo

from physicsml.lightning.model import PhysicsMLModelBase
from physicsml.models.ani.ani_datamodule import ANIDataModule
from physicsml.models.ani.supervised.ani_module import PooledANIModule
from physicsml.models.ani.supervised.default_configs import ANIModelConfig
from physicsml.utils import OptionalDependencyImportError


class ANIModel(PhysicsMLModelBase[ANIModelConfig]):
    def _info(self) -> ModelInfo:
        return ModelInfo(
            model_description="",
            config_description="",
        )

    @property
    def _config_builder(self) -> Type[ANIModelConfig]:
        return ANIModelConfig

    def _instantiate_module(self) -> Any:
        return PooledANIModule(
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
