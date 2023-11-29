# type: ignore
import logging
from typing import Any, Iterable, Optional

try:
    import openmm
except ImportError as exp:
    from physicsml.utils import OptionalCondaDependencyImportError

    raise OptionalCondaDependencyImportError("OPENMM", "openmm") from exp
try:
    from openmmtorch import TorchForce
except ImportError as exp:
    from physicsml.utils import OptionalCondaDependencyImportError

    raise OptionalCondaDependencyImportError("OPENMM-TORCH", "openmm-torch") from exp

from openmmml.mlpotential import (
    MLPotential,
    MLPotentialImpl,
    MLPotentialImplFactory,
)

from physicsml.plugins.openmm.load import to_openmm_torchscript

logger = logging.getLogger(__name__)


class PhysicsMLPotentialImplFactory(MLPotentialImplFactory):
    def createImpl(self, name: str, **kwargs: Any) -> MLPotentialImpl:
        return PhysicsMLPotentialImpl(**kwargs)


class PhysicsMLPotentialImpl(MLPotentialImpl):
    """
    Implements the physicsml potential in openMM via openmm-ml, using TorchForce to add the NNP to an openMM system.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        repo_url: Optional[str] = None,
        rev: Optional[str] = None,
        model_path_in_repo: Optional[str] = None,
        y_output: Optional[str] = None,
        output_scaling: Optional[float] = None,
        position_scaling: Optional[float] = None,
        device: str = "cpu",
        precision: str = "32",
    ) -> None:
        """

        Args:
            model_path (Optional[str], optional): Path to disk of a physicsml model. Must specify either this or model or repo_url and rev. Defaults to None.
            repo_url (Optional[str], optional): A dvc repo url of physicsml experiments. Must specify either this and rev or model or model_path. Defaults to None.
            rev (Optional[str], optional): The dvc commit hash of a physicsml model experiment. Must specify either this and repo_url or model or model_path. Defaults to None.
            model_path_in_repo (str, optional): The path to the physicsml model in the experiment. Defaults to "pipelines/migrate_models/model".
            y_output (Optional[str], optional): The output of the model (from its y_features). Defaults to 'y_graph_scalars'. Defaults to None.
            output_scaling (Optional[float], optional): The scaling of the output of the model (for changing units). Defaults to None.
            position_scaling (Optional[float], optional): The scaling of the positions input to the model (for changing units). Defaults to None.
            device (str, optional): The device to run inference on (either cpu or cuda). Defaults to "cpu".
            precision (str, optional): The precision to use (32 or 64). Defaults to "32".
        """
        super().__init__()

        assert (model_path is not None) ^ (
            (repo_url is not None) and (rev is not None)
        ), ValueError(
            "Must specify either 'model_path' or 'repo_url' and 'rev' (but not both).",
        )

        self.model_config = {
            "y_output": y_output,
            "output_scaling": output_scaling,
            "position_scaling": position_scaling,
            "device": device,
            "precision": precision,
        }

        if model_path is not None:
            self.model_config["model_path"] = model_path
        else:
            self.model_config["repo_url"] = repo_url
            self.model_config["rev"] = rev
            self.model_config["model_path_in_repo"] = model_path_in_repo

    def addForces(
        self,
        topology: openmm.app.Topology,
        system: openmm.System,
        atoms: Optional[Iterable[int]],
        force_group: int,
        filename: str = "physicsml_model.pt",
        **args: Any,
    ) -> None:
        """
        This uses the neural network potential to add forces to the OpenMM system.

        Args:
            topology (openmm.app.Topology): Topological information about the system.
            system (openmm.System): _description_
            atoms (Optional[Iterable[int]]): Atoms to include in the simulation.
            force_group (int): The force group that the force belongs to.
            filename (str, optional): Name of the NNP torchscript file used for inference. Defaults to "nnp_model.pt".
        """
        # create the PyTorch model that will be invoked by OpenMM.
        included_atoms = list(topology.atoms())

        if atoms is not None:
            included_atoms = [included_atoms[i] for i in atoms]

        atom_num_list = [atom.element.atomic_number for atom in included_atoms]
        self.model_config["atom_list"] = atom_num_list
        self.model_config["atom_idxs"] = atoms

        is_periodic = (
            topology.getPeriodicBoxVectors() is not None
        ) or system.usesPeriodicBoundaryConditions()

        # load model with system
        to_openmm_torchscript(
            **self.model_config,
            torchscipt_path=filename,
        )

        # add to system using TorchForce.
        force = TorchForce(filename)

        force.setForceGroup(force_group)
        force.setUsesPeriodicBoundaryConditions(is_periodic)
        system.addForce(force)


# register the model with openmm-ml under the name 'physicsml_model'
MLPotential.registerImplFactory("physicsml_model", PhysicsMLPotentialImplFactory())
