# type: ignore
from typing import Dict

import pytest
import torch

try:
    from openmmml.mlpotential import MLPotential

    import openmm as mm
    import openmm.app as app
    from physicsml.plugins.openmm.physicsml_potential import (
        PhysicsMLPotentialImplFactory,  # noqa F401
    )

    OPENMM_ = True
except:  # noqa
    OPENMM_ = False

NNP_MODELS = [
    {
        "model": "ANI",
        "model_path": "tests/data/ani_model",
        "total_value": -5265.576795347246,
        "mm_value": -159.59874317381482,
        "mixed_value": -2177.4458493497445,
    },
    {
        "model": "EGNN",
        "model_path": "tests/data/egnn_model",
        "total_value": -5270.628634669147,
        "mm_value": -159.59874317381482,
        "mixed_value": -2176.95439365827,
    },
    {
        "model": "MACE",
        "model_path": "tests/data/mace_model",
        "total_value": -5269.477412745436,
        "mm_value": -159.59874317381482,
        "mixed_value": -2177.3664037404014,
    },
    {
        "model": "NEQUIP",
        "model_path": "tests/data/nequip_model",
        "total_value": -5266.504151955372,
        "mm_value": -159.59874317381482,
        "mixed_value": -2176.9882577057383,
    },
]
ALANINE_DIPEPTIDE_PATH = "tests/data/alanine-dipeptide-truncated.pdb"


@pytest.mark.skipif(not OPENMM_, reason="no openmm")
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_nnp_potential_system_cpu(
    model_properties: Dict,
    platform: str = "cpu",
) -> None:
    # fetch the .pdb file
    pdb = app.PDBFile(ALANINE_DIPEPTIDE_PATH)

    # specify the Mace potential
    potential = MLPotential(
        "physicsml_model",
        model_path=model_properties["model_path"],
        precision="64",
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform,
    )

    # set the platform to run on
    platform_openmm = mm.Platform.getPlatformByName(f"{platform}".upper())

    mm_system = potential.createSystem(
        pdb.topology,
    )

    # create the context for mm system
    mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform_openmm)

    # set the positions
    mm_context.setPositions(pdb.positions)

    # get energy
    energy = mm_context.getState(getEnergy=True).getPotentialEnergy()

    assert energy._value == pytest.approx(model_properties["total_value"], 1e-4)


@pytest.mark.skipif(not OPENMM_, reason="no openmm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpus")
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_nnp_potential_system_gpu(
    model_properties: Dict,
    platform: str = "cuda",
) -> None:
    # fetch the .pdb file
    pdb = app.PDBFile(ALANINE_DIPEPTIDE_PATH)

    # specify the Mace potential
    potential = MLPotential(
        "physicsml_model",
        model_path=model_properties["model_path"],
        precision="32",
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform,
    )

    # set the platform to run on
    platform_openmm = mm.Platform.getPlatformByName(f"{platform}".upper())

    mm_system = potential.createSystem(
        pdb.topology,
    )

    # create the context for mm system
    mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform_openmm)

    # set the positions
    mm_context.setPositions(pdb.positions)

    # get energy
    energy = mm_context.getState(getEnergy=True).getPotentialEnergy()

    assert energy._value == pytest.approx(model_properties["total_value"], 1e-4)


@pytest.mark.skipif(not OPENMM_, reason="no openmm")
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_nnp_mixed_potential_system_cpu(
    model_properties: str,
    platform: str = "cpu",
) -> None:
    pdb = app.PDBFile(ALANINE_DIPEPTIDE_PATH)
    # specify the amber force fields
    force_field = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # create the mm system
    mm_system = force_field.createSystem(pdb.topology, nonbondedMethod=app.PME)

    # create list of atoms to be modelled using NNP
    ml_atoms = [a.index for a in next(pdb.topology.chains()).atoms()]

    # specify the Mace potential
    potential = MLPotential(
        "physicsml_model",
        model_path=model_properties["model_path"],
        precision="64",
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform,
    )
    # create mixed system
    mixed_system = potential.createMixedSystem(
        topology=pdb.topology,
        system=mm_system,
        atoms=ml_atoms,
        interpolate=False,
    )
    # set the platform to run on
    platform_openmm = mm.Platform.getPlatformByName(f"{platform}".upper())

    # create the context for mm system
    mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform_openmm)

    # create the context for mixed system
    mixed_context = mm.Context(
        mixed_system,
        mm.VerletIntegrator(0.001),
        platform_openmm,
    )

    # set the positions
    mm_context.setPositions(pdb.positions)
    mixed_context.setPositions(pdb.positions)

    # get energy
    mm_energy = mm_context.getState(getEnergy=True).getPotentialEnergy()
    mixed_energy = mixed_context.getState(getEnergy=True).getPotentialEnergy()

    assert mm_energy._value == pytest.approx(model_properties["mm_value"], 1e-4)
    assert mixed_energy._value == pytest.approx(model_properties["mixed_value"], 1e-4)


@pytest.mark.skipif(not OPENMM_, reason="no openmm")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no gpus")
@pytest.mark.parametrize("model_properties", NNP_MODELS)
def test_nnp_mixed_potential_system_gpu(
    model_properties: str,
    platform: str = "cuda",
) -> None:
    pdb = app.PDBFile(ALANINE_DIPEPTIDE_PATH)
    # specify the amber force fields
    force_field = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")

    # create the mm system
    mm_system = force_field.createSystem(pdb.topology, nonbondedMethod=app.PME)

    # create list of atoms to be modelled using NNP
    ml_atoms = [a.index for a in next(pdb.topology.chains()).atoms()]

    # specify the Mace potential
    potential = MLPotential(
        "physicsml_model",
        model_path=model_properties["model_path"],
        precision="32",
        position_scaling=10.0,
        output_scaling=4.184,
        device=platform,
    )
    # create mixed system
    mixed_system = potential.createMixedSystem(
        topology=pdb.topology,
        system=mm_system,
        atoms=ml_atoms,
        interpolate=False,
    )
    # set the platform to run on
    platform_openmm = mm.Platform.getPlatformByName(f"{platform}".upper())

    # create the context for mm system
    mm_context = mm.Context(mm_system, mm.VerletIntegrator(0.001), platform_openmm)

    # create the context for mixed system
    mixed_context = mm.Context(
        mixed_system,
        mm.VerletIntegrator(0.001),
        platform_openmm,
    )

    # set the positions
    mm_context.setPositions(pdb.positions)
    mixed_context.setPositions(pdb.positions)

    # get energy
    mm_energy = mm_context.getState(getEnergy=True).getPotentialEnergy()
    mixed_energy = mixed_context.getState(getEnergy=True).getPotentialEnergy()

    assert mm_energy._value == pytest.approx(model_properties["mm_value"], 1e-4)
    assert mixed_energy._value == pytest.approx(model_properties["mixed_value"], 1e-4)
