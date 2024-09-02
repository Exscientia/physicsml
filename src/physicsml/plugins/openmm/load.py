from typing import Any

import molflux.core as molflux_core

from physicsml.utils import load_from_dvc


def to_openmm_torchscript(
    model_path: str | None = None,
    repo_url: str | None = None,
    rev: str | None = None,
    model_path_in_repo: str | None = None,
    atom_list: list[int] | None = None,
    system_path: str | None = None,
    atom_idxs: list[int] | None = None,
    total_charge: int | None = None,
    y_output: str | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: list[list[float]] | None = None,
    output_scaling: float | None = None,
    position_scaling: float | None = None,
    device: str = "cpu",
    precision: str = "32",
    torchscipt_path: str | None = None,
) -> Any:
    assert (model_path is not None) ^ (
        (repo_url is not None)
        and (rev is not None)
        and (model_path_in_repo is not None)
    ), ValueError(
        "Must specify only one of 'model_path' (a path to a saved physicsml model on disk) or 'repo_url', 'rev', and 'model_path_in_repo'.",
    )

    if model_path is not None:
        model = molflux_core.load_model(model_path)
        featurisation_metadata = molflux_core.load_featurisation_metadata(
            f"{model_path.rstrip('/')}/featurisation_metadata.json",
        )
    elif (
        (repo_url is not None)
        and (rev is not None)
        and (model_path_in_repo is not None)
    ):
        model, featurisation_metadata = load_from_dvc(
            repo_url=repo_url,
            rev=rev,
            model_path_in_repo=model_path_in_repo,
        )
    else:
        raise RuntimeError("Could not load model.")

    openmm_model = model.to_openmm(  # type: ignore
        physicsml_model=model,
        featurisation_metadata=featurisation_metadata,
        atom_list=atom_list,
        system_path=system_path,
        atom_idxs=atom_idxs,
        total_charge=total_charge,
        y_output=y_output,
        pbc=pbc,
        cell=cell,
        output_scaling=output_scaling,
        position_scaling=position_scaling,
        device=device,
        precision=precision,
    )

    return openmm_model.to_torchscript(file_path=torchscipt_path)
