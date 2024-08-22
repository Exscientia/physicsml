from typing import Any

import molflux.core as molflux_core

from physicsml.utils import load_from_dvc


def to_ase_calculator(
    model_path: str | None = None,
    repo_url: str | None = None,
    rev: str | None = None,
    model_path_in_repo: str | None = None,
    y_output: str | None = None,
    pbc: tuple[bool, bool, bool] | None = None,
    cell: list[list[float]] | None = None,
    output_scaling: float | None = None,
    position_scaling: float | None = None,
    total_charge: int | None = None,
    device: str = "cpu",
    precision: str = "32",
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

    ase_calculator = model.to_ase(  # type: ignore
        physicsml_model=model,
        featurisation_metadata=featurisation_metadata,
        y_output=y_output,
        pbc=pbc,
        cell=cell,
        output_scaling=output_scaling,
        position_scaling=position_scaling,
        total_charge=total_charge,
        device=device,
        precision=precision,
    )

    return ase_calculator
