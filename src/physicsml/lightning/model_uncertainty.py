from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generic,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import scipy.stats as st
import torch
from datasets import Dataset
from molflux.modelzoo.model import (
    PredictionIntervalMixin,
    StandardDeviationMixin,
)
from molflux.modelzoo.models.lightning.config import DataModuleConfig, TrainerConfig
from molflux.modelzoo.typing import PredictionResult

from physicsml.lightning.model import PhysicsMLModelBase

if TYPE_CHECKING:
    from physicsml.lightning.config import PhysicsMLModelConfig

_PhysicsMLModelConfigT = TypeVar(
    "_PhysicsMLModelConfigT",
    bound="PhysicsMLModelConfig",
)


class PhysicsMLUncertaintyModelBase(
    PredictionIntervalMixin,
    StandardDeviationMixin,
    PhysicsMLModelBase[_PhysicsMLModelConfigT],
    Generic[_PhysicsMLModelConfigT],
):
    """ABC for all Huggingface PhysicsML models with uncertainty"""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def _predict_with_std(
        self,
        data: Dataset,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        """method for predicting"""
        del kwargs
        display_names = self._predict_display_names
        # if data is empty
        if not len(data):
            empty_out: PredictionResult = {
                display_name: [] for display_name in display_names
            }
            empty_out_std: PredictionResult = {
                f"{display_name}::std": [] for display_name in display_names
            }
            return empty_out, empty_out_std

        batch_preds = self._predict_batched(
            data=data,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
        )

        batched_preds_dict = {
            k: [dic[k] for dic in batch_preds] for k in batch_preds[0]
        }

        output = {}
        for key, cols in [
            ("y_node_scalars", self.model_config.datamodule.y_node_scalars),
            ("y_edge_scalars", self.model_config.datamodule.y_edge_scalars),
            ("y_graph_scalars", self.model_config.datamodule.y_graph_scalars),
        ]:
            if cols is not None:
                catted_value = torch.cat(batched_preds_dict[key], dim=0)
                for idx, col in enumerate(cols):
                    output[col] = catted_value[:, idx].tolist()

        for key, vec_col in [
            ("y_node_vector", self.model_config.datamodule.y_node_vector),
            ("y_edge_vector", self.model_config.datamodule.y_edge_vector),
            ("y_graph_vector", self.model_config.datamodule.y_graph_vector),
        ]:
            if vec_col is not None:
                output[vec_col] = torch.cat(batched_preds_dict[key], dim=0).tolist()

        # Only support uncertainty for graph scalars
        output_std = {}
        for key, cols in [
            ("y_graph_scalars::std", self.model_config.datamodule.y_graph_scalars),
        ]:
            if cols is not None:
                catted_value = torch.cat(batched_preds_dict[key], dim=0)
                for idx, col in enumerate(cols):
                    output_std[f"{col}::std"] = catted_value[:, idx].tolist()

        # fill in none lists for outputs with no std
        for key, value in output.items():
            if f"{key}::std" not in output_std:
                output_std[f"{key}::std"] = [None] * len(value)

        ordered_output = {
            display_name: output[y_feature]
            for display_name, y_feature in zip(display_names, self.y_features)
        }

        ordered_output_std = {
            f"{display_name}::std": output_std[f"{y_feature}::std"]
            for display_name, y_feature in zip(display_names, self.y_features)
        }

        return ordered_output, ordered_output_std

    def _predict_with_prediction_interval(
        self,
        data: Dataset,
        confidence: float,
        datamodule_config: Union[DataModuleConfig, Dict[str, Any], None] = None,
        trainer_config: Union[TrainerConfig, Dict[str, Any], None] = None,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        # get the result dictionary of means and standard deviations
        prediction_mean_results, prediction_std_results = self._predict_with_std(
            data,
            datamodule_config=datamodule_config,
            trainer_config=trainer_config,
        )

        # for each column, map the mean and standard deviation to a prediction interval
        prediction_results: PredictionResult = {}
        prediction_prediction_interval_results: PredictionResult = {}
        for display_name, mean, std in zip(
            self._predict_display_names,
            prediction_mean_results.values(),
            prediction_std_results.values(),
        ):
            if all(s is None for s in std):
                prediction_results[display_name] = mean
                prediction_prediction_interval_results[
                    f"{display_name}::prediction_interval"
                ] = [[None, None] for _ in std]
            else:
                # compute the prediction interval
                lower_bound, upper_bound = st.norm.interval(
                    confidence,
                    loc=mean,
                    scale=std,
                )

                # where nans occur (ex. from a 0 standard deviation), use the mean instead
                lower_bound = np.where(~np.isnan(lower_bound), lower_bound, mean)
                upper_bound = np.where(~np.isnan(upper_bound), upper_bound, mean)

                prediction_results[display_name] = mean
                prediction_prediction_interval_results[
                    f"{display_name}::prediction_interval"
                ] = list(zip(lower_bound, upper_bound))

        return prediction_results, prediction_prediction_interval_results

    def _calibrate_uncertainty(self, data: Any, **kwargs: Any) -> Any:
        pass

    def _predict_with_uncertainty(
        self,
        data: Any,
        confidence: float,
        **kwargs: Any,
    ) -> Tuple[PredictionResult, PredictionResult]:
        return self._predict_with_prediction_interval(
            data,
            confidence=confidence,
            **kwargs,
        )
