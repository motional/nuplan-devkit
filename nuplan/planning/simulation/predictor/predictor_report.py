from dataclasses import dataclass, fields
from typing import Dict, List

import numpy as np


@dataclass(frozen=True)
class PredictorReport:
    """
    Information about predictor runtimes, etc. to store to disk.
    """

    compute_predictions_runtimes: List[float]  # time series of compute_predictions invocation runtimes [s]

    def compute_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics over report fields.
        :return: dictionary containing summary statistics of each field.
        """
        summary = {}
        for field in fields(self):
            attr_value = getattr(self, field.name)
            # Compute summary stats for each field. They are all lists of floats, defined in PredictorReport.
            summary[f"{field.name}_mean"] = np.mean(attr_value)
            summary[f"{field.name}_median"] = np.median(attr_value)
            summary[f"{field.name}_95_percentile"] = np.percentile(attr_value, 95)
            summary[f"{field.name}_std"] = np.std(attr_value)

        return summary


@dataclass(frozen=True)
class MLPredictorReport(PredictorReport):
    """MLPredictor-specific runtime stats."""

    feature_building_runtimes: List[float]  # time series of feature building runtimes [s]
    inference_runtimes: List[float]  # time series of model inference runtimes [s]
