from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Type

import numpy as np
import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature, FeatureDataType


@dataclass
class DummyVectorMapFeature(AbstractModelFeature):
    """Dummy vector map feature used in testing."""

    data1: List[FeatureDataType]
    data2: List[FeatureDataType]
    data3: List[Dict[str, FeatureDataType]]

    def __post_init__(self) -> None:
        """Sanitize attributes of dataclass."""
        if len(self.data1) != len(self.data2) != len(self.data3):
            raise RuntimeError(
                f"Not consistent length of batches! {len(self.data1)}!= {len(self.data2)} != {len(self.data3)}"
            )

        if self.num_of_batches == 0:
            raise ValueError("Batch size has to be larger than 0!")

        assert "test" in self.data3[0].keys(), f"Test is not present in data: {self.data3[0].keys()}!"

    @classmethod
    def collate(cls, batch: List[DummyVectorMapFeature]) -> DummyVectorMapFeature:
        """Inherited, see superclass."""
        return DummyVectorMapFeature(
            data1=[data for b in batch for data in b.data1],
            data2=[data for b in batch for data in b.data2],
            data3=[data for b in batch for data in b.data3],
        )

    @property
    def num_of_batches(self) -> int:
        """Number of batches in the feature."""
        return len(self.data1)

    def to_feature_tensor(self) -> DummyVectorMapFeature:
        """Inherited, see superclass."""
        return DummyVectorMapFeature(self.data1, self.data2, self.data3)

    def to_device(self, device: torch.device) -> DummyVectorMapFeature:
        """Implemented. See interface."""
        return DummyVectorMapFeature(
            data1=[data.to(device) for data in self.data1],
            data2=[data.to(device) for data in self.data2],
            data3=[{"test": data["test"].to(device) for data in self.data3}],
        )

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> DummyVectorMapFeature:
        """Implemented. See interface."""
        return DummyVectorMapFeature(data1=data["data1"], data2=data["data2"], data3=data["data3"])

    def unpack(self) -> List[DummyVectorMapFeature]:
        """Implemented. See interface."""
        return [
            DummyVectorMapFeature([data1], [data2], [data3])
            for data1, data2, data3 in zip(self.data1, self.data2, self.data3)
        ]


class DummyVectorMapBuilder(AbstractFeatureBuilder):
    """Dummy vector map feature builder used in testing."""

    @classmethod
    def get_feature_unique_name(cls) -> str:
        """Inherited, see superclass."""
        return "vector_map"

    @classmethod
    def get_feature_type(cls) -> Type[AbstractModelFeature]:
        """Inherited, see superclass."""
        return DummyVectorMapFeature

    def get_features_from_simulation(
        self, history: SimulationHistoryBuffer, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        """Inherited, see superclass."""
        return NotImplemented

    def get_features_from_scenario(self, scenario: AbstractScenario) -> AbstractModelFeature:
        """Inherited, see superclass."""
        return DummyVectorMapFeature(
            data1=[np.zeros((10, 10, 10))],
            data2=[np.zeros((10, 10, 10))],
            data3=[{"test": np.zeros((10, 10, 10))}],
        )
