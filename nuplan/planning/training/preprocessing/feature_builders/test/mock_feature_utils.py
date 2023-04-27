import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType
from nuplan.planning.training.preprocessing.feature_builders.abstract_feature_builder import AbstractFeatureBuilder
from nuplan.planning.training.preprocessing.features.abstract_model_feature import AbstractModelFeature
from nuplan.planning.training.preprocessing.features.trajectory import Trajectory
from nuplan.planning.training.preprocessing.target_builders.abstract_target_builder import AbstractTargetBuilder


@dataclass
class MockFeature(AbstractModelFeature):
    """
    A simple implementation of the AbstractModelFeature interface to be used with unit tests.
    """

    data: torch.Tensor

    @classmethod
    def deserialize(cls, serialized: Dict[str, Any]) -> AbstractModelFeature:
        """Implemented. See interface."""
        return MockFeature(data=serialized["data"])

    def to_feature_tensor(self) -> AbstractModelFeature:
        """Implemented. See interface."""
        return self

    def to_device(self, device: torch.device) -> AbstractModelFeature:
        """Implemented. See interface."""
        return self

    def unpack(self) -> List[AbstractModelFeature]:
        """Implemented. See interface."""
        raise NotImplementedError


class MockFeatureBuilder(AbstractFeatureBuilder, AbstractTargetBuilder):
    """
    A simple implementation of the AbstractFeatureBuilder and AbstractTargetBuilder interfaces to be used with unit tests.
    """

    def __init__(self, data_tensor: torch.Tensor):
        """
        The init method.
        :param data_tensor: The static data tensor to return from the get_features() methods.
        """
        self.data_tensor = data_tensor

    @classmethod
    def get_feature_type(self) -> Type[AbstractModelFeature]:
        """Implemented. See interface."""
        return MockFeature

    @classmethod
    def get_feature_unique_name(self) -> str:
        """Implemented. See interface."""
        return "MockFeature"

    def get_features_from_scenario(self, scenario: AbstractScenario) -> MockFeature:
        """Implemented. See interface."""
        return MockFeature(data=self.data_tensor)

    def get_features_from_simulation(
        self, current_input: PlannerInput, initialization: PlannerInitialization
    ) -> AbstractModelFeature:
        """Implemented. See interface."""
        raise NotImplementedError

    def get_targets(self, scenario: AbstractScenario) -> MockFeature:
        """Implemented. See interface."""
        return MockFeature(data=self.data_tensor)


class MockTorchModuleWrapperTrajectoryPredictor(TorchModuleWrapper):
    """
    A simple implementation of the TorchModuleWrapper interface for use with unit tests.
    It validates the input tensor, and returns a trajectory object.
    """

    def __init__(
        self,
        future_trajectory_sampling: TrajectorySampling,
        feature_builders: List[AbstractFeatureBuilder],
        target_builders: List[AbstractTargetBuilder],
        raise_on_builder_access: bool = False,
        raise_on_forward: bool = False,
        expected_forward_tensor: Optional[torch.Tensor] = None,
        data_tensor_to_return: Optional[torch.Tensor] = None,
    ) -> None:
        """
        The init method.
        :param future_trajectory_sampling: The TrajectorySampling to use.
        :param feature_builders: The feature builders used by the model.
        :param target_builders: The target builders used by the model.
        :param raise_on_builder_access: If set, an exeption will be raised if the builders are accessed.
        :param raise_on_forward: If set, an exception will be raised if the forward function is called.
        :param expected_forward_tensor: The tensor that is expected to be provided to to the forward function.
        :param data_tensor_to_return: The tensor that expected to be returned from the forward function.
        """
        super().__init__(future_trajectory_sampling, feature_builders, target_builders)

        self.raise_on_builder_access = raise_on_builder_access
        self.raise_on_forward = raise_on_forward
        self.expected_forward_tensor = expected_forward_tensor
        self.data_tensor_to_return = data_tensor_to_return

        # Sanity checks on input.
        if not self.raise_on_builder_access:
            if self.feature_builders is None or len(self.feature_builders) == 0:
                raise ValueError(
                    textwrap.dedent(
                        """
                    raise_on_builder_access set to False with None or 0-length feature builders.
                    This is likely a misconfigured unit test.
                    """
                    )
                )
            if self.target_builders is None or len(self.target_builders) == 0:
                raise ValueError(
                    textwrap.dedent(
                        """
                    raise_on_builder_access set to False with None or 0-length target builders.
                    This is likely a misconfigured unit test.
                    """
                    )
                )

        if not self.raise_on_forward:
            if self.expected_forward_tensor is None:
                raise ValueError(
                    textwrap.dedent(
                        """
                    raise_on_forward set to false with None expected_forward_tensor.
                    This is likely a misconfigured unit test.
                    """
                    )
                )
            if self.data_tensor_to_return is None:
                raise ValueError(
                    textwrap.dedent(
                        """
                    raise_on_forward set to false with None data_tensor_to_return.
                    This is likely a misconfigured unit test.
                    """
                    )
                )

    def get_list_of_required_feature(self) -> List[AbstractFeatureBuilder]:
        """
        Implemented. See interface.
        """
        if self.raise_on_builder_access:
            raise ValueError("get_list_of_required_feature() called when raise_on_builder_access set.")
        result: List[AbstractFeatureBuilder] = TorchModuleWrapper.get_list_of_required_feature(self)
        return result

    def get_list_of_computed_target(self) -> List[AbstractTargetBuilder]:
        """
        Implemented. See interface.
        """
        if self.raise_on_builder_access:
            raise ValueError("get_list_of_computed_target() called when raise_on_builder_access set.")
        result: List[AbstractTargetBuilder] = TorchModuleWrapper.get_list_of_computed_target(self)
        return result

    def forward(self, features: FeaturesType) -> TargetsType:
        """
        Implemented. See interface.
        """
        if self.raise_on_forward:
            raise ValueError("forward() called when raise_on_forward set.")

        self._validate_input_feature(features)

        return {"trajectory": Trajectory(data=self.data_tensor_to_return)}

    def _validate_input_feature(self, features: FeaturesType) -> None:
        """
        Validates that the proper feature is provided.
        Raises an exception if it is not.
        :param features: The feature provided to the model.
        """
        if "MockFeature" not in features:
            raise ValueError(f"MockFeature not in provided features. Available keys: {sorted(list(features.keys()))}")

        if len(features) != 1:
            raise ValueError(f"Expected a single feature. Instead got {len(features)}: {sorted(list(features.keys()))}")

        mock_feature = features["MockFeature"]
        if not isinstance(mock_feature, MockFeature):
            raise ValueError(f"Expected feature of type MockFeature, but got {type(mock_feature)}")

        mock_feature_data = mock_feature.data
        torch.testing.assert_close(mock_feature_data, self.expected_forward_tensor)
