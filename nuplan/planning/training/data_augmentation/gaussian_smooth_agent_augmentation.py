from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import gaussian_filter1d

from nuplan.planning.training.data_augmentation.abstract_data_augmentation import AbstractAugmentor
from nuplan.planning.training.data_augmentation.data_augmentation_util import GaussianNoise
from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


class GaussianSmoothAgentAugmentor(AbstractAugmentor):
    """
    Augmentor that takes the perturbed ego current position and generates a smooth trajectory
    over the current and future trajectory.
    """

    def __init__(
        self,
        mean: List[float],
        std: List[float],
        sigma: float,
        augment_prob: float,
    ) -> None:
        """
        Initialize the augmentor class.
        :param mean: parameter to set mean vector of the Gaussian noise on [x, y, yaw].
        :param std: parameter to set standard deviation vector of the Gaussian noise on [x, y, yaw].
        :param sigma: parameter to control the Gaussian smooth level.
        :param augment_prob: probability between 0 and 1 of applying the data augmentation.
        """
        self._sigma = sigma
        self._augment_prob = augment_prob
        self._random_offset_generator = GaussianNoise(mean, std)

    def augment(self, features: FeaturesType, targets: TargetsType) -> Tuple[FeaturesType, TargetsType]:
        """Inherited, see superclass."""
        if np.random.rand() >= self._augment_prob:
            return features, targets

        ego_trajectory: npt.NDArray[np.float32] = np.concatenate(
            [features['agents'].ego[0][-1:, :], targets['trajectory'].data]
        )

        # TODO: Add self.mean and self.std to the noise
        trajectory_length, trajectory_dim = ego_trajectory.shape
        ego_trajectory += np.random.randn(trajectory_length, trajectory_dim) * np.expand_dims(
            np.exp(-np.arange(trajectory_length)), axis=1
        )

        ego_x, ego_y, ego_yaw = ego_trajectory.T
        step_t = np.linspace(0, 1, len(ego_x))
        step_resample_t = np.linspace(0, 1, 100)

        ego_resample_x = np.interp(step_resample_t, step_t, ego_x)
        ego_resample_y = np.interp(step_resample_t, step_t, ego_y)
        ego_resample_yaw = np.interp(step_resample_t, step_t, ego_yaw)
        ego_perturb_x = gaussian_filter1d(ego_resample_x, self._sigma)
        ego_perturb_y = gaussian_filter1d(ego_resample_y, self._sigma)
        ego_perturb_yaw = gaussian_filter1d(ego_resample_yaw, self._sigma)

        ego_perturb_x = np.interp(step_t, step_resample_t, ego_perturb_x)
        ego_perturb_y = np.interp(step_t, step_resample_t, ego_perturb_y)
        ego_perturb_yaw = np.interp(step_t, step_resample_t, ego_perturb_yaw)
        ego_perturb: npt.NDArray[np.float32] = np.vstack((ego_perturb_x, ego_perturb_y, ego_perturb_yaw)).T

        features["agents"].ego[0][-1] = ego_perturb[0]
        targets["trajectory"].data = ego_perturb[1:]

        return features, targets

    @property
    def required_features(self) -> List[str]:
        """Inherited, see superclass."""
        return ['agents']

    @property
    def required_targets(self) -> List[str]:
        """Inherited, see superclass."""
        return ['trajectory']
