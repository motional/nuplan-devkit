from typing import List, Tuple

from nuplan.planning.training.modeling.types import FeaturesType, TargetsType


def _batch_abstract_features(initial_not_batched_features: FeaturesType,
                             to_be_batched_features: List[FeaturesType]) -> FeaturesType:
    """
    Batch abstract feature with custom collate function
    :param initial_not_batched_features: features from initial batch which are used only for keys
    :param to_be_batched_features: list of features which should be batched
    :return: batched features
    """
    output_features = {}
    for key in initial_not_batched_features.keys():
        list_features = [feature_single[key] for feature_single in to_be_batched_features]
        output_features[key] = initial_not_batched_features[key].collate(list_features)

    return output_features


class FeatureCollate:

    def __call__(self, batch: List[Tuple[FeaturesType, TargetsType]]) -> Tuple[FeaturesType, TargetsType]:
        """
        Collate list of [Features,Targets] into batch
        :param batch: list of tuples to be batched
        :return (features, targets) already batched
        """
        assert len(batch) > 0, "Batch size has to be greater than 0!"

        to_be_batched_features = [batch_i[0] for batch_i in batch]
        to_be_batched_targets = [batch_i[1] for batch_i in batch]

        initial_features, initial_targets = batch[0]

        out_features = _batch_abstract_features(initial_features, to_be_batched_features)
        out_targets = _batch_abstract_features(initial_targets, to_be_batched_targets)

        return out_features, out_targets
