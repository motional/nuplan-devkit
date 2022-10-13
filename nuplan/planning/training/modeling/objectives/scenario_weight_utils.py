from typing import Dict, List

import torch

from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario


def extract_scenario_type_weight(
    scenarios: List[AbstractScenario], scenario_type_loss_weights: Dict[str, float], device: torch.device
) -> torch.Tensor:
    """
    Gets the scenario loss weights.
    :param scenarios: List of scenario objects
    :return: Tensor with scenario_weights
    """
    default_scenario_weight = 1.0
    scenario_weights = [scenario_type_loss_weights.get(s.scenario_type, default_scenario_weight) for s in scenarios]
    return torch.FloatTensor(scenario_weights).to(device)
