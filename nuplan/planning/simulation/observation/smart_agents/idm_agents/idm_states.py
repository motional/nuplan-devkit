from dataclasses import dataclass


@dataclass
class IDMAgentState:
    progress: float  # [m] distane a long a path
    velocity: float  # [m/s] velocity along the oath

    def to_array(self):
        return [self.progress, self.velocity]


@dataclass
class IDMLeadAgentState(IDMAgentState):
    length_rear: float  # [m] length from vehicle CoG to the rear bumper

    def to_array(self):
        return [self.progress, self.velocity, self.length_rear]
