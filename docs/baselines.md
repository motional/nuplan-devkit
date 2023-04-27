# Baselines

We provide several baselines within the devkit. These baselines are standard comparison points in which to compare new 
planners. Moreover, the baselines serve as a starting point for users to prototype their planner or simply tinker with it. 

## SimplePlanner
The SimplePlanner, as the name suggests, has little planning capability. The planner plans a straight line at a constant
speed. The only logic of this planner is to decelerate if the current velocity exceeds the `max_velocity`.

Link to the [code](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/simple_planner.py)

## IDMPlanner
The Intelligent Driver Model Planner (IDMPlanner) consists of two parts:

1. Path planning
2. Longitudinal control (IDM policy)

### Path planning
The path planning is a breadth-first search algorithm. It finds a path towards the mission goal.
The path consists of a serie of lane and lane connectors that leads to the roadblock containing the mission goal.
The baseline is then extracted from the found path and is used as the reference path for the planner.

### IDM Policy
Now that the planner has a reference path, it must then decide how fast to go along this path. For this, it follows
the [IDM policy](https://en.wikipedia.org/wiki/Intelligent_driver_model). The policy describes how fast the planner should
go based on the distance between itself and a given agent. Of course, it is wise to choose the closest agent in the path of the planner.

Hence, the IDMPlanner uses breadth-first search to find the path towards the mission goal, and the IDM policy describes how far along that path the planner should be.

Link to the [code](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/planner/idm_planner.py)

## UrbanDriverOpenLoopModel (MLPlanner)
The UrbanDriverOpenLoopModel functions as an example trained machine learning planner using the `MLPlanner` interface.
The implementation is an open-loop version of L5Kit's [implementation](https://github.com/woven-planet/l5kit/blob/master/l5kit/l5kit/planning/vectorized/open_loop_model.py) 
of ["Urban Driver: Learning to Drive from Real-world Demonstrations Using Policy Gradients"](https://woven-planet.github.io/l5kit/urban_driver.html)
adapted to the nuPlan framework. 

The model processes vectorized agent and map inputs into local feature descriptors that 
are passed to a global attention mechanism for yielding a predicted ego trajectory. The model is trained using imitation 
learning to match expert trajectories available in the nuPlan dataset. Some amount of data augmentation is performed on 
the agents and expert trajectory provided during training to mitigate data distribution drift encountered during 
closed-loop simulation.

Link to the [code](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/training/modeling/models/urban_driver_open_loop_model.py)
