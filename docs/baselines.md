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