nuPlan Planning Challenge
=========================

### Overview

The main focus of the nuPlan planning challenge is to evaluate a motion planning system in realistic driving scenarios, using multiple performance metrics. In this challenge, it is assumed that a planner will consume a top-down semantic representation of detected traffic participants (vehicles, bicycles, etc.) and static obstacles and plan the vehicle’s future trajectory for a specific time horizon. The challenge is organized into three increasingly complex modes:

1.  Open-loop
2.  Closed-loop non-reactive agents
3.  Closed-loop reactive agents

### Competition Timeline

*   January 30th, 2023 - Warm-up phase and Test phase opens for submission
*   May 26th, 2023 - Competition closes - Finalists are announced and invited to share their code with us for verification
*   June 2th, 2023 - Winners are announced
*   June 18th, 2023 - The competition results are presented at the [End-to-End Autonomous Driving](https://opendrivelab.com/event/cvpr23_ADworkshop) workshop at CVPR 2023


### How to enter

#### Getting Started

1.  Sign up and download the dataset from [https://www.nuscenes.org/nuplan](https://www.nuscenes.org/nuplan).
2.  Clone the devkit. Feel free to ask any questions by raising an issue on our GitHub: [https://github.com/motional/nuplan-devkit](https://github.com/motional/nuplan-devkit)
3.  Create a team on [EvalAI](https://eval.ai/web/challenges/challenge-page/1856/overview).

#### Making a submission

1. Go through the [nuPlan submission tutorial](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_submission_tutorial.html). The tutorial will show how to create a valid Docker image.
2. Submit your Docker image through [EvalAI](https://eval.ai/web/challenges/challenge-page/1856/overview).
3. Send the following information to nuScenes@motional.com after submitting on EvalAI

```
- Team name
- Method name
- Authors
- Affiliations
- Country (Of the individual or team's institution)
- Method description (5+ sentences)
- Project URL (If applicable)
- Paper URL (If applicable)
- Submission ID
```

#### Evaluation protocol and rules
Participants of the nuPlan planning challenge will submit their code to the challenge’s cloud-based evaluation server through EvalAI - an open-source evaluation platform. The code will be submitted in the form of a Docker image and will be executed for each of the challenge modes
(open-loop, non-reactive closed-loop and reactive closed-loop simulation) against the hidden test set of the challenge.

After submission, metrics that were generated for each simulation will be aggregated and sent to the public leaderboard. A final scoring function will be used to rank all submissions in each challenge. In addition, granular metrics and error analysis plots will be available for each submission on the challenge’s page of the nuPlan official website (coming soon).
For each of the three planning challenges, the following set of rules applies:
- A fixed set of scenario simulations for each scenario type (e.g. lane change) will be run
to evaluate a submission.
- Each scenario simulation will start from a fixed point in time that represents the beginning
of the scenario (e.g. the start of an unprotected turn).
- The current and past (up to 2s) scene information for each simulation iteration will be
passed to the planner - that includes ego pose, ego route, agents, and static/dynamic map
information.
- Each planner will have a fixed time budget of 1s for each simulation iteration, after
which the simulation will time out.
- The simulation horizon will be up to 15s for each scenario and the simulation frequency
will be 10Hz.
- There will be a strict limit of three submissions for each participant for this competition.
- Every submission should provide a short description of the method used.
- Submissions that obfuscate code in the submitted Docker image will not be considered for the competition and therefore winning prizes. However, participants are still welcome to submit their "top secret" planner
- The top performing submissions will be inspected for compliance with the challenge’s
rules - any attempt to circumvent these rules will result in a permanent ban of the team
or company from all future nuPlan challenges.

### Phases
#### Warm-up Phase
The warm-up phase allows participants to test the submission pipeline. During this phase, participants may submit up 
to 10 submissions. The submissions will be run on all three challenges. The scenario types used for the warm-up phase 
are the same as the ones that will be used for the test phase. However, the number of scenarios run will be smaller,
and the data is not taken from the test split. The metrics and the overall scores will be displayed on the leaderboard.

Start: January 30th, 2023

End: May 26th, 2023

#### Test Phase
The results from the test phase will be used to determine the winners of each challenge. All submissions will be evaluated on a 
larger set of scenarios from the test split. The scores from the test phases will be used to shortlist finalists. Finalists will then 
be invited to share their code with the organizers. Once the submissions from the finalists are verified by the 
organizers, the same scores are used to determine the winners of each challenge. Participants will be allowed to 
submit a maximum of three submissions.

Start: January 30th, 2023

End: May 26th, 2023

### Challenges

### Planner Output Requirements

**Warning**: The following requirements must be met for a submission to be considered valid. Failing to meet these requirements will result in your submission failing.

| Planner Trajectory Requirements | Value |
| ------------------------------- | ----- |
| Expected minimum horizon length | 8s    |
| Expected minimum horizon steps  | 2     |
| Expected signals                | <ul><li> x position in global frame of Ego's rear axles center<li> y position in global frame of Ego's rear axles center <li> heading in global frame <li> time stamp </li></ul>|

### Evaluation server


| Configuration                      | Value                                                                                                                                                                       |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Simulation iteration time budget   | 1s                                                                                                                                                                     |
| Max submissions per each user/team | 3                                                                                                                                                                           |
| Instance type                      | <ul><li>16 CPU, 64 RAM<li>Resource partitioning: <ul><li>SimulationContainer: 0GPU, 2vCPU, 28GB RAM<li>SubmissionContainer: 1GPU, 8vCPU, 32GB RAM</li></ul></li></ul>|

All challenges will be run with the following common configurations

| Common Challenge Configuration | Value |
| ------------------------------ | ----- |
| Frequency                      | 10Hz  |
| Rollout horizon                | 15s   |
| Past observation horizon       | 2s    |


#### Challenge 1: Open-loop

In this challenge, the planning system is tasked with imitating the expert driver in open-loop. For every time step, the planner’s predicted trajectory is scored based on a set of predefined open-loop metrics. Namely, it is compared directly against the driven ground-truth trajectory. As the name suggests, the planned trajectory will not be used to alter the state of the simulation.

| Configuration              | Value                        |
| -------------------------- |----------------------------- |
| Metric computation horizon | 3s, 5s, 8s @ 1Hz sampling    |
| Controller                 | N/A                          |
| Observations               |  <ul><li>Vehicles<li>Pedestrians<li>Cyclists<li>Generic Objects<li>Traffic cones<li>Barriers<li>Construction Zone Signs</li></ul>                          |


##### Scoring

The overall score of challenge 1 considers the following factors:

*   Average Displacement Error (ADE)
*   Final Displacement Error (FDE)
*   Average Heading Error (AHE)
*   Final Heading Error (FHE)
*   Miss Rate

The details of each metric can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html#).
The details of the scoring hierarchy can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_metrics_aggeregation.html)

#### Challenge 2: Closed-loop non-reactive agents

In this challenge, the planner outputs a planned trajectory using the information available at each time step, similar to the previous case. The predicted trajectory is used as a reference for a tracking controller to simulate the vehicle’s state at the next time step. Nonetheless, the other agents are replayed from a log according to their observed states in open-loop - hence, the name non-reactive. Most importantly, the ground-truth is no longer applicable to the evaluation of the planner. Instead, a suite of closed-loop metrics will be used to judge the performance of a planner

| Configuration | Value |
| ------------- | ----- |
| Controller    | [LQR](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py)  |
| Observations  | <ul><li>Vehicles<li>Pedestrians<li>Cyclists<li>Generic Objects<li>Traffic cone<li>Barriers<li>Construction Zones Signs</li></ul>                          |


##### Scoring

The overall score of challenge 2 considers the following factors:

*   At-fault collision
*   Drivable area compliance
*   Driving direction compliance
*   Making progress
*   Time to collision (TTC)
*   Speed limit compliance
*   Ego progress along the expert's route ratio
*   Comfort

The details of each metric can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html).
The details of the scoring hierarchy can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_metrics_aggeregation.html)

#### Challenge 3: Closed-loop reactive agents

In the final and most complex challenge, the vehicles in the scene become reactive. In other words, the vehicles will react to the actions of the planner-controlled vehicle along with the behavior of other agents, the policy determining the behavior of agents is an Intelligent Driver Model ([IDM](https://en.wikipedia.org/wiki/Intelligent_driver_model)) policy. The rest of the observations are still replayed in open-loop. Similar to the non-reactive closed-loop challenge, a suite of metrics will be used to judge the performance of a planner in closed-loop.

| Configuration | Value |
| ------------- | ----- |
| Controller    | [LQR](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/simulation/controller/tracker/lqr.py) |
| Observations  | <ul><li>Reactive:<ul><li>Vehicles</li></ul><li>Open-loop<ul><li>Pedestrians<li>Generic Objects<li>Traffic cone<li>Barriers<li>Construction Zones Signs</li></ul> </li></ul>|

##### Scoring

The overall score of challenge 3 considers the following factors:

*   At-fault collision
*   Drivable area compliance
*   Driving direction compliance
*   Making progress
*   Time to collision (TTC)
*   Speed limit compliance
*   Ego progress along the expert's route ratio
*   Comfort

The details of each metric can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/metrics_description.html#).
The details of the scoring hierarchy can be found [here](https://nuplan-devkit.readthedocs.io/en/latest/nuplan_metrics_aggeregation.html)

### Scenario Selection

All three challenges will be run on the following scenario types.

| Scenario Type                                               | Description                                                                                                                                                                         |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| starting\_straight\_traffic\_light\_intersection\_traversal | Ego at the start of a traversal going straight across an intersection area controlled by traffic lights while not stopped                                                           |
| high\_lateral\_acceleration                                 | Ego high ego acceleration (1.5 < acceleration < 3 m/s^2) across the lateral axis with high yaw rate while not turning                                                               |
| changing\_lane                                              | Ego at the start of a lane change towards an adjacent lane                                                                                                                          |
| high\_magnitude\_speed                                      | Ego high velocity magnitude with low acceleration (velocity > 9 m/s)                                                                                                                |
| low\_magnitude\_speed                                       | Ego low ego velocity magnitude (0.3 < velocity < 1.2 m/s) with low acceleration while not stopped                                                                                   |
| starting\_left\_turn                                        | Ego at the start of a traversal turning left across an intersection area while not stopped                                                                                          |
| starting\_right\_turn                                       | Ego at the start of a traversal turning right across an intersection area while not stopped                                                                                         |
| stopping\_with\_lead                                        | Ego starting to decelerate (acceleration magnitude < -0.6 m/s^2, velocity magnitude < 0.3 m/s) with a leading vehicle ahead (distance < 6 m) at any area                            |
| following\_lane\_with\_lead                                 | Ego following (velocity > 3.5 m/s) its current lane with a moving leading vehicle ahead on the same lane (velocity > 3.5 m/s, longitudinal distance < 7.5 m)                        |
| near\_multiple\_vehicles                                    | Ego nearby (distance < 8 m) of multiple (>6) moving vehicles while ego is moving (velocity > 6 m/s)                                                                                 |
| traversing\_pickup\_dropoff                                 | Ego during the traversal of a pickup/drop-off area while not stopped                                                                                                                |
| behind\_long\_vehicle                                       | Ego behind (3 m < longitudinal distance < 10 m) a long (length > 8 m) vehicle in the same lane as ego (lateral distance < 0.5 m)                                                    |
| waiting\_for\_pedestrian\_to\_cross                         | Ego waiting for a nearby (distance < 8 m, time to intersection < 1.5 m) pedestrian to cross a crosswalk area while ego is not stopped and the pedestrian is not at a pickup/drop-off area |
| stationary\_in\_traffic                                     | Ego is stationary with multiple (>6) vehicles nearby (distance < 8 m)                                                                                                               |

### How to win

The submissions are ranked based on the mean overall score across all three challenges.

#### Prizes

- 1st place: USD $10,000
- 2nd place: USD $8,000
- 3rd place: USD $5,000
- Innovation Prize: USD $5,000
  - The Innovation Prize will be awarded to the most innovative submission judged by a panel of Motional experts.
- Internship opportunity
  - Eligible individual participants may be considered for internships at Motional based on their performance on the challenges. Any offers for internships are subject to the Competition Terms, the participant successfully completing the Motional application process, and any other terms and conditions solely determined by Motional.

[Terms and conditions](https://eval.ai/web/challenges/challenge-page/1856/evaluation) apply.

### Contact

To contact the organizers, please make an issue on our [GitHub page](https://github.com/motional/nuplan-devkit/issues) or email us at nuScenes@motional.com
