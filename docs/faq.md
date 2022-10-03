# FAQ

## Q: Preprocessing (cache) takes forever
A: **Preprocessing could be bottlenecked by file I/O**. Make sure your dataset is placed at a location with enough I/O bandwith, eg. *local SSDs*. Since our feature builder queries the database files, lacking of I/O bandwith bottlenecks the query speed. Also please check the amount of samples you want to preprocess (details in the next point).

## Q: How many samples should I use? / Dataset size / Subsampling
A: **Preprocessing the full dataset exaustively may not be ideal**. By default we use `NuPlanScenario` so that it generates a sample per timestep in the database, which is `20Hz` -- very dense. By setting `scenario_filter.limit_total_scenarios=0.1` for example, it reduces to `2Hz` so that the number of samples by `1/10`. Note that this rate could be independent to the frequency used in input/target features. You can always have your desired sampling rate of input/target features in your customized `FeatureBuilder` or `TargetBuilder` when extracting features.

## Q: My gradient becomes NaN / Numerical stability / Float precision
A: **FP16 causes numerical instability problems**.
Before deep diving into your own model, make sure your `lightning.trainer.params.precision` is set to `32`.
In the default [configuration](https://github.com/motional/nuplan-devkit/blob/master/nuplan/planning/script/config/training) we set `lightning.trainer.params.precision=16`, meaning that FP16 is used for training. If you are not 100% sure about your model is numerically stable using FP16, please always use FP32.
