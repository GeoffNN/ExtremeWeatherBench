# Evaluate t0-beta

[t0-beta](https://huggingface.co/theforecastingcompany/t0-beta) is a general
purpose time-series model from The Forecasting Company. The preparation script
`data_prep/t0_forecasts.py` runs it independently at each grid cell and writes
median forecasts that EWB can evaluate through `XarrayForecast`.

This is a univariate baseline: it does not use spatial attention, other weather
variables, or future covariates. It evaluates deterministic medians, not the
model's full predictive distribution. No benchmark scores are supplied here.

## Generate forecasts

Run from a checkout of this repository, with the optional runtime installed:

```shell
uv sync --group dev --all-extras
uv pip install 'tfc-t0>=0.5.0'
uv run --no-sync python data_prep/t0_forecasts.py \
    --input temperature_history.nc \
    --variable surface_air_temperature \
    --init-time 2021-06-25T00:00:00 \
    --context-length 512 --horizon 40 --batch-size 32 \
    --output t0_beta_forecasts.nc
```

The input is a local NetCDF file containing the selected variable with dimensions
`time`, `latitude`, and `longitude`. Use a regular, strictly increasing time axis
and a rectilinear grid. Rename input variables and coordinates to EWB conventions
before running; for example, ERA5 `t2m` becomes `surface_air_temperature`, in kelvin.
The script preserves units and does not convert them. Subset the grid to cover the
event region before generation; output is assembled in memory.

Each `--init-time` (repeat the option for multiple initializations) must occur in
the input. The context includes that timestamp and the preceding
`context_length - 1` samples. Later observations are excluded. Lead times start
one input sampling interval after initialization; `horizon` counts samples,
not hours. With six-hourly input, `--horizon 40` produces ten days of forecasts.
Missing or infinite history values are rejected. Negative longitudes are
converted to the 0–360 convention and sorted with their data.

The runtime must be at least 0.5.0 to read t0-beta's normalization configuration.
Use `--device cuda` for GPU inference (CPU is the default). For reproducibility,
pass a Hugging Face commit hash via `--revision` and retain the runtime version,
input provenance and command alongside your results.

## Evaluate the generated dataset

Choose cases whose entire spatial region and evaluation period are covered by
your forecasts. For example, use the Pacific Northwest heat dome case defined
in the [single-case recipe](single_case.md):

```python
import xarray as xr
import extremeweatherbench as ewb

# pnw_heat_dome is the IndividualCase from the single-case recipe.
with xr.open_dataset("t0_beta_forecasts.nc") as dataset:
    forecast = ewb.XarrayForecast(
        ds=dataset.load(),
        name="t0-beta",
        variables=["surface_air_temperature"],
    )

evaluation = ewb.EvaluationObject(
    event_type="heat_wave",
    forecast=forecast,
    target=ewb.ERA5(variables=["surface_air_temperature"]),
    metric_list=[
        ewb.metrics.RootMeanSquaredError(),
        ewb.metrics.MaximumMeanAbsoluteError(),
    ],
)
runner = ewb.evaluation(
    case_metadata=[pnw_heat_dome],
    evaluation_objects=[evaluation],
)
results = runner.run_evaluation()
results.to_csv("t0_beta_heatwave_results.csv", index=False)
```

Historical reanalysis used as context makes this a reanalysis-initialized
hindcast experiment. It does not establish real-time forecast skill: account
for source availability delays in any operational comparison. Check the model's
training data coverage before claiming out-of-sample results on historical EWB
cases; the public model card does not establish a weather-data training cutoff.
