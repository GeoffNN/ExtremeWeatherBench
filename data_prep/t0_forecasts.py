"""Generate gridded t0-beta median hindcasts for ExtremeWeatherBench."""

# Ruff parses jaxtyping shape strings as forward annotations.
# ruff: noqa: F722

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from numpy.typing import NDArray

if TYPE_CHECKING:
    from jaxtyping import Float

MODEL_ID: str = "theforecastingcompany/t0-beta"


def generate_forecasts(
    history: xr.DataArray,
    init_times: Sequence[np.datetime64],
    predict: Callable[
        [Float[NDArray[np.float32], "batch time"], int],
        Float[NDArray[np.float32], "batch horizon"],
    ],
    *,
    context_length: int = 512,
    horizon: int = 40,
    batch_size: int = 32,
) -> xr.Dataset:
    """Forecast each grid cell using only samples through each initialization.

    ``predict`` receives independent univariate rows and returns their medians.
    The input variable must already have its EWB name and units. Only finite
    histories on a regular time axis and a rectilinear grid are supported.
    """
    if min(context_length, horizon, batch_size) < 1:
        raise ValueError("context_length, horizon and batch_size must be positive")
    if history.name is None or set(history.dims) != {"time", "latitude", "longitude"}:
        raise ValueError("Provide a named variable with time, latitude, longitude axes")
    for axis in ("time", "latitude", "longitude"):
        if axis not in history.coords or history[axis].dims != (axis,):
            raise ValueError(f"{axis} must be a one-dimensional coordinate")
        if history.sizes[axis] == 0:
            raise ValueError(f"{axis} must not be empty")
    times: NDArray[np.datetime64] = history.time.values
    if not np.issubdtype(times.dtype, np.datetime64) or np.isnat(times).any():
        raise ValueError("time must contain valid datetime64 values")
    steps: NDArray[np.timedelta64] = np.diff(times)
    if len(steps) == 0 or steps[0] <= np.timedelta64(0, "ns"):
        raise ValueError("At least two strictly increasing timestamps are required")
    if not np.all(steps == steps[0]):
        raise ValueError("time must be regular and strictly increasing")
    initializations: NDArray[np.datetime64] = np.asarray(init_times, dtype=times.dtype)
    if initializations.ndim != 1 or len(initializations) == 0:
        raise ValueError("Provide at least one initialization timestamp")
    if len(np.unique(initializations)) != len(initializations):
        raise ValueError("Initialization timestamps must be unique")
    indices: NDArray[np.int64] = np.searchsorted(times, initializations)
    if np.any(indices >= len(times)) or not np.array_equal(
        times[indices], initializations
    ):
        raise ValueError(
            "Every initialization must occur in the history time coordinate"
        )
    if np.any(indices < context_length - 1):
        raise ValueError("Insufficient history for the requested context_length")
    for axis in ("latitude", "longitude"):
        if not np.isfinite(history[axis].values).all():
            raise ValueError(f"{axis} must contain finite coordinates")
    if np.any(np.abs(history.latitude.values) > 90):
        raise ValueError("latitude must lie between -90 and 90 degrees")
    history = history.assign_coords(longitude=history.longitude % 360).sortby(
        "longitude"
    )
    for axis in ("latitude", "longitude"):
        if len(np.unique(history[axis])) != history.sizes[axis]:
            raise ValueError(f"{axis} must contain unique coordinates")
    history = history.transpose("time", "latitude", "longitude")
    n_lat: int = history.sizes["latitude"]
    n_lon: int = history.sizes["longitude"]
    forecasts: NDArray[np.float32] = np.empty(
        (len(initializations), horizon, n_lat, n_lon), dtype=np.float32
    )
    for init_index, stop in enumerate(indices):
        context: Float[NDArray[np.float32], "grid time"] = (
            np.asarray(
                history.isel(time=slice(stop - context_length + 1, stop + 1)).values,
                dtype=np.float32,
            )
            .reshape(context_length, -1)
            .T
        )
        if not np.isfinite(context).all():
            raise ValueError("History context must contain only finite values")
        output: Float[NDArray[np.float32], "grid horizon"] = np.empty(
            (n_lat * n_lon, horizon), dtype=np.float32
        )
        for start in range(0, len(context), batch_size):
            batch: Float[NDArray[np.float32], "batch time"] = context[
                start : start + batch_size
            ]
            median: Float[NDArray[np.float32], "batch horizon"] = np.asarray(
                predict(batch, horizon), dtype=np.float32
            )
            if median.shape != (len(batch), horizon):
                raise ValueError("Predictor must return shape (batch, horizon)")
            if not np.isfinite(median).all():
                raise ValueError("Predictor returned nonfinite values")
            output[start : start + len(batch)] = median
        forecasts[init_index] = output.T.reshape(horizon, n_lat, n_lon)
    result: xr.DataArray = xr.DataArray(
        forecasts,
        dims=("init_time", "lead_time", "latitude", "longitude"),
        coords={
            "init_time": initializations.astype("datetime64[ns]"),
            "lead_time": np.arange(1, horizon + 1) * steps[0],
            "latitude": history.latitude,
            "longitude": history.longitude,
        },
        name=history.name,
        attrs=history.attrs.copy(),
    )
    return result.to_dataset()


def main() -> None:
    """Load a local history file and write EWB-compatible NetCDF forecasts."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--init-time", required=True, action="append")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--revision", default="main")
    args: argparse.Namespace = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    try:
        from packaging.version import Version
        from t0 import T0Forecaster
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional runtime: uv pip install 'tfc-t0>=0.5.0'"
        ) from exc
    runtime_version: str = version("tfc-t0")
    if Version(runtime_version) < Version("0.5.0"):
        raise RuntimeError("t0-beta requires tfc-t0>=0.5.0 for correct normalization")
    model: T0Forecaster = (
        T0Forecaster.from_pretrained(MODEL_ID, revision=args.revision)
        .to(args.device)
        .eval()
    )

    def predict(
        context: Float[NDArray[np.float32], "batch time"], horizon: int
    ) -> Float[NDArray[np.float32], "batch horizon"]:
        return (
            model.predict(context, horizon=horizon, quantile_levels=[0.5])
            .median.cpu()
            .numpy()
        )

    with xr.open_dataset(args.input) as dataset:
        forecasts: xr.Dataset = generate_forecasts(
            dataset[args.variable],
            [np.datetime64(value) for value in args.init_time],
            predict,
            context_length=args.context_length,
            horizon=args.horizon,
            batch_size=args.batch_size,
        )
    forecasts.attrs.update(
        model=MODEL_ID,
        model_revision=args.revision,
        runtime_version=runtime_version,
        context_length=args.context_length,
        forecast_statistic="median",
        inference="independent univariate grid cells",
    )
    forecasts.to_netcdf(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
