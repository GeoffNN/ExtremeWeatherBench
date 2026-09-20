"""Tests for preparing t0-beta forecasts without loading model weights."""

from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.typing import NDArray

from data_prep.t0_forecasts import generate_forecasts


def make_history() -> xr.DataArray:
    times: NDArray[np.datetime64] = np.arange(
        np.datetime64("2024-06-01T00", "h"),
        np.datetime64("2024-06-03T00", "h"),
        np.timedelta64(6, "h"),
    )
    values: NDArray[np.float64] = np.arange(48, dtype=float).reshape(8, 2, 3)
    return xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [50.0, 51.0], "longitude": [0, 90, 180]},
        name="surface_air_temperature",
        attrs={"units": "K"},
    )


def persistence(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
    return np.repeat(context[:, -1:], horizon, axis=1)


def test_batches_preserve_gridpoints_and_use_only_available_history() -> None:
    history: xr.DataArray = make_history()
    batches: list[NDArray[np.float32]] = []

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        batches.append(context.copy())
        return persistence(context, horizon)

    init_times: list[np.datetime64] = [history.time.values[2], history.time.values[4]]
    result: xr.Dataset = generate_forecasts(
        history, init_times, predict, context_length=3, horizon=2, batch_size=4
    )

    assert result.sizes == {
        "init_time": 2,
        "lead_time": 2,
        "latitude": 2,
        "longitude": 3,
    }
    np.testing.assert_array_equal(result.init_time.values, init_times)
    np.testing.assert_array_equal(
        result.lead_time.values, np.array([6, 12], dtype="timedelta64[h]")
    )
    np.testing.assert_array_equal(result.latitude.values, history.latitude.values)
    np.testing.assert_array_equal(result.longitude.values, history.longitude.values)
    assert len(batches) == 4
    assert [batch.shape for batch in batches] == [(4, 3), (2, 3), (4, 3), (2, 3)]
    for index, time_index in enumerate([2, 4]):
        contexts: NDArray[np.float32] = np.concatenate(
            batches[2 * index : 2 * index + 2]
        )
        expected: NDArray[np.float64] = (
            history.values[time_index - 2 : time_index + 1].reshape(3, 6).T
        )
        np.testing.assert_array_equal(contexts, expected)
        actual: NDArray[np.float32] = (
            result.surface_air_temperature.isel(init_time=index)
            .transpose("lead_time", "latitude", "longitude")
            .values
        )
        np.testing.assert_array_equal(
            actual, np.repeat(history.values[time_index][None], 2, axis=0)
        )


def test_future_observations_do_not_change_forecast() -> None:
    history: xr.DataArray = make_history()
    modified: xr.DataArray = history.copy(deep=True)
    modified.values[3:] = -99999
    init_times: list[np.datetime64] = [history.time.values[2]]
    original: xr.Dataset = generate_forecasts(
        history, init_times, persistence, context_length=3, horizon=2
    )
    changed: xr.Dataset = generate_forecasts(
        modified, init_times, persistence, context_length=3, horizon=2
    )
    xr.testing.assert_equal(original, changed)


def test_longitude_normalization_preserves_values() -> None:
    history: xr.DataArray = make_history().assign_coords(longitude=[-90, 0, 90])
    result: xr.Dataset = generate_forecasts(
        history, [history.time.values[2]], persistence, context_length=3, horizon=1
    )
    np.testing.assert_array_equal(result.longitude.values, [0, 90, 270])
    actual: NDArray[np.float32] = (
        result.surface_air_temperature.isel(init_time=0, lead_time=0)
        .transpose("latitude", "longitude")
        .values
    )
    np.testing.assert_array_equal(actual, history.values[2][:, [1, 2, 0]])


@pytest.mark.parametrize("parameter", ["context_length", "horizon", "batch_size"])
@pytest.mark.parametrize("value", [0, -1])
def test_rejects_nonpositive_counts(parameter: str, value: int) -> None:
    history: xr.DataArray = make_history()
    options: dict[str, int] = {"context_length": 3, "horizon": 2, "batch_size": 4}
    options[parameter] = value
    with pytest.raises(ValueError):
        generate_forecasts(history, [history.time.values[2]], persistence, **options)


def test_rejects_irregular_history() -> None:
    history: xr.DataArray = make_history().isel(time=[0, 1, 3, 4, 5, 6, 7])
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[3]], persistence, context_length=3
        )


def test_rejects_insufficient_history() -> None:
    history: xr.DataArray = make_history()
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[1]], persistence, context_length=3
        )


def test_rejects_initialization_not_in_history() -> None:
    history: xr.DataArray = make_history()
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [np.datetime64("2024-06-01T13")], persistence, context_length=2
        )


def test_rejects_empty_initializations() -> None:
    with pytest.raises(ValueError):
        generate_forecasts(make_history(), [], persistence, context_length=2)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_predictions(bad_value: float) -> None:
    history: xr.DataArray = make_history()

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        return np.full((context.shape[0], horizon), bad_value, dtype=np.float32)

    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[2]], predict, context_length=3, horizon=2
        )


@pytest.mark.parametrize("shape", [(1, 2), (4, 3), (4, 2, 1)])
def test_rejects_wrong_prediction_shape(shape: tuple[int, ...]) -> None:
    history: xr.DataArray = make_history()

    def predict(context: NDArray[np.float32], horizon: int) -> NDArray[np.float32]:
        return np.zeros(shape, dtype=np.float32)

    with pytest.raises(ValueError):
        generate_forecasts(
            history,
            [history.time.values[2]],
            predict,
            context_length=3,
            horizon=2,
            batch_size=4,
        )


def test_transposed_history_preserves_forecasts() -> None:
    history: xr.DataArray = make_history()
    init_times: list[np.datetime64] = [history.time.values[2]]
    expected: xr.Dataset = generate_forecasts(
        history, init_times, persistence, context_length=3, horizon=2
    )
    actual: xr.Dataset = generate_forecasts(
        history.transpose("longitude", "time", "latitude"),
        init_times,
        persistence,
        context_length=3,
        horizon=2,
    )
    xr.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_context(bad_value: float) -> None:
    history: xr.DataArray = make_history()
    history.values[0, 0, 0] = bad_value
    with pytest.raises(ValueError):
        generate_forecasts(
            history, [history.time.values[2]], persistence, context_length=3, horizon=2
        )


def test_netcdf_roundtrip_preserves_forecast(tmp_path: Path) -> None:
    history: xr.DataArray = make_history()
    result: xr.Dataset = generate_forecasts(
        history, [history.time.values[2]], persistence, context_length=3, horizon=2
    )
    result.to_netcdf(tmp_path / "forecast.nc")
    with xr.open_dataset(tmp_path / "forecast.nc", decode_timedelta=True) as restored:
        xr.testing.assert_equal(result, restored)
