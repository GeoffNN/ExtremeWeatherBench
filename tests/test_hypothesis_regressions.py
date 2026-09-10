"""Deterministic repros for defects found by test_hypothesis_fixtures.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from extremeweatherbench import cases, inputs, metrics, regions, utils
from extremeweatherbench.sources import xarray_dataset


def test_convert_init_time_to_valid_time_coerces_int_lead_time_to_hours():
    """Int-hours lead_time is added to init_time as hours, not nanoseconds."""
    init_time = pd.date_range("2021-06-20", periods=2, freq="6h")
    lead_time = np.array([0, 6, 12])
    ds = xr.Dataset(
        {
            "surface_air_temperature": (
                ["init_time", "lead_time"],
                np.arange(6.0).reshape(2, 3),
            )
        },
        coords={"init_time": init_time, "lead_time": lead_time},
    )
    result = utils.convert_init_time_to_valid_time(ds)
    lead_hours = pd.to_timedelta(lead_time, unit="h").to_numpy()
    # Overlapping init_time/lead_time combinations share a valid_time slot,
    # so the result axis is the unique set of combined times, not all pairs.
    combos = (init_time.values[:, None] + lead_hours[None, :]).ravel()
    expected = np.sort(np.unique(combos))
    actual = np.sort(result.valid_time.values)
    np.testing.assert_array_equal(actual, expected)


def test_check_for_spatial_data_handles_antimeridian_seam():
    longitude = np.array([170.0, 175.0, -180.0, -175.0, -170.0])
    latitude = np.array([20.0, 30.0, 40.0])
    ds = xr.Dataset(
        {"v": (["latitude", "longitude"], np.zeros((3, 5)))},
        coords={"latitude": latitude, "longitude": longitude},
    )
    region = regions.BoundingBoxRegion(
        latitude_min=20.0,
        latitude_max=40.0,
        longitude_min=150.0,
        longitude_max=-100.0,
    )
    assert xarray_dataset.check_for_spatial_data(ds, region)


def _global_grid(step: float, lon_min: float = 0.0) -> xr.Dataset:
    """Build a global grid whose values vary smoothly with longitude."""
    longitude = np.arange(lon_min, lon_min + 360.0, step)
    latitude = np.arange(40.0, 76.0, step)
    field = np.sin(np.deg2rad(longitude))[None, :] * np.ones((latitude.size, 1))
    return xr.Dataset(
        {"v": (["latitude", "longitude"], field)},
        coords={"latitude": latitude, "longitude": longitude},
    )


# Cases 125 and 143 are the events.yaml regions that wrap the 0/360 seam.
@pytest.mark.parametrize(("case_id", "span"), [(125, 32.4), (143, 19.8)])
def test_meridian_crossing_case_mask_returns_contiguous_longitudes(case_id, span):
    """A region wrapping 0/360 masks to one unbroken longitude block.

    The OR-mask branch selects the right cells but used to leave them in
    raw ascending order, splitting the region into a low block and a high
    block with the rest of the globe as a gap between them.
    """
    step = 0.25
    event = next(case for case in cases.load_cases() if case.case_id_number == case_id)

    result = event.location.mask(_global_grid(step))

    longitude = result.longitude.values
    np.testing.assert_allclose(np.diff(longitude), step)
    # Clipping to grid cells can cost up to one step at either end.
    assert longitude.max() - longitude.min() == pytest.approx(span, abs=2 * step)


def test_antimeridian_region_mask_returns_contiguous_longitudes():
    """A region wrapping -180/180 masks to one unbroken block.

    Such a region is a MultiPolygon with a lobe against either side of
    the seam, so its total_bounds span the globe and the mask used to
    keep every longitude in the dataset.
    """
    step = 0.25
    grid = _global_grid(step, lon_min=-180.0)
    region = regions.BoundingBoxRegion.create_region(
        latitude_min=45.0,
        latitude_max=70.0,
        longitude_min=175.0,
        longitude_max=-175.0,
    )

    result = region.mask(grid)

    longitude = result.longitude.values
    assert longitude.size < grid.sizes["longitude"]
    np.testing.assert_allclose(np.diff(longitude), step)
    assert longitude.max() - longitude.min() == pytest.approx(10.0, abs=2 * step)


def test_meridian_crossing_grid_alignment_survives_seam():
    """Aligning two grids across the 0/360 seam preserves the field.

    Guards the longitude convention the mask now returns: forecast and
    target are masked by the same wrapping region at different
    resolutions, so alignment has to interpolate across the seam.
    """
    event = next(case for case in cases.load_cases() if case.case_id_number == 125)
    valid_time = pd.date_range("2022-01-14", periods=1, freq="6h")

    def masked(step):
        grid = event.location.mask(_global_grid(step))
        return grid.expand_dims(valid_time=valid_time)

    forecast, target = inputs.align_forecast_to_target(
        masked(0.25), masked(0.5), method="linear"
    )

    aligned = forecast["v"].isel(valid_time=0, latitude=0).values
    expected = np.sin(np.deg2rad(target.longitude.values))
    np.testing.assert_allclose(aligned, expected, atol=1e-6)


def _build_forecast_target(target_values: np.ndarray) -> tuple:
    valid_time = pd.date_range("2021-06-20", periods=3, freq="6h")
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])
    forecast = xr.DataArray(
        np.arange(12.0).reshape(3, 2, 2),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    target = xr.DataArray(
        target_values,
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    return forecast, target


def test_maximum_mean_absolute_error_returns_nan_on_all_nan_target():
    forecast, target = _build_forecast_target(np.full((3, 2, 2), np.nan))
    metric = metrics.MaximumMeanAbsoluteError()
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())


def test_minimum_mean_absolute_error_returns_nan_on_all_nan_target():
    forecast, target = _build_forecast_target(np.full((3, 2, 2), np.nan))
    metric = metrics.MinimumMeanAbsoluteError()
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())


def test_duration_mean_error_returns_nan_on_empty_valid_time_axis():
    valid_time = pd.DatetimeIndex([], dtype="datetime64[ns]")
    lat = np.array([10.0, 20.0])
    lon = np.array([100.0, 110.0])
    forecast = xr.DataArray(
        np.empty((0, 2, 2)),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    target = xr.DataArray(
        np.empty((0, 2, 2)),
        dims=["valid_time", "latitude", "longitude"],
        coords={"valid_time": valid_time, "latitude": lat, "longitude": lon},
    )
    metric = metrics.DurationMeanError(threshold_criteria=273.0)
    result = metric.compute_metric(forecast, target)
    assert bool(result.isnull().all())
