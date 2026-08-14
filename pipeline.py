# -*- coding: utf-8 -*-
"""
pipeline.py

This pipeline takes a short-range ensemble forecast of NeuralGCM and asks
"how extreme is this forecast relative to what this location's climate
usually produces?" Comparing the forecast to station rain-gauge records or
reanalysis data introduces a source of bias so the pipeline builds a
distribution from the model's own climatology to compare from. You are
able to specify the number of ensemble members, which locations to
forecast, and what events to backtest.

"""

import os
import gcsfs
import jax
import numpy as np
import pickle
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # no display in a container
import matplotlib.pyplot as plt

from dinosaur import horizontal_interpolation
from dinosaur import spherical_harmonic
from dinosaur import xarray_utils
from scipy.stats import genextreme
from scipy.optimize import minimize
import neuralgcm

# ============================== CONFIG ======================================

OUTPUT_DIR = os.environ.get("OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOCATIONS = {
    "Houston": (29.76, -95.37),
    "Tampa": (27.95, -82.46),
    "Abbotsford": (49.05, -122.33),
    "Raleigh": (35.78, -78.64),
    "BatonRouge": (30.45, -91.15),
    # Add / Edit desired locations
}
BACKTEST_EVENTS = [
    {"name": "Hurricane Harvey (Houston)", "location": "Houston",
     "init_date": "2017-08-23", "forecast_days": 6},
    {"name": "2016 Louisiana Floods (Baton Rouge)", "location": "BatonRouge",
     "init_date": "2016-08-10", "forecast_days": 6},
    {"name": "Nov 2021 Pacific NW Floods (Abbotsford)", "location": "Abbotsford",
     "init_date": "2021-11-12", "forecast_days": 6},
    # Add / Edit events to backtest (set init_date to be a few days before the event occurred)
]
FORECAST_START = "2026-01-20"
FORECAST_DAYS = 6  # 6 days maximum for free-tier Colab; keep as a sane default here too
CHECKPOINT_PATH = "v1_precip/stochastic_precip_2_8_deg.pkl"
CLIMATOLOGY_ZARR = (
    "gs://neuralgcm/amip_runs/v1_precip_stochastic_2_8_deg/"
    "2001-to-2021_128x64_gauss_37-level_stride3h.zarr"
)
ERA5_PATH = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
CLIMATOLOGY_RUN_INDICES = list(range(15))  # how many of the 37 simulated realizations to pool
CLIMATOLOGY_CACHE_PATH = os.path.join(OUTPUT_DIR, "climatology_annual_max_cache.nc")
ACCUMULATION_WINDOWS_DAYS = [1, 3, 5]  # multi-day rain totals
ENSEMBLE_SIZE = 10  # Number of stochastic ensemble members to run per forecast

# =============================================================================


def get_devices():
    """Call this explicitly (e.g. from app.py at startup) rather than
    relying on module-level execution -- returns (devices, backend) so the
    UI can display which backend (cpu/gpu) is actually in use."""
    return jax.devices(), jax.default_backend()


def load_model():
    # Download precipitation-capable checkpoint from Google public storage
    gcs = gcsfs.GCSFileSystem(token="anon")
    with gcs.open(f"gs://neuralgcm/models/{CHECKPOINT_PATH}", "rb") as f:
        ckpt = pickle.load(f)
    if isinstance(ckpt, dict):
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        description = ckpt.get("description", "No 'description' key found in checkpoint.")
    else:
        description = getattr(ckpt, "description", "Checkpoint has no 'description' attribute.")
    print(f"Checkpoint description: {description}")

    model = neuralgcm.PressureLevelModel.from_checkpoint(ckpt)
    return model


def nearest_grid_index(ds, lat, lon):
    # This takes the location's real world longitude/latitude and finds the closest grid cell
    lon_0_360 = lon % 360
    lat_idx = int(np.abs(ds.latitude.values - lat).argmin())
    lon_idx = int(np.abs(ds.longitude.values - lon_0_360).argmin())
    actual_lat = float(ds.latitude.values[lat_idx])
    actual_lon = float(ds.longitude.values[lon_idx])
    return lat_idx, lon_idx, actual_lat, actual_lon


def inspect_prediction_timedelta_axis(ds, precip_var, lat, lon, run_index):
    # Prints 40 raw precipitation values
    lat_idx, lon_idx, alat, alon = nearest_grid_index(ds, lat, lon)
    print(f"\n--- Inspecting run {run_index}, grid cell ({alat:.2f}, {alon:.2f}) ---")
    sample = ds[precip_var].isel(
        time=run_index, surface=0, latitude=lat_idx, longitude=lon_idx,
        prediction_timedelta=slice(0, 40),
    )
    print(sample.values)


def daily_precip_from_cumulative(cumulative_da, time_dim="time"):
    # We must run this to get a single value for precipitation in a day because the raw data
    # doesn't tell you "how much rain fell today" -- it tells you "how much rain has fallen
    # in total since the very start."
    increment = cumulative_da.diff(time_dim)
    increment = increment.where(
        increment >= 0, cumulative_da.isel({time_dim: slice(1, None)})
    )
    daily = increment.resample({time_dim: "1D"}).sum()
    return daily


def increments_from_cumulative(cumulative_da, time_dim="time"):
    # Turn a running total into "how much fell in this step", but WITHOUT the day-grouping
    increment = cumulative_da.diff(time_dim)
    increment = increment.where(
        increment >= 0, cumulative_da.isel({time_dim: slice(1, None)})
    )
    return increment


def fit_gev(annual_max_values, shape_bounds=(-0.5, 0.5)):
    # Fit a GEV distribution to a series of annual maxima via MLE, with the shape parameter
    # constrained to a physically plausible range.
    data = np.asarray(annual_max_values)
    data = data[np.isfinite(data)]
    if data.size == 0:
        raise RuntimeError(
            "fit_gev() received zero finite samples -- there is no usable "
            "climatology data for this location/window. Check the grid "
            "cell (likely over water or otherwise missing data)."
        )

    try:
        c0, loc0, scale0 = genextreme.fit(data)
        c0 = float(np.clip(c0, shape_bounds[0], shape_bounds[1]))
    except Exception:
        loc0 = float(data.mean())
        scale0 = float(data.std()) or 1.0
        c0 = 0.0

    def neg_log_likelihood(params):
        c, loc, scale = params
        if scale <= 0:
            return np.inf
        logpdf = genextreme.logpdf(data, c, loc=loc, scale=scale)
        if not np.all(np.isfinite(logpdf)):
            return np.inf
        return -np.sum(logpdf)

    bounds = [shape_bounds, (None, None), (1e-6, None)]
    result = minimize(
        neg_log_likelihood, x0=[c0, loc0, scale0],
        method="L-BFGS-B", bounds=bounds,
    )

    if not result.success:
        print(f"WARNING: constrained GEV fit did not converge cleanly "
              f"({result.message}). Falling back to unconstrained "
              f"genextreme.fit(), which may fall outside the plausible "
              f"shape range -- treat with caution.")
        return genextreme.fit(data)

    shape, loc, scale = result.x
    return shape, loc, scale


def gev_percentile_and_return_period(forecast_value, gev_params):
    # Given fitted GEV, return forecast percentile and implied return period
    shape, loc, scale = gev_params
    cdf = genextreme.cdf(forecast_value, shape, loc, scale)
    percentile = 100 * cdf
    exceedance_prob = 1 - cdf
    return_period_years = (1 / exceedance_prob) if exceedance_prob > 0 else float("inf")
    return percentile, return_period_years


def build_climatology(locations, run_indices=None, windows=None, cache_path=None):
    # Pulls precipitation data from NeuralGCM's public 20-year AMIP archive, converts the raw
    # cumulative precipitation field into daily totals, and computes annual-maximum
    # daily/multi-day rainfall at the grid cell nearest each target location.
    #
    # FIX: defaults are None here, resolved against module constants below,
    # instead of `run_indices=CLIMATOLOGY_RUN_INDICES` etc directly in the
    # signature -- see module docstring for why this matters for Streamlit.
    run_indices = run_indices if run_indices is not None else CLIMATOLOGY_RUN_INDICES
    windows = windows if windows is not None else ACCUMULATION_WINDOWS_DAYS
    cache_path = cache_path if cache_path is not None else CLIMATOLOGY_CACHE_PATH

    if os.path.exists(cache_path):
        print(f"Loading cached climatology from {cache_path} "
              f"(delete this file to force a rebuild)")
        cached = xr.open_dataarray(cache_path)
        climatology = {}
        for name in locations:
            if name not in cached.location.values:
                raise RuntimeError(
                    f"'{name}' is not in the cached climatology "
                    f"({cache_path}). Delete the cache file to rebuild it "
                    f"with the full current LOCATIONS/windows set."
                )
            climatology[name] = {
                "grid_lat": float(cached.sel(location=name)["grid_lat"].values),
                "grid_lon": float(cached.sel(location=name)["grid_lon"].values),
                "annual_max_by_window": {
                    w: cached.sel(location=name, window=w)
                    for w in windows if w in cached.window.values
                },
            }
        return climatology, None

    ds = xr.open_zarr(CLIMATOLOGY_ZARR, chunks={"prediction_timedelta": 500},
                       storage_options=dict(token="anon"))
    print("\n=== AMIP climatology dataset ===")
    print(ds)

    precip_candidates = [v for v in ds.data_vars if "precip" in v.lower()]
    if not precip_candidates:
        raise RuntimeError("No variable with 'precip' in its name found.")
    precip_var = precip_candidates[0]
    print(f"Using climatology precipitation variable: {precip_var}")

    first_name, (first_lat, first_lon) = next(iter(locations.items()))
    inspect_prediction_timedelta_axis(ds, precip_var, first_lat, first_lon, run_indices[0])

    names, lat_idxs, lon_idxs, actual_lats, actual_lons = [], [], [], [], []
    for name, (lat, lon) in locations.items():
        lat_idx, lon_idx, alat, alon = nearest_grid_index(ds, lat, lon)
        names.append(name)
        lat_idxs.append(lat_idx)
        lon_idxs.append(lon_idx)
        actual_lats.append(alat)
        actual_lons.append(alon)

    lat_da = xr.DataArray(lat_idxs, dims="location", coords={"location": names})
    lon_da = xr.DataArray(lon_idxs, dims="location", coords={"location": names})
    run_da = xr.DataArray(run_indices, dims="run", coords={"run": run_indices})

    print(f"\nPulling {len(names)} locations across {len(run_indices)} "
          f"realizations in a single combined request...")
    subset = ds[precip_var].isel(
        time=run_da, surface=0, latitude=lat_da, longitude=lon_da
    )
    subset = subset.compute()
    per_window_per_run = {w: [] for w in windows}
    for run_idx in run_indices:
        run_slice = subset.sel(run=run_idx)
        base_time = ds.time.values[run_idx]
        real_time = base_time + ds.prediction_timedelta.values.astype("timedelta64[h]")
        run_slice = run_slice.assign_coords(time=("prediction_timedelta", real_time))
        run_slice = run_slice.swap_dims({"prediction_timedelta": "time"})

        daily = daily_precip_from_cumulative(run_slice, time_dim="time")

        for w in windows:
            if w == 1:
                rolling_total = daily
            else:
                rolling_total = daily.rolling(time=w, min_periods=w).sum()
            annual_max = rolling_total.groupby("time.year").max()  # Block maxima
            per_window_per_run[w].append(annual_max)

    pooled_by_window = []
    for w in windows:
        pooled = xr.concat(per_window_per_run[w], dim="run")
        pooled = pooled.stack(sample=("run", "year"))
        pooled = pooled.reset_index("sample", drop=True)
        pooled = pooled.expand_dims(window=[w])
        pooled_by_window.append(pooled)

    combined = xr.concat(pooled_by_window, dim="window")
    combined = combined.assign_coords(
        grid_lat=("location", actual_lats),
        grid_lon=("location", actual_lons),
    )
    combined.to_netcdf(cache_path)
    print(f"Cached pooled multi-window climatology to {cache_path}")

    climatology = {}
    for name, alat, alon in zip(names, actual_lats, actual_lons):
        by_window = {}
        for w in windows:
            series = combined.sel(location=name, window=w)
            by_window[w] = series
            finite = series.values[np.isfinite(series.values)]
            if finite.size == 0:
                print(f"{name}, {w}-day window: 0 usable pooled samples "
                      f"(all NaN) -- check this location's grid cell / "
                      f"data availability before trusting downstream results.")
            else:
                print(f"{name}, {w}-day window: {len(finite)} usable pooled samples, "
                      f"mean={finite.mean():.5f} m, max={finite.max():.5f} m")
        climatology[name] = {
            "grid_lat": alat,
            "grid_lon": alon,
            "annual_max_by_window": by_window,
        }

    return climatology, precip_var


def run_forecast(model, init_date, days, rng_seed=42):
    # Initializes a forecast from real ERA5 reanalysis at a chosen date, regrids it to
    # NeuralGCM's native grid, and runs the model forward using the stochastic
    # precipitation checkpoint.
    full_era5 = xr.open_zarr(ERA5_PATH, chunks=None, storage_options=dict(token="anon"))

    needed_vars = model.input_variables + model.forcing_variables
    era5_slice = full_era5[needed_vars].sel(time=init_date, method="nearest").compute()
    era5_slice = era5_slice.expand_dims("time")

    for var in needed_vars:
        frac_nan = float(np.isnan(era5_slice[var].values).mean())
        if frac_nan > 0.99:
            raise RuntimeError(
                f"ERA5 variable '{var}' at {init_date} is {frac_nan*100:.1f}% NaN. "
                f"This almost always means the requested date is too recent -- "
                f"ERA5 reanalysis typically lags 5+ days (preliminary) to several "
                f"months (final, quality-controlled) behind the present. Pick an "
                f"init_date at least a month in the past and retry."
            )
        elif frac_nan > 0.05:
            print(f"WARNING: '{var}' at {init_date} is {frac_nan*100:.1f}% NaN "
                  f"-- check this before trusting downstream results.")

    era5_grid = spherical_harmonic.Grid(
        latitude_nodes=full_era5.sizes["latitude"],
        longitude_nodes=full_era5.sizes["longitude"],
        latitude_spacing=xarray_utils.infer_latitude_spacing(full_era5.latitude),
        longitude_offset=xarray_utils.infer_longitude_offset(full_era5.longitude),
    )
    regridder = horizontal_interpolation.ConservativeRegridder(
        era5_grid, model.data_coords.horizontal, skipna=True
    )
    eval_era5 = xarray_utils.regrid(era5_slice, regridder)
    eval_era5 = xarray_utils.fill_nan_with_nearest(eval_era5)

    inputs = model.inputs_from_xarray(eval_era5.isel(time=0))
    input_forcings = model.forcings_from_xarray(eval_era5.isel(time=0))

    rng_key = jax.random.key(rng_seed)
    initial_state = model.encode(inputs, input_forcings, rng_key)
    all_forcings = model.forcings_from_xarray(eval_era5.head(time=1))

    inner_steps = 24  # save output every 24 hours -> daily
    outer_steps = days + 1
    timedelta = np.timedelta64(1, "h") * inner_steps
    times = np.arange(outer_steps) * inner_steps

    final_state, predictions = model.unroll(
        initial_state,
        all_forcings,
        steps=outer_steps,
        timedelta=timedelta,
        start_with_input=True,
    )
    predictions_ds = model.data_to_xarray(predictions, times=times)

    print("\n=== Forecast output dataset ===")
    print(predictions_ds)
    return predictions_ds


def run_forecast_ensemble(model, init_date, days, ensemble_size=None, base_seed=42):
    # Run the SAME forecast multiple times with different RNG seeds to get a spread of
    # plausible outcomes instead of trusting one trajectory.
    ensemble_size = ensemble_size if ensemble_size is not None else ENSEMBLE_SIZE

    ensemble_predictions = []
    for member in range(ensemble_size):
        seed = base_seed + member
        print(f"\n--- Running ensemble member {member}/{ensemble_size} (seed={seed}) ---")
        predictions_ds = run_forecast(model, init_date, days, rng_seed=seed)
        ensemble_predictions.append(predictions_ds)

    if ensemble_size >= 2:
        var_name = [v for v in ensemble_predictions[0].data_vars if "precip" in v.lower()][0]
        diff_std = float((ensemble_predictions[0][var_name] - ensemble_predictions[1][var_name]).values.std())
        print(f"\nStd of member-0-vs-member-1 precip difference: {diff_std:.8f}")
        if diff_std < 1e-6:
            print("WARNING: near-zero difference between members -- the RNG "
                  "seed does not appear to be reaching the model's stochastic "
                  "physics. Ensemble spread below is NOT meaningful until "
                  "this is resolved.")

    return ensemble_predictions


def compare_forecast_to_climatology(predictions_ds, climatology, locations, windows=None):
    # Extracts the forecast's precipitation at each location's grid cell, and reports where
    # it falls relative to the fitted GEV climatology -- both as a percentile and as an
    # implied return period (e.g. "roughly a 1-in-40-year event").
    windows = windows if windows is not None else ACCUMULATION_WINDOWS_DAYS

    precip_candidates = [v for v in predictions_ds.data_vars if "precip" in v.lower()]
    if not precip_candidates:
        raise RuntimeError("No precip variable found in forecast output.")
    forecast_precip_var = precip_candidates[0]

    rows = []
    for name, (lat, lon) in locations.items():
        lat_idx, lon_idx, alat, alon = nearest_grid_index(predictions_ds, lat, lon)
        cell = predictions_ds[forecast_precip_var].isel(latitude=lat_idx, longitude=lon_idx)
        cell = cell.as_numpy()
        daily_forecast = increments_from_cumulative(cell) if "time" in cell.dims else cell

        if name in ("Raleigh", "Tampa"):
            print(f"DEBUG {name} daily forecast values: {daily_forecast.values}")

        for w in windows:
            rolling_forecast = daily_forecast if w == 1 else daily_forecast.rolling(time=w, min_periods=w).sum()

            valid_values = rolling_forecast.values[np.isfinite(rolling_forecast.values)]
            if valid_values.size == 0:
                print(f"{name}, {w}-day window: forecast too short for a "
                      f"full {w}-day window, skipping.")
                continue
            peak_forecast_value = float(valid_values.max())

            if name not in climatology or w not in climatology[name]["annual_max_by_window"]:
                raise RuntimeError(f"No climatology available for {name}, window={w}")

            hist_raw = climatology[name]["annual_max_by_window"][w].values
            n_nan = int(np.isnan(hist_raw).sum())
            hist = hist_raw[np.isfinite(hist_raw)]
            if n_nan > 0:
                print(f"{name}, {w}-day window: dropped {n_nan}/{len(hist_raw)} "
                      f"non-finite climatology samples before fitting")

            empirical_percentile = float((hist < peak_forecast_value).mean() * 100)
            gev_shape, gev_loc, gev_scale = fit_gev(hist)
            gev_percentile, return_period_years = gev_percentile_and_return_period(
                peak_forecast_value, (gev_shape, gev_loc, gev_scale)
            )

            rows.append({
                "location": name,
                "grid_lat": alat,
                "grid_lon": alon,
                "window_days": w,
                "forecast_peak_precip_mm": peak_forecast_value * 1000,
                "climatology_samples": len(hist),
                "climatology_mean_annual_max_mm": float(hist.mean()) * 1000,
                "empirical_percentile": empirical_percentile,
                "gev_percentile": gev_percentile,
                "gev_implied_return_period_years": return_period_years,
                "gev_shape": gev_shape,
                "gev_loc_mm": gev_loc * 1000,
                "gev_scale_mm": gev_scale * 1000,
            })

    return pd.DataFrame(rows)


def compare_ensemble_to_climatology(ensemble_predictions, climatology, locations, windows=None):
    # Run the single-forecast comparison on every ensemble member, then summarize the
    # SPREAD across members instead of reporting one number.
    per_member_dfs = []
    for i, predictions_ds in enumerate(ensemble_predictions):
        df = compare_forecast_to_climatology(predictions_ds, climatology, locations, windows=windows)
        df["ensemble_member"] = i
        per_member_dfs.append(df)

    all_members = pd.concat(per_member_dfs, ignore_index=True)

    summary = all_members.groupby(["location", "window_days"]).agg(
        forecast_peak_precip_mm_mean=("forecast_peak_precip_mm", "mean"),
        forecast_peak_precip_mm_min=("forecast_peak_precip_mm", "min"),
        forecast_peak_precip_mm_max=("forecast_peak_precip_mm", "max"),
        gev_return_period_years_mean=("gev_implied_return_period_years", "mean"),
        gev_return_period_years_min=("gev_implied_return_period_years", "min"),
        gev_return_period_years_max=("gev_implied_return_period_years", "max"),
        pct_members_above_10yr_return=("gev_implied_return_period_years",
                                        lambda x: float((x >= 10).mean() * 100)),
    ).reset_index()

    return all_members, summary


def run_backtest(model, climatology, events, ensemble_size=None):
    # Reuses the same forecast/comparison machinery, but initializes from a real historical
    # date shortly before a documented extreme event, to test the pipeline against a known
    # outcome rather than an arbitrary week.
    ensemble_size = ensemble_size if ensemble_size is not None else ENSEMBLE_SIZE

    all_member_rows = []
    summary_rows = []
    for event in events:
        name = event["name"]
        loc_key = event["location"]
        if loc_key not in LOCATIONS:
            raise RuntimeError(f"Backtest location '{loc_key}' not in LOCATIONS.")

        print(f"\n=== Backtesting: {name} (init {event['init_date']}) ===")
        ensemble_predictions = run_forecast_ensemble(
            model, event["init_date"], event["forecast_days"], ensemble_size=ensemble_size
        )
        single_location = {loc_key: LOCATIONS[loc_key]}
        all_members, summary = compare_ensemble_to_climatology(
            ensemble_predictions, climatology, single_location
        )
        all_members["event_name"] = name
        all_members["init_date"] = event["init_date"]
        summary["event_name"] = name
        summary["init_date"] = event["init_date"]
        all_member_rows.append(all_members)
        summary_rows.append(summary)

    return pd.concat(all_member_rows, ignore_index=True), pd.concat(summary_rows, ignore_index=True)


# Column reference (see also the README):
# grid_lat / grid_lon: actual coordinates of the nearest model grid cell for the city
# window_days: specifies a single day (1) or multi-day (3,5) total precipitation
# forecast_peak_precip_mm: the single wettest window of the specified forecast, in mm
# climatology_samples: how many pooled samples of historical annual-maximum data we're comparing against
# climatology_mean_annual_max_mm: how much rain typically falls on the location's WORST day of a typical year
# empirical_percentile: of the historical worst days of the year, what percentage were LESS rainy than this forecast
# gev_percentile: fits the pooled samples to a GEV distribution and gives a continuous percentile
# gev_implied_return_period_years: once every x year event, so at least once every one year
# gev_shape, gev_loc_mm, gev_scale_mm: define the shape of the fitted curve
# (gev_shape should fall roughly between -0.5 and 0.5 as a sanity check)


# ------------------------------- Plotting -----------------------------------
# Each function now RETURNS a matplotlib Figure instead of calling
# plt.show() (a no-op headless). app.py renders it with st.pyplot(fig).

def plot_gev_fit(location, window, climatology, forecast_value_mm=None, forecast_label=None):
    hist_raw = climatology[location]["annual_max_by_window"][window].values
    hist = hist_raw[np.isfinite(hist_raw)] * 1000  # convert to mm
    if hist.size == 0:
        raise RuntimeError(
            f"No finite climatology samples for {location}, {window}-day "
            f"window -- nothing to plot."
        )

    shape, loc, scale = fit_gev(hist_raw[np.isfinite(hist_raw)])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hist, bins=25, density=True, alpha=0.5, color="steelblue",
            label=f"{len(hist)} pooled climatology samples")

    x = np.linspace(hist.min() * 0.5, max(hist.max(), forecast_value_mm or 0) * 1.15, 500)
    pdf = genextreme.pdf(x / 1000, shape, loc=loc, scale=scale) / 1000
    ax.plot(x, pdf, color="black", linewidth=2, label="Fitted GEV")

    if forecast_value_mm is not None:
        ax.axvline(forecast_value_mm, color="crimson", linewidth=2, linestyle="--",
                    label=forecast_label or f"Forecast: {forecast_value_mm:.1f}mm")

    ax.set_xlabel(f"{window}-day accumulated precipitation (mm)")
    ax.set_ylabel("Density")
    ax.set_title(f"{location}: {window}-day climatology vs. forecast")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_gev_fit_ensemble(location, window, climatology, members_df):
    # Same GEV histogram as plot_gev_fit(), but marks every ensemble member's
    # forecast value as a separate vertical line instead of one.
    hist_raw = climatology[location]["annual_max_by_window"][window].values
    hist = hist_raw[np.isfinite(hist_raw)] * 1000
    if hist.size == 0:
        raise RuntimeError(
            f"No finite climatology samples for {location}, {window}-day "
            f"window -- nothing to plot."
        )
    shape, loc, scale = fit_gev(hist_raw[np.isfinite(hist_raw)])

    forecasts = members_df[
        (members_df["location"] == location) &
        (members_df["window_days"] == window)
    ]["forecast_peak_precip_mm"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(hist, bins=25, density=True, alpha=0.5, color="steelblue",
            label=f"{len(hist)} pooled climatology samples")

    x_max = max(hist.max(), forecasts.max() if len(forecasts) else hist.max())
    x = np.linspace(hist.min() * 0.5, x_max * 1.15, 500)
    pdf = genextreme.pdf(x / 1000, shape, loc=loc, scale=scale) / 1000
    ax.plot(x, pdf, color="black", linewidth=2, label="Fitted GEV")

    for i, f in enumerate(forecasts):
        ax.axvline(f, color="crimson", alpha=0.5, linewidth=1.5,
                    label="Ensemble members" if i == 0 else None)

    ax.set_xlabel(f"{window}-day accumulated precipitation (mm)")
    ax.set_ylabel("Density")
    ax.set_title(f"{location}: {window}-day climatology vs. {len(forecasts)}-member ensemble")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_ensemble_spread(members_df, window=5, group_col="event_name"):
    subset = members_df[members_df["window_days"] == window]
    groups = subset[group_col].unique()

    fig, ax = plt.subplots(figsize=(9, 5))
    positions = range(len(groups))
    data = [subset[subset[group_col] == g]["gev_implied_return_period_years"].values
            for g in groups]

    ax.boxplot(data, positions=positions, widths=0.5, showfliers=True)
    for i, d in enumerate(data):
        jitter = np.random.normal(0, 0.05, size=len(d))
        ax.scatter(np.full(len(d), i) + jitter, d, alpha=0.6, color="steelblue", zorder=3)

    ax.axhline(10, color="crimson", linestyle="--", linewidth=1, label="10-year threshold")
    ax.set_yscale("log")
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(g).split(" (")[0] for g in groups], rotation=15, ha="right")
    ax.set_ylabel("Implied return period (years, log scale)")
    ax.set_title(f"Ensemble spread of return periods, {window}-day window")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_hit_rates(summary_df, index_col="event_name"):
    pivot = summary_df.pivot(index=index_col, columns="window_days",
                              values="pct_members_above_10yr_return")

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("% of ensemble members > 10-year return period")
    ax.set_xlabel("")
    ax.set_title(f"Hit rate by {index_col} and accumulation window")
    ax.set_xticklabels([str(l.get_text()).split(" (")[0] for l in ax.get_xticklabels()],
                        rotation=15, ha="right")
    ax.legend(title="Window (days)")
    plt.tight_layout()
    return fig