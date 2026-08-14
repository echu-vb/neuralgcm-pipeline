# main.py
from pipeline import (
    get_devices, load_model, build_climatology,
    run_forecast_ensemble, compare_ensemble_to_climatology,
    run_backtest, LOCATIONS, BACKTEST_EVENTS,
    FORECAST_START, FORECAST_DAYS, OUTPUT_DIR,
)

if __name__ == "__main__":
    devices, backend = get_devices()
    print(f"Running on: {backend} ({devices})")

    model = load_model()
    climatology, _ = build_climatology(LOCATIONS)

    ensemble = run_forecast_ensemble(model, FORECAST_START, FORECAST_DAYS)
    all_members, summary = compare_ensemble_to_climatology(ensemble, climatology, LOCATIONS)
    summary.to_csv(f"{OUTPUT_DIR}/forecast_summary.csv", index=False)

    backtest_members, backtest_summary = run_backtest(model, climatology, BACKTEST_EVENTS)
    backtest_summary.to_csv(f"{OUTPUT_DIR}/backtest_summary.csv", index=False)

    print("Done. Results written to", OUTPUT_DIR)