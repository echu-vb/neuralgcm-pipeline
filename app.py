# -*- coding: utf-8 -*-
"""
app.py -- Streamlit UI for pipeline.py

Not yet run end-to-end (no GPU / no GCS network access in the environment
this was written in). Build the image and run it locally before trusting
this to work exactly as written -- see README.md.
"""

import io
import sys
import contextlib
from datetime import date, timedelta

import streamlit as st
import pandas as pd

import pipeline


st.set_page_config(page_title="NeuralGCM Extreme Rainfall", layout="wide")


# ------------------------- stdout capture -> live log panel ------------------
# pipeline.py's diagnostic print()s (NaN warnings, dropped-sample counts,
# the ensemble RNG-diversity check, etc.) matter -- this project's own
# debugging history is full of cases where trusting a result without
# reading these would have been a mistake. This captures them into the UI
# instead of losing them to a terminal nobody's watching.

class _LiveLogWriter(io.TextIOBase):
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.buffer = ""

    def write(self, s):
        self.buffer += s
        # Keep the panel from growing unbounded across a long run.
        self.placeholder.code(self.buffer[-8000:], language=None)
        return len(s)


def run_with_log(fn, *args, **kwargs):
    log_placeholder = st.empty()
    writer = _LiveLogWriter(log_placeholder)
    with contextlib.redirect_stdout(writer):
        result = fn(*args, **kwargs)
    return result


# ------------------------------ cached model load -----------------------------

@st.cache_resource(show_spinner="Loading NeuralGCM checkpoint from GCS...")
def get_model():
    return pipeline.load_model()


# --------------------------------- session state ------------------------------

if "locations" not in st.session_state:
    st.session_state.locations = dict(pipeline.LOCATIONS)

if "backtest_events" not in st.session_state:
    st.session_state.backtest_events = pd.DataFrame(pipeline.BACKTEST_EVENTS)

for key in ["climatology", "main_members_df", "main_summary_df",
            "backtest_members_df", "backtest_summary_df"]:
    if key not in st.session_state:
        st.session_state[key] = None


# ------------------------------------ sidebar ---------------------------------

st.sidebar.header("Forecast")

forecast_date = st.sidebar.date_input(
    "Forecast start date",
    value=date.today() - timedelta(days=45),
    max_value=date.today(),
    help="ERA5 needs real lag time to be available -- very recent dates will "
         "fail. Default is 45 days back as a safe starting point.",
)
if (date.today() - forecast_date).days < 30:
    st.sidebar.warning(
        "This date is less than 30 days in the past. ERA5 reanalysis often "
        "isn't available yet that recently -- the run may fail with a NaN-data "
        "error. Consider picking an older date."
    )

ensemble_size = st.sidebar.slider(
    "Ensemble size", min_value=1, max_value=20, value=pipeline.ENSEMBLE_SIZE,
    help="Number of stochastic forecast draws. Cost scales linearly with this "
         "-- each member is a full separate forecast run.",
)
st.sidebar.caption(
    f"This will run {ensemble_size} forecast{'s' if ensemble_size != 1 else ''} "
    f"sequentially per action below."
)

with st.sidebar.expander("Advanced settings"):
    accumulation_windows = st.multiselect(
        "Accumulation windows (days)",
        options=[1, 3, 5, 7, 10],
        default=pipeline.ACCUMULATION_WINDOWS_DAYS,
        help="Single-day totals alone significantly understate sustained, "
             "multi-day extreme rainfall events -- keep more than one window "
             "unless you have a specific reason not to.",
    )
    if not accumulation_windows:
        st.error("Select at least one accumulation window.")
        accumulation_windows = pipeline.ACCUMULATION_WINDOWS_DAYS

    n_realizations = st.slider(
        "Climatology realizations to pool", min_value=1, max_value=37,
        value=len(pipeline.CLIMATOLOGY_RUN_INDICES),
        help="More realizations = more stable GEV fits, at the cost of a "
             "proportionally larger one-time download. 15 was sufficient in "
             "testing to resolve unstable fits seen at 5.",
    )
    climatology_run_indices = list(range(n_realizations))

    st.markdown("**Locations**")
    locations_df = pd.DataFrame(
        [{"name": k, "lat": v[0], "lon": v[1]} for k, v in st.session_state.locations.items()]
    )
    edited_locations_df = st.data_editor(
        locations_df, num_rows="dynamic", use_container_width=True,
        key="locations_editor",
    )
    new_locations = {
        row["name"]: (float(row["lat"]), float(row["lon"]))
        for _, row in edited_locations_df.iterrows()
        if row["name"]
    }
    if new_locations != st.session_state.locations:
        st.session_state.locations = new_locations
        st.session_state.climatology = None  # stale -- force rebuild
        st.info("Locations changed -- climatology will be rebuilt on next run.")

st.sidebar.header("Backtest events")
edited_events_df = st.sidebar.data_editor(
    st.session_state.backtest_events,
    num_rows="dynamic",
    use_container_width=True,
    column_config={
        "name": st.column_config.TextColumn("Event name"),
        "location": st.column_config.SelectboxColumn(
            "Location", options=list(st.session_state.locations.keys())
        ),
        "init_date": st.column_config.TextColumn(
            "Init date (YYYY-MM-DD)", help="A few days before the event's peak."
        ),
        "forecast_days": st.column_config.NumberColumn(
            "Forecast days", min_value=1, max_value=10
        ),
    },
    key="events_editor",
)
st.session_state.backtest_events = edited_events_df


# ------------------------------------ main area --------------------------------

st.title("NeuralGCM Extreme Rainfall Comparison")
st.caption(
    "Compares a NeuralGCM forecast against a climatology built from the "
    "model's own 20-year simulation -- see README.md for methodology and "
    "known limitations before interpreting results."
)

devices, backend = pipeline.get_devices()
st.caption(f"JAX backend: **{backend}** | Devices: {devices}")
if backend == "cpu":
    st.info(
        "Running on CPU. This works but forecast steps will be noticeably "
        "slower than on GPU -- see README.md for the GPU profile if you have "
        "an NVIDIA card available."
    )

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("1. Load model", use_container_width=True):
        st.session_state.model = get_model()
        st.success("Model loaded.")

with col2:
    climatology_ready = st.session_state.get("model") is not None
    if st.button("2. Build / load climatology", use_container_width=True,
                 disabled=not climatology_ready):
        st.session_state.climatology, _ = run_with_log(
            pipeline.build_climatology,
            st.session_state.locations,
            run_indices=climatology_run_indices,
            windows=accumulation_windows,
        )
        st.success("Climatology ready.")
    if not climatology_ready:
        st.caption("Load the model first.")

with col3:
    forecast_ready = (
        st.session_state.get("model") is not None
        and st.session_state.get("climatology") is not None
    )
    st.caption("Run the forecast/backtest below once both steps above are done.")

st.divider()

tab_forecast, tab_backtest = st.tabs(["Main forecast", "Backtest"])

with tab_forecast:
    if st.button("Run main forecast", disabled=not forecast_ready):
        with st.spinner(f"Running {ensemble_size}-member ensemble forecast..."):
            ensemble_predictions = run_with_log(
                pipeline.run_forecast_ensemble,
                st.session_state.model,
                forecast_date.strftime("%Y-%m-%d"),
                pipeline.FORECAST_DAYS,
                ensemble_size=ensemble_size,
            )
            main_members_df, main_summary_df = pipeline.compare_ensemble_to_climatology(
                ensemble_predictions, st.session_state.climatology,
                st.session_state.locations, windows=accumulation_windows,
            )
            st.session_state.main_members_df = main_members_df
            st.session_state.main_summary_df = main_summary_df

    if st.session_state.main_summary_df is not None:
        st.subheader("Summary (across ensemble)")
        st.dataframe(st.session_state.main_summary_df, use_container_width=True)

        st.subheader("All ensemble members")
        st.dataframe(st.session_state.main_members_df, use_container_width=True)

        st.subheader("Plots")
        plot_window = st.selectbox(
            "Window for plots", options=accumulation_windows,
            index=len(accumulation_windows) - 1, key="main_plot_window",
        )
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig = pipeline.plot_hit_rates(st.session_state.main_summary_df, index_col="location")
            st.pyplot(fig)
        with pcol2:
            fig = pipeline.plot_ensemble_spread(
                st.session_state.main_members_df, window=plot_window, group_col="location"
            )
            st.pyplot(fig)

        plot_location = st.selectbox(
            "Location for GEV detail plot",
            options=list(st.session_state.locations.keys()), key="main_plot_location",
        )
        fig = pipeline.plot_gev_fit_ensemble(
            plot_location, plot_window, st.session_state.climatology,
            st.session_state.main_members_df,
        )
        st.pyplot(fig)

        st.download_button(
            "Download members CSV",
            st.session_state.main_members_df.to_csv(index=False),
            file_name="main_forecast_members.csv",
        )

with tab_backtest:
    events = st.session_state.backtest_events.to_dict("records")
    events = [e for e in events if e.get("name") and e.get("location") and e.get("init_date")]
    invalid = [e for e in events if e["location"] not in st.session_state.locations]
    if invalid:
        st.error(
            f"These events reference a location not in the current locations "
            f"list: {[e['name'] for e in invalid]}. Fix them in the sidebar "
            f"table before running."
        )

    if st.button("Run backtest", disabled=not forecast_ready or bool(invalid) or not events):
        with st.spinner(f"Running backtest across {len(events)} event(s), "
                         f"{ensemble_size} members each..."):
            backtest_members_df, backtest_summary_df = run_with_log(
                pipeline.run_backtest,
                st.session_state.model, st.session_state.climatology, events,
                locations=st.session_state.locations, ensemble_size=ensemble_size,
            )
            st.session_state.backtest_members_df = backtest_members_df
            st.session_state.backtest_summary_df = backtest_summary_df

    if st.session_state.backtest_summary_df is not None:
        st.subheader("Summary (across ensemble)")
        st.dataframe(st.session_state.backtest_summary_df, use_container_width=True)

        st.subheader("All ensemble members")
        st.dataframe(st.session_state.backtest_members_df, use_container_width=True)

        st.subheader("Plots")
        plot_window = st.selectbox(
            "Window for plots", options=accumulation_windows,
            index=len(accumulation_windows) - 1, key="backtest_plot_window",
        )
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            fig = pipeline.plot_hit_rates(st.session_state.backtest_summary_df, index_col="event_name")
            st.pyplot(fig)
        with pcol2:
            fig = pipeline.plot_ensemble_spread(
                st.session_state.backtest_members_df, window=plot_window, group_col="event_name"
            )
            st.pyplot(fig)

        st.download_button(
            "Download members CSV",
            st.session_state.backtest_members_df.to_csv(index=False),
            file_name="backtest_members.csv",
        )
