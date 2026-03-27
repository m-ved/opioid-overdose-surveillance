"""
Opioid Overdose Surveillance Dashboard
Visualizes REAL data from CDC VSRR, Census ACS, and CDC WONDER.
Designed for non-technical users with plain-language explanations.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import DATA_DIR

st.set_page_config(
    page_title="Opioid Overdose Surveillance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

REAL_DATA_DIR = DATA_DIR / "real"

# Human-readable drug names
DRUG_LABELS = {
    "Synthetic opioids, excl. methadone (T40.4)": "💀 Fentanyl & Synthetic Opioids",
    "Heroin (T40.1)": "💉 Heroin",
    "Natural & semi-synthetic opioids (T40.2)": "💊 Prescription Painkillers (Oxy, Hydro)",
    "Methadone (T40.3)": "🏥 Methadone",
    "Cocaine (T40.5)": "❄️ Cocaine",
    "Psychostimulants with abuse potential (T43.6)": "⚡ Methamphetamine & Stimulants",
    "Opioids (T40.0-T40.4,T40.6)": "🔴 All Opioids Combined",
    "Number of Drug Overdose Deaths": "📊 Total Drug Overdose Deaths",
}

# US state populations (2022 Census estimates) for per-capita calculations
STATE_POPULATIONS = {
    "AL": 5074296, "AK": 733583, "AZ": 7359197, "AR": 3045637, "CA": 39029342,
    "CO": 5839926, "CT": 3626205, "DE": 1018396, "DC": 671803, "FL": 22244823,
    "GA": 10912876, "HI": 1440196, "ID": 1939033, "IL": 12582032, "IN": 6833037,
    "IA": 3200517, "KS": 2937150, "KY": 4512310, "LA": 4590241, "ME": 1385340,
    "MD": 6164660, "MA": 6981974, "MI": 10037261, "MN": 5717184, "MS": 2940057,
    "MO": 6177957, "MT": 1122867, "NE": 1967923, "NV": 3177772, "NH": 1395231,
    "NJ": 9261699, "NM": 2113344, "NY": 19677151, "NC": 10698973, "ND": 779261,
    "OH": 11756058, "OK": 4019800, "OR": 4240137, "PA": 12972008, "RI": 1093734,
    "SC": 5282634, "SD": 909824, "TN": 7051339, "TX": 30029572, "UT": 3380800,
    "VT": 647064, "VA": 8642274, "WA": 7785786, "WV": 1775156, "WI": 5892539,
    "WY": 576851,
}

# ─── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                padding: 15px; border-radius: 10px; border: 1px solid #30475e; }
    .stMetric label { color: #a0a0b0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] { color: #e8e8e8 !important; }
    h1, h2, h3 { color: #e8e8e8 !important; }
    .block-container { padding-top: 2rem; }
    .callout-box {
        background: linear-gradient(135deg, #1a2332 0%, #1e2d3d 100%);
        border-left: 4px solid #4ecdc4;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 20px 0;
        font-size: 0.95rem;
        color: #c8d6e5;
    }
    .warning-box {
        background: linear-gradient(135deg, #2d1a1a 0%, #3d1e1e 100%);
        border-left: 4px solid #ff6b6b;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 20px 0;
        color: #e8c8c8;
    }
    .help-box {
        background: linear-gradient(135deg, #1a2d1a 0%, #1e3d1e 100%);
        border-left: 4px solid #6bff6b;
        padding: 15px 20px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 20px 0;
        color: #c8e8c8;
    }
</style>
""", unsafe_allow_html=True)

# ─── Helper ──────────────────────────────────────────────────────────────────────

def human_drug_name(raw):
    return DRUG_LABELS.get(raw, raw)

def callout(text, style="callout-box"):
    st.markdown(f'<div class="{style}">{text}</div>', unsafe_allow_html=True)

# ─── Data Loading ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def load_vsrr():
    path = REAL_DATA_DIR / "cdc_vsrr_overdose.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    month_map = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8,
        "September": 9, "October": 10, "November": 11, "December": 12
    }
    df["month_num"] = df["month"].map(month_map)
    df["date"] = pd.to_datetime(df[["year", "month_num"]].rename(columns={"month_num": "month"}).assign(day=1))
    df["drug_name"] = df["indicator"].map(human_drug_name)
    return df

@st.cache_data(ttl=3600)
def load_census():
    path = REAL_DATA_DIR / "census_acs_socioeconomic.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)

@st.cache_data(ttl=3600)
def load_wonder():
    path = REAL_DATA_DIR / "cdc_wonder_overdose.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


vsrr = load_vsrr()
census = load_census()
wonder = load_wonder()

if vsrr.empty and census.empty:
    st.error("⚠️ No data found. Run `python run.py` first to fetch data from CDC/Census APIs.")
    st.stop()

# ─── Sidebar ─────────────────────────────────────────────────────────────────────

st.sidebar.title("🏥 Opioid Surveillance")
st.sidebar.markdown("---")

page = st.sidebar.radio("📊 Dashboard", [
    "🌍 National Overview",
    "📈 Overdose Trends",
    "🗺️ State Comparison",
    "💊 Drug Breakdown",
    "🏘️ Vulnerability Map",
    "🔍 Look Up Your Zip Code",
])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🆘 Get Help Now")
st.sidebar.markdown("**SAMHSA Helpline (24/7)**")
st.sidebar.markdown("📞 [1-800-662-4357](tel:18006624357)")
st.sidebar.markdown("💬 [findtreatment.gov](https://findtreatment.gov)")
st.sidebar.markdown("🏥 [Naloxone Finder](https://www.naloxoneforall.org)")
st.sidebar.markdown("---")
st.sidebar.caption("📡 Data Sources:")
st.sidebar.caption(f"CDC VSRR: {len(vsrr):,} records")
st.sidebar.caption(f"Census ACS: {len(census):,} zip codes")
st.sidebar.caption("All data is real & publicly available.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: National Overview
# ═══════════════════════════════════════════════════════════════════════════════

if page == "🌍 National Overview":
    st.title("🌍 National Opioid Overdose Overview")
    callout(
        "📖 <strong>What is this?</strong> This page shows how many Americans are dying from "
        "drug overdoses each year. The numbers come from death certificates reported to the CDC "
        "and are updated every month. The '12-month rolling total' smooths out seasonal variation "
        "so you can see the real trend."
    )

    if not vsrr.empty:
        us_total = vsrr[
            (vsrr["indicator"] == "Number of Drug Overdose Deaths") &
            (vsrr["period"] == "12 month-ending") &
            (vsrr["state_abbr"] == "US")
        ].copy().sort_values("date")

        if not us_total.empty:
            latest = us_total.iloc[-1]
            prev_year = us_total[us_total["year"] == latest["year"] - 1]
            delta = delta_pct = None
            if not prev_year.empty:
                prev = prev_year.iloc[-1]
                delta = int(latest["death_count"] - prev["death_count"])
                delta_pct = delta / prev["death_count"] * 100

            # Top metrics with context
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Annual Overdose Deaths",
                          f"{int(latest['death_count']):,}",
                          f"{delta:+,} ({delta_pct:+.1f}%)" if delta else None,
                          delta_color="inverse")
            with c2:
                daily = int(latest["death_count"]) // 365
                st.metric("Deaths Per Day", f"~{daily}")
            with c3:
                st.metric("Latest Data", f"{latest['month']} {int(latest['year'])}")
            with c4:
                st.metric("States Tracked", f"{vsrr['state_abbr'].nunique() - 1}")

            if delta and delta < 0:
                callout(
                    f"🟢 <strong>Good news:</strong> Overdose deaths have decreased by {abs(delta):,} "
                    f"({abs(delta_pct):.1f}%) compared to last year. This is likely due to increased "
                    f"naloxone availability and treatment programs — but {int(latest['death_count']):,} "
                    f"deaths per year is still a crisis.",
                    "help-box"
                )
            elif delta and delta > 0:
                callout(
                    f"🔴 <strong>Warning:</strong> Overdose deaths have increased by {delta:,} "
                    f"({delta_pct:+.1f}%) compared to last year.",
                    "warning-box"
                )

            # Main trend chart
            st.subheader("📈 How Has This Changed Over Time?")
            fig = px.area(
                us_total, x="date", y="death_count",
                labels={"date": "", "death_count": "Deaths (12-Month Total)"},
                template="plotly_dark",
            )
            fig.update_traces(
                line=dict(width=3, color="#ff6b6b"),
                fillcolor="rgba(255,107,107,0.1)",
            )
            fig.update_layout(
                height=400, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_tickformat=",", hovermode="x unified",
            )
            # Add peak annotation
            peak = us_total.loc[us_total["death_count"].idxmax()]
            fig.add_annotation(
                x=peak["date"], y=peak["death_count"],
                text=f"Peak: {int(peak['death_count']):,}",
                showarrow=True, arrowhead=2,
                font=dict(size=12, color="#ff6b6b"),
            )
            st.plotly_chart(fig, use_container_width=True)
            callout(
                "📖 <strong>Reading this chart:</strong> Each point shows the total number of "
                "overdose deaths in the previous 12 months. If the line goes up, more people "
                "are dying. The line peaked around 2022 and has started to come down — a sign "
                "that public health interventions may be working."
            )

        # Choropleth
        st.markdown("---")
        st.subheader("🗺️ Which States Are Hit Hardest?")
        latest_year = int(vsrr["year"].max())
        state_data = vsrr[
            (vsrr["indicator"] == "Number of Drug Overdose Deaths") &
            (vsrr["period"] == "12 month-ending") &
            (vsrr["month"] == "December") &
            (vsrr["year"] == latest_year) &
            (~vsrr["state_abbr"].isin(["US", "YC"]))
        ].copy()

        if not state_data.empty:
            # Add per-capita rate
            state_data["population"] = state_data["state_abbr"].map(STATE_POPULATIONS)
            state_data["rate_per_100k"] = (
                state_data["death_count"] / state_data["population"] * 100000
            ).round(1)

            view = st.radio("Show:", ["Per-Capita Rate (fairer comparison)", "Total Deaths"],
                            horizontal=True)

            color_col = "rate_per_100k" if "Per-Capita" in view else "death_count"
            color_label = "Deaths per 100K" if "Per-Capita" in view else "Total Deaths"

            fig_map = px.choropleth(
                state_data.dropna(subset=[color_col]),
                locations="state_abbr", locationmode="USA-states",
                color=color_col, scope="usa",
                color_continuous_scale="YlOrRd",
                labels={color_col: color_label, "state_abbr": "State"},
                hover_name="state_name",
                hover_data={"death_count": ":,.0f", "rate_per_100k": ":.1f"},
                template="plotly_dark",
            )
            fig_map.update_layout(
                height=500, margin=dict(l=0, r=0, t=0, b=0),
                geo=dict(bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            callout(
                "📖 <strong>Why per-capita matters:</strong> Big states like California and Texas "
                "have more total deaths simply because they have more people. The 'per-capita rate' "
                "divides by population so you can fairly compare — smaller states like West Virginia "
                "and Tennessee often have the <em>highest rates</em> even though their total counts are lower."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Overdose Trends
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "📈 Overdose Trends":
    st.title("📈 How Are Overdose Deaths Changing?")
    callout(
        "📖 <strong>What is this?</strong> Compare overdose death trends across states. "
        "Select the states you care about and see whether things are getting better or worse."
    )

    if not vsrr.empty:
        states = sorted(vsrr["state_name"].dropna().unique())
        default = [s for s in ["United States", "Ohio", "California", "Florida", "West Virginia"]
                   if s in states]
        selected_states = st.multiselect("Compare States:", states, default=default or states[:5])

        trend_data = vsrr[
            (vsrr["state_name"].isin(selected_states)) &
            (vsrr["indicator"] == "Number of Drug Overdose Deaths") &
            (vsrr["period"] == "12 month-ending")
        ].sort_values("date")

        if not trend_data.empty:
            fig = px.line(
                trend_data, x="date", y="death_count", color="state_name",
                labels={"date": "", "death_count": "Deaths (12-Month)", "state_name": "State"},
                template="plotly_dark",
            )
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(
                height=500, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_tickformat=",", hovermode="x unified",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, use_container_width=True)

            # Year-over-year change table
            st.subheader("📊 Year-over-Year: Getting Better or Worse?")
            callout(
                "📖 This table shows whether each state had <strong>more or fewer</strong> deaths "
                "compared to the previous year. 🟢 Green = improvement, 🔴 Red = worsening."
            )

            latest_year = int(trend_data["year"].max())
            pivot = trend_data[
                (trend_data["month"] == "December") &
                (trend_data["year"].isin([latest_year, latest_year - 1]))
            ].pivot_table(values="death_count", index="state_name", columns="year")

            if len(pivot.columns) == 2:
                prev, curr = pivot.columns
                pivot["Change"] = pivot[curr] - pivot[prev]
                pivot["Trend"] = pivot["Change"].apply(
                    lambda x: "🟢 Improving" if x < 0 else "🔴 Worsening"
                )
                pivot = pivot.sort_values("Change")
                pivot.columns = [str(int(c)) if isinstance(c, (int, float)) else c for c in pivot.columns]
                st.dataframe(
                    pivot.style.format("{:,.0f}", subset=[str(int(prev)), str(int(curr)), "Change"]),
                    use_container_width=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: State Comparison
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ State Comparison":
    st.title("🗺️ Compare States Side-by-Side")
    callout(
        "📖 <strong>What is this?</strong> See which states have the most overdose deaths "
        "and which drugs are causing the most harm in each state."
    )

    if not vsrr.empty:
        year_options = sorted(vsrr["year"].dropna().unique().astype(int), reverse=True)
        selected_year = st.selectbox("Select Year:", year_options)

        state_annual = vsrr[
            (vsrr["indicator"] == "Number of Drug Overdose Deaths") &
            (vsrr["period"] == "12 month-ending") &
            (vsrr["month"] == "December") &
            (vsrr["year"] == selected_year) &
            (~vsrr["state_abbr"].isin(["US", "YC"]))
        ].copy().sort_values("death_count", ascending=False)

        if not state_annual.empty:
            # Per capita
            state_annual["population"] = state_annual["state_abbr"].map(STATE_POPULATIONS)
            state_annual["rate_per_100k"] = (
                state_annual["death_count"] / state_annual["population"] * 100000
            ).round(1)

            col1, col2 = st.columns([2, 1])

            with col1:
                st.subheader(f"Death Rate per 100,000 ({selected_year})")
                fig = px.choropleth(
                    state_annual.dropna(subset=["rate_per_100k"]),
                    locations="state_abbr", locationmode="USA-states",
                    color="rate_per_100k", scope="usa",
                    color_continuous_scale="Turbo",
                    hover_name="state_name",
                    hover_data={"death_count": ":,.0f", "rate_per_100k": ":.1f"},
                    labels={"rate_per_100k": "Per 100K"},
                    template="plotly_dark",
                )
                fig.update_layout(
                    height=450, margin=dict(l=0, r=0, t=0, b=0),
                    geo=dict(bgcolor="rgba(0,0,0,0)"),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("Highest Rates")
                top = state_annual.dropna(subset=["rate_per_100k"]).nlargest(15, "rate_per_100k")
                fig_bar = px.bar(
                    top, x="rate_per_100k", y="state_name", orientation="h",
                    labels={"rate_per_100k": "Per 100K", "state_name": ""},
                    template="plotly_dark",
                    color="rate_per_100k", color_continuous_scale="YlOrRd",
                )
                fig_bar.update_layout(
                    height=450, showlegend=False,
                    margin=dict(l=0, r=20, t=0, b=40),
                    coloraxis_showscale=False, yaxis_autorange="reversed",
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            callout(
                "📖 <strong>What stands out?</strong> States like West Virginia, Tennessee, and "
                "Louisiana often have the highest <em>per-capita</em> rates — meaning a larger share "
                "of their population is affected, even if they have fewer total deaths than California or Texas."
            )

            # Drug type by state
            st.markdown("---")
            st.subheader(f"💊 What Drugs Are Killing People in Each State? ({selected_year})")

            drug_indicators = [
                "Heroin (T40.1)",
                "Natural & semi-synthetic opioids (T40.2)",
                "Synthetic opioids, excl. methadone (T40.4)",
                "Cocaine (T40.5)",
                "Psychostimulants with abuse potential (T43.6)",
            ]

            drug_state = vsrr[
                (vsrr["indicator"].isin(drug_indicators)) &
                (vsrr["period"] == "12 month-ending") &
                (vsrr["month"] == "December") &
                (vsrr["year"] == selected_year) &
                (~vsrr["state_abbr"].isin(["US", "YC"]))
            ].copy()
            drug_state["drug_name"] = drug_state["indicator"].map(human_drug_name)

            if not drug_state.empty:
                top_states = state_annual.head(10)["state_name"].tolist()
                drug_top = drug_state[drug_state["state_name"].isin(top_states)]

                fig_heat = go.Figure(data=go.Heatmap(
                    z=drug_top.pivot_table(values="death_count", index="state_name",
                                            columns="drug_name", aggfunc="sum").values,
                    x=drug_top.pivot_table(values="death_count", index="state_name",
                                            columns="drug_name", aggfunc="sum").columns.tolist(),
                    y=drug_top.pivot_table(values="death_count", index="state_name",
                                            columns="drug_name", aggfunc="sum").index.tolist(),
                    colorscale="Inferno",
                    hovertemplate="State: %{y}<br>Drug: %{x}<br>Deaths: %{z:,}<extra></extra>",
                ))
                fig_heat.update_layout(
                    template="plotly_dark", height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    xaxis_tickangle=-20,
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                callout(
                    "📖 <strong>Reading the heatmap:</strong> Brighter colors = more deaths. "
                    "Notice how <strong>Fentanyl & Synthetic Opioids</strong> dominates nearly "
                    "every state — it's the #1 killer in the overdose crisis."
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Drug Breakdown
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💊 Drug Breakdown":
    st.title("💊 Which Drugs Are Causing the Most Deaths?")
    callout(
        "📖 <strong>What is this?</strong> Not all drug overdoses are the same. This page breaks "
        "down deaths by specific drug type. The biggest story of the last decade is the explosive "
        "rise of <strong>fentanyl</strong> — a synthetic opioid 50-100x stronger than morphine."
    )

    if not vsrr.empty:
        drug_indicators = [
            "Heroin (T40.1)",
            "Natural & semi-synthetic opioids (T40.2)",
            "Methadone (T40.3)",
            "Synthetic opioids, excl. methadone (T40.4)",
            "Cocaine (T40.5)",
            "Psychostimulants with abuse potential (T43.6)",
        ]

        us_drugs = vsrr[
            (vsrr["state_abbr"] == "US") &
            (vsrr["indicator"].isin(drug_indicators)) &
            (vsrr["period"] == "12 month-ending")
        ].sort_values("date")

        if not us_drugs.empty:
            st.subheader("📈 Death Trends by Drug Type (2015–Present)")
            fig = px.line(
                us_drugs, x="date", y="death_count", color="drug_name",
                labels={"date": "", "death_count": "Deaths (12-Month)", "drug_name": "Drug"},
                template="plotly_dark",
            )
            fig.update_traces(line=dict(width=2.5))
            fig.update_layout(
                height=500, margin=dict(l=40, r=20, t=30, b=40),
                yaxis_tickformat=",", hovermode="x unified",
                legend=dict(orientation="h", y=-0.3, font=dict(size=10)),
            )
            st.plotly_chart(fig, use_container_width=True)
            callout(
                "📖 <strong>The fentanyl tsunami:</strong> Notice how the red line (Fentanyl) "
                "skyrocketed starting around 2015. Illicitly-manufactured fentanyl is now mixed "
                "into heroin, cocaine, and counterfeit pills — people often don't know they're "
                "taking it."
            )

            # Key insights
            st.markdown("---")
            col1, col2 = st.columns(2)

            latest_year = int(us_drugs["year"].max())
            latest_drugs = us_drugs[
                (us_drugs["month"] == "December") & (us_drugs["year"] == latest_year)
            ].sort_values("death_count", ascending=False)

            with col1:
                st.subheader(f"💀 Death Toll by Drug ({latest_year})")
                if not latest_drugs.empty:
                    for _, row in latest_drugs.iterrows():
                        name = row["drug_name"]
                        count = int(row["death_count"])
                        # Year-over-year
                        prev = us_drugs[
                            (us_drugs["indicator"] == row["indicator"]) &
                            (us_drugs["month"] == "December") &
                            (us_drugs["year"] == latest_year - 1)
                        ]
                        if not prev.empty:
                            change = (count - prev.iloc[0]["death_count"]) / prev.iloc[0]["death_count"] * 100
                            trend = "🟢" if change < 0 else "🔴"
                            st.markdown(f"**{name}**\n\n{count:,} deaths ({trend} {change:+.1f}% vs last year)")
                        else:
                            st.markdown(f"**{name}**: {count:,} deaths")
                        st.markdown("")

            with col2:
                st.subheader("⚠️ Fentanyl vs Heroin")
                synthetic = us_drugs[us_drugs["indicator"] == "Synthetic opioids, excl. methadone (T40.4)"]
                heroin = us_drugs[us_drugs["indicator"] == "Heroin (T40.1)"]

                fig_compare = go.Figure()
                if not synthetic.empty:
                    fig_compare.add_trace(go.Scatter(
                        x=synthetic["date"], y=synthetic["death_count"],
                        name="Fentanyl", line=dict(width=3, color="#ff4444"),
                        fill="tozeroy", fillcolor="rgba(255,68,68,0.15)",
                    ))
                if not heroin.empty:
                    fig_compare.add_trace(go.Scatter(
                        x=heroin["date"], y=heroin["death_count"],
                        name="Heroin", line=dict(width=3, color="#4488ff"),
                        fill="tozeroy", fillcolor="rgba(68,136,255,0.15)",
                    ))
                fig_compare.update_layout(
                    template="plotly_dark", height=350,
                    margin=dict(l=40, r=20, t=10, b=40),
                    yaxis_tickformat=",", yaxis_title="Deaths",
                    hovermode="x unified",
                    legend=dict(orientation="h", y=1.05),
                )
                st.plotly_chart(fig_compare, use_container_width=True)
                callout(
                    "📖 In 2015, heroin killed more people than fentanyl. By 2018, fentanyl "
                    "had completely <strong>overtaken heroin</strong> — and the gap keeps widening. "
                    "Fentanyl is now mixed into the drug supply, making every substance more dangerous.",
                    "warning-box"
                )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Vulnerability Map
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🏘️ Vulnerability Map":
    st.title("🏘️ Which Communities Are Most Vulnerable?")
    callout(
        "📖 <strong>What is this?</strong> Overdose deaths don't happen equally everywhere. "
        "Communities with high poverty, unemployment, and lack of health insurance are more "
        "vulnerable. This page shows socioeconomic risk factors for every US zip code, "
        "using real Census data."
    )

    if not census.empty:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Zip Codes Analyzed", f"{len(census):,}")
        with c2:
            st.metric("Avg Poverty Rate", f"{census['poverty_rate'].mean():.1f}%")
        with c3:
            median_inc = census["median_household_income"].median()
            st.metric("Median Income (US)", f"${median_inc:,.0f}")
        with c4:
            high_vuln = (census["vulnerability_score"] > 0.3).sum()
            pct = high_vuln / len(census) * 100
            st.metric("High-Risk Zip Codes", f"{high_vuln:,} ({pct:.1f}%)")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Vulnerability Score Distribution")
            fig_hist = px.histogram(
                census, x="vulnerability_score", nbins=50,
                labels={"vulnerability_score": "Vulnerability Score (0 = low risk, 1 = high risk)"},
                template="plotly_dark", color_discrete_sequence=["#ff6b6b"],
            )
            fig_hist.add_vline(x=0.3, line_dash="dash", line_color="yellow",
                               annotation_text="⚠️ High Risk Line")
            fig_hist.update_layout(height=350, margin=dict(l=40, r=20, t=10, b=40))
            st.plotly_chart(fig_hist, use_container_width=True)
            callout(
                "📖 <strong>Vulnerability score</strong> combines poverty rate (40%), "
                "unemployment rate (30%), and uninsured rate (30%) into a single number. "
                "Scores above 0.3 indicate communities with multiple compounding risk factors."
            )

        with col2:
            st.subheader("💰 Poverty vs Unemployment")
            sample = census.dropna(subset=["poverty_rate", "unemployment_rate"]).sample(
                min(5000, len(census)), random_state=42
            )
            fig_sc = px.scatter(
                sample, x="poverty_rate", y="unemployment_rate",
                color="vulnerability_score", color_continuous_scale="YlOrRd",
                labels={"poverty_rate": "Poverty Rate (%)", "unemployment_rate": "Unemployment (%)",
                        "vulnerability_score": "Risk"},
                template="plotly_dark", opacity=0.4,
            )
            fig_sc.update_layout(height=350, margin=dict(l=40, r=20, t=10, b=40))
            st.plotly_chart(fig_sc, use_container_width=True)
            callout(
                "📖 Communities in the <strong>upper-right</strong> have both high poverty AND "
                "high unemployment — these are the most vulnerable to overdose crises."
            )

        # Most vulnerable table
        st.markdown("---")
        st.subheader("🚨 50 Most Vulnerable Zip Codes in America")
        top_vuln = census.nlargest(50, "vulnerability_score")[[
            "zip_code", "NAME", "vulnerability_score",
            "poverty_rate", "unemployment_rate", "uninsured_pct",
            "median_household_income", "total_population",
        ]].copy()
        top_vuln.columns = [
            "Zip Code", "Location", "Risk Score",
            "Poverty %", "Unemployment %", "Uninsured %",
            "Median Income", "Population",
        ]
        st.dataframe(
            top_vuln.style
                .format({"Risk Score": "{:.3f}", "Poverty %": "{:.1f}%",
                         "Unemployment %": "{:.1f}%", "Uninsured %": "{:.1f}%",
                         "Median Income": "${:,.0f}", "Population": "{:,.0f}"})
                .background_gradient(cmap="YlOrRd", subset=["Risk Score"]),
            use_container_width=True, height=500,
        )
    else:
        st.warning("Census data not loaded. Run `python run.py` first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Look Up Your Zip Code
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔍 Look Up Your Zip Code":
    st.title("🔍 Look Up Your Community")
    callout(
        "📖 <strong>How vulnerable is YOUR community?</strong> Enter your zip code below "
        "to see how your area compares to the rest of the country on key risk factors."
    )

    if not census.empty:
        zip_input = st.text_input("Enter your zip code:", placeholder="e.g. 60601")

        if zip_input:
            match = census[census["zip_code"] == zip_input.strip()]

            if not match.empty:
                row = match.iloc[0]
                st.markdown(f"### 📍 {row.get('NAME', zip_input)}")

                # Risk score gauge
                score = row["vulnerability_score"]
                if score > 0.3:
                    level, color, emoji = "High Risk", "#ff4444", "🔴"
                elif score > 0.15:
                    level, color, emoji = "Moderate Risk", "#ffaa00", "🟡"
                else:
                    level, color, emoji = "Lower Risk", "#44ff44", "🟢"

                st.markdown(f"## {emoji} Vulnerability: **{level}** (score: {score:.3f})")

                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    pov = row["poverty_rate"]
                    nat_pov = census["poverty_rate"].median()
                    st.metric("Poverty Rate", f"{pov:.1f}%",
                              f"{'Above' if pov > nat_pov else 'Below'} national median ({nat_pov:.1f}%)")
                with c2:
                    unemp = row["unemployment_rate"]
                    nat_unemp = census["unemployment_rate"].median()
                    st.metric("Unemployment", f"{unemp:.1f}%",
                              f"{'Above' if unemp > nat_unemp else 'Below'} national median ({nat_unemp:.1f}%)")
                with c3:
                    unins = row["uninsured_pct"]
                    nat_unins = census["uninsured_pct"].median()
                    st.metric("Uninsured Rate", f"{unins:.1f}%",
                              f"{'Above' if unins > nat_unins else 'Below'} national median ({nat_unins:.1f}%)")
                with c4:
                    income = row["median_household_income"]
                    nat_inc = census["median_household_income"].median()
                    st.metric("Median Income", f"${income:,.0f}",
                              f"{'Above' if income > nat_inc else 'Below'} national median (${nat_inc:,.0f})")

                # Where this zip ranks
                rank = (census["vulnerability_score"] <= score).sum()
                percentile = rank / len(census) * 100
                st.markdown(f"---")
                st.markdown(f"📊 **Your zip code is more vulnerable than {percentile:.0f}% of all US zip codes.**")

                # Show distribution with marker
                fig = px.histogram(
                    census, x="vulnerability_score", nbins=50,
                    labels={"vulnerability_score": "Vulnerability Score"},
                    template="plotly_dark", color_discrete_sequence=["#444466"],
                )
                fig.add_vline(x=score, line_color=color, line_width=3,
                              annotation_text=f"Your Zip: {score:.3f}")
                fig.update_layout(height=250, margin=dict(l=40, r=20, t=10, b=40))
                st.plotly_chart(fig, use_container_width=True)

                # Resources
                if score > 0.2:
                    st.markdown("---")
                    st.subheader("🆘 Resources for Your Community")
                    st.markdown("""
                    - 📞 **SAMHSA National Helpline**: [1-800-662-4357](tel:18006624357) (free, 24/7)
                    - 🏥 **Find Treatment Near You**: [findtreatment.gov](https://findtreatment.gov)
                    - 💊 **Get Free Naloxone**: [naloxoneforall.org](https://www.naloxoneforall.org)
                    - 📋 **Report Suspicious Activity**: [DEA Tip Line](https://www.deadiversion.usdoj.gov)
                    """)
            else:
                st.warning(f"Zip code '{zip_input}' not found. Try a 5-digit US zip code (e.g. 60601, 10001).")
    else:
        st.warning("Census data not loaded. Run `python run.py` first.")
