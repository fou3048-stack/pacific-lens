"""Pacific Lens | Run: streamlit run app.py
Dependencies: streamlit>=1.49,<2 pandas>=2,<3 numpy>=1.26,<3
              plotly>=5.24,<7 yfinance>=0.2.54,<1
"""
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Pacific Lens", page_icon="🌏", layout="wide")

# Edit these to personalize your publication.
AUTHOR = "Frederick Ou"
MARKETS = {
    "United States": ("^GSPC", "S&P 500", "USD"),
    "Japan": ("^N225", "Nikkei 225", "JPY"),
    "Hong Kong": ("^HSI", "Hang Seng", "HKD"),
    "Mainland China": ("000001.SS", "Shanghai Composite", "CNY"),
    "India": ("^BSESN", "BSE Sensex", "INR"),
    "South Korea": ("^KS11", "KOSPI", "KRW"),
    "Taiwan": ("^TWII", "Taiwan Weighted", "TWD"),
    "Singapore": ("^STI", "Straits Times", "SGD"),
}
COLORS = ["#0369a1", "#0f766e", "#b45309", "#be123c",
          "#6d28d9", "#c2410c", "#1d4ed8", "#a21caf"]
st.markdown("""
<style>
.stApp {background:#ffffff; color:#24364b;}
[data-testid="stHeader"] {background:#ffffff;}
[data-testid="stSidebar"] {background:#f3f6fa; border-right:1px solid #dce4ed;}
html, body, [data-testid="stAppViewContainer"],
[data-testid="stMarkdownContainer"], input, textarea, button {
 font-family: "Inter", "Segoe UI", Arial, sans-serif;
}
h1,h2,h3 {color:#16324f !important; letter-spacing:-.025em;}
[data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stMetricLabel"],
[data-testid="stMetricValue"], [data-testid="stCaptionContainer"] {color:#24364b;}
.hero {padding:2.2rem; border:1px solid #d8e5ed; border-radius:20px;
 background:linear-gradient(120deg,#f0f8fa,#f8fafc); margin-bottom:1.5rem;}
.eyebrow {color:#0f766e; letter-spacing:2px; font-size:12px; font-weight:700;}
.hero h1 {font-size:clamp(32px,5vw,54px); margin:10px 0; font-weight:750;}
.hero p {color:#42566c; max-width:750px; line-height:1.75;}
[data-testid="stMetric"] {background:#f8fafc; padding:18px;
 border:1px solid #dce4ed; border-radius:14px;}
</style>
""", unsafe_allow_html=True)


def chart(fig):
    fig.update_layout(template="plotly_white", paper_bgcolor="#ffffff",
                      plot_bgcolor="#ffffff", font=dict(color="#24364b", family="Segoe UI, Arial", size=14),
                      legend_title_text="", margin=dict(l=15, r=15, t=55, b=20))
    fig.update_xaxes(gridcolor="#e8edf3")
    fig.update_yaxes(gridcolor="#e8edf3")
    st.plotly_chart(fig, width="stretch", theme=None)


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(names, days):
    """Yahoo daily Close only. Never substitute generated prices."""
    import yfinance as yf
    cutoff = datetime.now(timezone.utc).date()
    start = cutoff - timedelta(days=days)
    fetched = datetime.now(timezone.utc).isoformat(timespec="seconds")
    data, errors = {}, []
    for name in names:
        try:
            raw = yf.Ticker(MARKETS[name][0]).history(
                start=str(start), end=str(cutoff), interval="1d",
                auto_adjust=False, actions=False, repair=False, timeout=15)
            if raw is None or raw.empty:
                raise ValueError("No prices returned")
            close = pd.to_numeric(raw["Close"], errors="coerce")
            close = close.replace([np.inf, -np.inf], np.nan).dropna()
            close = close[close > 0]
            # Preserve each exchange's session date, without converting to UTC.
            close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
            close = close[~close.index.duplicated(keep="last")].sort_index()
            close = close[close.index < pd.Timestamp(cutoff)]
            if len(close) < 3:
                raise ValueError("Insufficient prices")
            data[name] = close
        except Exception as exc:
            errors.append(f"{name} ({MARKETS[name][0]}): Yahoo data unavailable ({type(exc).__name__}).")
    return pd.DataFrame(data).sort_index(), errors, fetched


def compare_prices(prices):
    """Use shared endpoints; preserve intervening local sessions for risk."""
    common = prices.dropna(how="any")
    if len(common) < 3:
        return None
    first, last = common.index[0], common.index[-1]
    window = prices.loc[first:last]
    indexed = window.div(window.loc[first]).mul(100)
    performance = ((window.loc[last] / window.loc[first] - 1) * 100).sort_values(ascending=False)
    daily = pd.DataFrame({n: window[n].dropna().pct_change(fill_method=None)
                          for n in window.columns})
    drawdown = pd.DataFrame({n: (window[n].dropna() / window[n].dropna().cummax() - 1) * 100
                             for n in window.columns})
    volatility = daily.std() * np.sqrt(252) * 100
    # Only correlate returns spanning identical start AND end session dates.
    matched = daily.dropna(how="any")
    previous = pd.DataFrame({n: pd.Series(window[n].dropna().index,
                                          index=window[n].dropna().index).shift(1)
                             for n in window.columns})
    same_interval = previous.loc[matched.index].nunique(axis=1).eq(1)
    matched = matched.loc[same_interval]
    return window, indexed, performance, drawdown, volatility, matched


st.sidebar.title("🌏 Pacific Lens")
st.sidebar.caption("Asia × United States")
page = st.sidebar.radio("Explore", ["Market dashboard", "Scenario lab",
                                     "Research journal", "About & sources"])
st.sidebar.caption("Source: Yahoo Finance via yfinance")
st.sidebar.caption("Historical daily closes • refresh cache: 15 minutes")
st.markdown("""
<div class="hero"><div class="eyebrow">A STUDENT PERSPECTIVE ON GLOBAL MARKETS</div>
<h1>Across markets. Beyond headlines.</h1>
<p>Explore Asia alongside the United States. Compare the evidence,
test assumptions, and build a transparent record of your thinking.</p></div>
""", unsafe_allow_html=True)

if page == "Market dashboard":
    st.subheader("Asia–U.S. market explorer")
    choices = st.multiselect("Asian markets", list(MARKETS)[1:],
                             default=["Japan", "Mainland China", "India"])
    period = st.select_slider("Lookback", ["3 months", "6 months", "1 year", "3 years"],
                               value="1 year")
    days = {"3 months": 92, "6 months": 183, "1 year": 365, "3 years": 1095}[period]
    if st.button("Refresh market data"):
        load_prices.clear()
    with st.spinner("Preparing comparison…"):
        prices, errors, fetched = load_prices(tuple(["United States"] + choices), days)
    for error in errors:
        st.warning(error)
    st.caption(f"Retrieved: {fetched}. Cached for up to 15 minutes.")
    if prices.empty:
        st.error("Yahoo returned no usable data. Check your connection or try Refresh later. No substitute data is displayed.")
        st.stop()
    comparison = compare_prices(prices)
    if comparison is None:
        st.warning("Not enough shared dates. Select fewer markets or a longer period.")
        st.stop()
    aligned, indexed, performance, drawdown, volatility, returns = comparison
    c1, c2, c3 = st.columns(3)
    c1.metric("Highest selected return", performance.index[0], f"{performance.iloc[0]:+.2f}%")
    us = performance.get("United States")
    c2.metric("U.S. price return", "Unavailable" if us is None else f"{us:+.2f}%")
    c3.metric("Markets available", len(aligned.columns))
    st.caption(f"Comparison window: {aligned.index[0]:%Y-%m-%d} to {aligned.index[-1]:%Y-%m-%d}.")
    st.caption("Today's UTC calendar date is excluded to avoid partial daily bars. "
               "Latest available closes appear below; these can differ from Yahoo's live quote.")
    quotes = []
    for name in prices.columns:
        series = prices[name].dropna()
        quotes.append({"Market": name, "Yahoo symbol": MARKETS[name][0],
                       "Index": MARKETS[name][1], "Currency": MARKETS[name][2],
                       "Last close": float(series.iloc[-1]),
                       "Session change (%)": float((series.iloc[-1] / series.iloc[-2] - 1) * 100),
                       "Session date": str(series.index[-1].date()),
                       "Yahoo history": f"https://finance.yahoo.com/quote/{MARKETS[name][0]}/history/"})
    st.dataframe(pd.DataFrame(quotes), hide_index=True, width="stretch",
                 column_config={"Last close": st.column_config.NumberColumn(format="%.2f"),
                                "Session change (%)": st.column_config.NumberColumn(format="%.2f"),
                                "Yahoo history": st.column_config.LinkColumn("Verify on Yahoo")})
    st.info("Local-currency index price returns: not USD investment returns and not dividend-inclusive. "
            "Exchange hours differ; matching calendar dates does not mean simultaneous prices.")
    tab1, tab2, tab3 = st.tabs(["Performance", "Risk & correlation", "Data table"])
    with tab1:
        chart(px.line(indexed, labels={"value": "Index rebased to 100", "index": "Date"},
                      color_discrete_sequence=COLORS, title="Same starting point. Different paths."))
        chart(px.bar(x=performance.index, y=performance.values,
                     labels={"x": "Market", "y": "Price return (%)"},
                     color=performance.index, color_discrete_sequence=COLORS))
    with tab2:
        chart(px.line(drawdown, labels={"value": "Drawdown (%)", "index": "Date"},
                      color_discrete_sequence=COLORS, title="Decline from the running peak"))
        if len(returns) >= 3:
            chart(px.imshow(returns.corr(), zmin=-1, zmax=1, text_auto=".2f",
                            color_continuous_scale="RdBu_r", title="Return correlation"))
        else:
            st.info("Too few matching return intervals for a correlation matrix.")
        st.caption(f"Correlation uses {len(returns)} returns with matching start and end dates. "
                   "Market closing times still differ. Volatility uses each market's own sessions "
                   "with √252 annualization. Drawdowns retain all local sessions in the comparison window.")
    with tab3:
        summary = pd.DataFrame({"Price return (%)": performance,
                                "Approx. annual volatility (%)": volatility,
                                "Max drawdown (%)": drawdown.min()})
        st.dataframe(summary.round(2), width="stretch")
        metadata = pd.DataFrame([{"Market": n, "Index": MARKETS[n][1],
                                  "Currency": MARKETS[n][2],
                                  "Latest available date": prices[n].last_valid_index().date()}
                                 for n in prices.columns])
        st.dataframe(metadata, hide_index=True, width="stretch")
        st.dataframe(aligned.tail(15), width="stretch")
        export = aligned.rename_axis("Date").reset_index()
        export["Data source"] = "Yahoo Finance via yfinance (daily Close, unadjusted)"
        st.download_button("Download comparison prices", export.to_csv(index=False),
                           "pacific_lens_prices.csv", "text/csv")

elif page == "Scenario lab":
    st.subheader("What would your assumptions imply?")
    st.info("This is a hypothetical calculator, not a market forecast or a probability model.")
    country = st.selectbox("Asian market for this scenario", list(MARKETS)[1:])
    left, right = st.columns(2)
    with left:
        asia = st.slider("Assumed Asian local price return (%)", -60, 60, 8)
        fx = st.slider("Assumed local currency change versus USD (%)", -40, 40, 0,
                       help="Positive means the Asian currency appreciates against USD.")
    with right:
        us_return = st.slider("Assumed U.S. price return (%)", -60, 60, 6)
        months = st.slider("Horizon (months)", 1, 36, 12)
    usd_return = ((1 + asia / 100) * (1 + fx / 100) - 1) * 100
    a, b, c = st.columns(3)
    a.metric(f"{country} in USD", f"{usd_return:+.2f}%")
    b.metric("U.S. scenario", f"{us_return:+.2f}%")
    c.metric("Asia minus U.S.", f"{usd_return - us_return:+.2f} percentage points")
    t = np.arange(months + 1)
    paths = pd.DataFrame({f"{country} / USD": 100 * (1 + usd_return / 100) ** (t / months),
                          "United States": 100 * (1 + us_return / 100) ** (t / months)}, index=t)
    chart(px.line(paths, labels={"index": "Month", "value": "Hypothetical value (start = 100)"},
                  color_discrete_sequence=COLORS))
    st.caption("Lines interpolate smoothly to your assumed endpoints; actual markets fluctuate. "
               "Taxes, dividends, trading costs, and currency hedging are excluded.")
    with st.expander("How the currency calculation works"):
        st.latex(r"R_{USD}=(1+R_{local})(1+R_{FX})-1")
        st.write("Returns are decimals in this formula. Currency appreciation increases a "
                 "USD investor's return; depreciation reduces it.")

elif page == "Research journal":
    st.subheader("Your ideas. Your evidence. Your track record.")
    st.caption("A private session workspace. Entries are not publicly published or permanently stored. "
               "Download your JSON backup before closing; import it next time.")
    if "journal" not in st.session_state:
        st.session_state.journal = []
    uploaded = st.file_uploader("Restore an exported journal", type="json")
    if uploaded is not None and st.button("Import journal (replaces this session)"):
        try:
            payload = json.load(uploaded)
            fields = {"created", "market", "question", "thesis", "evidence", "counterargument",
                      "deadline", "confidence", "outcome", "reflection"}
            if not isinstance(payload, list) or len(payload) > 500:
                raise ValueError("Expected a list of at most 500 entries.")
            for row in payload:
                if not isinstance(row, dict) or set(row) != fields:
                    raise ValueError("Use the exact exported journal format.")
                if not all(isinstance(row[k], str) for k in fields - {"confidence"}):
                    raise ValueError("Invalid text fields.")
                if type(row["confidence"]) is not int or not 0 <= row["confidence"] <= 100:
                    raise ValueError("Invalid confidence.")
                date.fromisoformat(row["deadline"])
            st.session_state.journal = payload
            st.success("Journal restored.")
        except (ValueError, TypeError):
            st.error("Invalid journal. Upload an unmodified JSON backup from this app.")
    with st.form("new_entry", clear_on_submit=True):
        market = st.selectbox("Focus", list(MARKETS))
        question = st.text_input("Research question / measurable forecast")
        thesis = st.text_area("My argument: what I expect and why")
        evidence = st.text_area("Evidence: source URLs, publication dates, and key observations")
        counter = st.text_area("What could prove me wrong?")
        deadline = st.date_input("Review date", date.today() + timedelta(days=90),
                                 min_value=date.today())
        confidence = st.slider("Subjective confidence (%)", 0, 100, 60)
        submit = st.form_submit_button("Add research entry")
        if submit:
            if not all(x.strip() for x in [question, thesis, evidence, counter]):
                st.warning("Fill in the question, argument, evidence, and counterargument.")
            else:
                st.session_state.journal.append(dict(
                    created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    market=market, question=question, thesis=thesis, evidence=evidence,
                    counterargument=counter, deadline=str(deadline), confidence=confidence,
                    outcome="Pending", reflection=""))
                st.success("Added to this session. Download a backup below.")
    for i, entry in enumerate(st.session_state.journal):
        with st.expander(f"{entry['market']} · {entry['question']}"):
            for label, key in [("Original argument", "thesis"), ("Evidence", "evidence"),
                               ("Counterargument", "counterargument")]:
                st.write(f"**{label}**")
                st.text(entry[key])
            st.caption(f"Created {entry['created']} | Review {entry['deadline']} | "
                       f"Initial confidence {entry['confidence']}%")
            with st.form(f"review_{i}"):
                options = ["Pending", "Supported", "Mixed", "Not supported"]
                current = entry["outcome"] if entry["outcome"] in options else "Pending"
                outcome = st.selectbox("Assessment", options, index=options.index(current))
                reflection = st.text_area("What happened? Cite evidence and explain what you learned.",
                                           value=entry["reflection"])
                if st.form_submit_button("Save review"):
                    entry.update(outcome=outcome, reflection=reflection)
                    st.success("Review saved for this session.")
    st.download_button("Download journal backup (.json)",
                       json.dumps(st.session_state.journal, indent=2),
                       "pacific_lens_journal.json", "application/json")
    st.caption("These editable personal notes are not an independently verified forecast record.")

else:
    st.subheader(f"Pacific Lens · {AUTHOR}")
    st.write("A student research project exploring the relationship between Asian and U.S. "
             "financial markets through comparisons, explicit assumptions, and reflective writing.")
    st.markdown("""
### Research questions to investigate
- Does faster economic growth necessarily produce higher stock-market returns?
- How can currency movements change an international investor's results?
- Why might Asian and U.S. markets react differently to the same event?

### Methodology
Prices are index closing levels in their local currencies. Comparison charts
use the first and last dates shared by the selected markets. All intervening
local sessions are retained; missing prices are never forward-filled.
Each series starts at 100 on the first shared date. Returns exclude dividends.
The current UTC date is excluded to avoid partial daily bars.
Volatility is based on each index's own session returns and uses 252 sessions/year.
Correlation includes only returns whose start and end session dates match across markets.
These indices differ in construction and coverage; none represents an entire economy.
Drawdown is the percentage decline from the highest observed level in the chosen window.
Prices come only from Yahoo Finance through yfinance. Unavailable markets are reported and omitted.

### Sources for original research
- [World Bank Open Data](https://data.worldbank.org/)
- [IMF Data](https://data.imf.org/)
- [Asian Development Bank](https://data.adb.org/)
- [Federal Reserve economic data](https://fred.stlouisfed.org/)
- [Bank of Japan](https://www.boj.or.jp/en/)
- [Yahoo Finance](https://finance.yahoo.com/)

### Editorial standards
Date each claim and link its evidence. Distinguish facts, interpretations, and
forecasts. Record alternative explanations and correct mistakes visibly.
Disclose assistance with coding or editing while taking ownership of your conclusions.
Yahoo access through yfinance is intended for personal research; review provider
terms and arrange suitable data rights before public redistribution.
""")
st.divider()
st.caption(f"Pacific Lens • {AUTHOR} • Educational research, not investment advice.")
