"""Streamlit dashboard for WTI Crude Intelligence Copilot."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Sequence


from pathlib import Path

from murban_copilot.domain.entities import MarketData, MovingAverages, SpreadData, MarketSignal
from murban_copilot.domain.spread_calculator import SpreadCalculator
from murban_copilot.infrastructure.market_data.yahoo_client import YahooFinanceClient
from murban_copilot.infrastructure.llm import create_llm_client, get_client_type_name
from murban_copilot.infrastructure.llm.llm_client import LlamaClient
from murban_copilot.infrastructure.llm.mock_client import MockLlamaClient
from murban_copilot.infrastructure.config import ConfigLoader
from murban_copilot.infrastructure.health import HealthChecker, HealthStatus
from murban_copilot.infrastructure.logging import setup_logging, get_logger
from murban_copilot.application.fetch_market_data import FetchMarketDataUseCase
from murban_copilot.application.analyze_spread import AnalyzeSpreadUseCase
from murban_copilot.application.generate_signal import GenerateSignalUseCase

logger = get_logger(__name__)


# Page config
st.set_page_config(
    page_title="Brent/WTI (West Texas Intermediate) Intelligence Copilot",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    /* Base app styling */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }

    /* Main content text */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #ffffff !important;
    }
    .stApp p, .stApp span, .stApp label, .stApp div {
        color: #fafafa;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown span {
        color: #fafafa !important;
    }

    /* Subheader styling */
    [data-testid="stSubheader"] {
        color: #ffffff !important;
    }

    /* Caption/footer styling */
    .stCaption, [data-testid="stCaption"], small {
        color: #a0aec0 !important;
    }

    /* Expander styling */
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    .streamlit-expanderHeader,
    .streamlit-expanderHeader p {
        color: #fafafa !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
        color: #fafafa !important;
    }

    /* Divider styling */
    [data-testid="stHorizontalRule"], hr {
        border-color: #2d3748 !important;
    }

    /* Sidebar styling */
    .stSidebar {
        background-color: #1a1f2c;
    }
    .stSidebar .stMarkdown,
    .stSidebar label,
    .stSidebar .stSelectbox label,
    .stSidebar .stSlider label,
    .stSidebar .stCheckbox label,
    .stSidebar .stCheckbox span,
    .stSidebar .stCheckbox p,
    .stSidebar [data-testid="stCheckbox"] label,
    .stSidebar [data-testid="stCheckbox"] span,
    .stSidebar [data-testid="stSidebarContent"] {
        color: #fafafa !important;
    }
    .stSidebar .stSlider [data-testid="stTickBarMin"],
    .stSidebar .stSlider [data-testid="stTickBarMax"] {
        color: #a0aec0 !important;
    }
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #ffffff !important;
    }

    /* Main dashboard metric styling */
    [data-testid="stMetric"] label,
    [data-testid="stMetricLabel"] {
        color: #a0aec0 !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] {
        color: #fafafa !important;
    }
    [data-testid="stMetric"] [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] {
        color: #48bb78 !important;
    }

    /* Custom metric card styling */
    .metric-card {
        background-color: #1a1f2c;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2d3748;
        color: #fafafa;
    }
    .metric-card h3 {
        color: #ffffff !important;
        margin-bottom: 0.5rem;
    }
    .metric-card p {
        color: #fafafa !important;
        margin: 0.25rem 0;
    }

    /* Signal colors */
    .signal-bullish {
        color: #48bb78 !important;
        font-weight: bold;
    }
    .signal-bearish {
        color: #fc8181 !important;
        font-weight: bold;
    }
    .signal-neutral {
        color: #ecc94b !important;
        font-weight: bold;
    }

    /* Disclaimer styling */
    .disclaimer {
        font-size: 0.75rem;
        color: #a0aec0 !important;
        padding: 1rem;
        background-color: #1a1f2c;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }

    /* Spinner text */
    .stSpinner > div > span {
        color: #fafafa !important;
    }

    /* Alert/message styling */
    [data-testid="stAlert"] p {
        color: #fafafa !important;
    }

    /* Download button styling - high contrast for dark mode */
    [data-testid="stDownloadButton"] button {
        background-color: #4299e1 !important;
        color: #ffffff !important;
        border: none !important;
        font-weight: 600 !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background-color: #3182ce !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


def _init_llm_client(
    session_key: str,
    config,
    cache_config,
    verbose: bool,
    fallback=None,
) -> None:
    """Initialize an LLM client in session state.

    Args:
        session_key: Key to store the client in st.session_state
        config: LLMModelConfig for the client (or None)
        cache_config: CacheConfig for caching behavior
        verbose: Whether to enable verbose logging
        fallback: Fallback client to use if config is None
    """
    if session_key in st.session_state:
        return

    if config:
        st.session_state[session_key] = create_llm_client(
            config=config,
            cache_config=cache_config,
            verbose=verbose,
        )
        client_type = get_client_type_name(st.session_state[session_key])
        logger.info(f"Initialized {session_key} [{client_type}]: {config.model_repo}")
    elif fallback:
        st.session_state[session_key] = fallback
        logger.info(f"Using fallback for {session_key}")
    else:
        st.session_state[session_key] = LlamaClient(cache_config=cache_config)
        logger.info(f"Initialized {session_key} with defaults")


def init_services():
    """Initialize services with caching."""
    # Load full application configuration
    if "app_config" not in st.session_state:
        config_loader = ConfigLoader()
        st.session_state.app_config = config_loader.load_app_config()
        config_path = config_loader.get_config_path()
        if config_path:
            logger.info(f"Loaded application config from: {config_path}")
        else:
            logger.info("Using default application configuration")

    config = st.session_state.app_config

    # Initialize market data client from config
    if "market_client" not in st.session_state:
        st.session_state.market_client = YahooFinanceClient.from_config(config.market_data)
        logger.info(f"Initialized market client with tickers: WTI={config.market_data.wti_ticker}, Brent={config.market_data.brent_ticker}")

    # Initialize spread calculator from config
    if "spread_calculator" not in st.session_state:
        st.session_state.spread_calculator = SpreadCalculator.from_config(config.analysis)
        logger.info(f"Initialized spread calculator with MA windows: {config.analysis.short_ma_window}/{config.analysis.long_ma_window}")

    # For backward compatibility, also store llm_config reference
    if "llm_config" not in st.session_state:
        st.session_state.llm_config = config.llm

    # Initialize analysis LLM client using factory
    _init_llm_client(
        session_key="analysis_llm_client",
        config=config.llm.analysis,
        cache_config=config.llm.cache,
        verbose=config.llm.defaults.verbose,
    )

    # Initialize extraction LLM client (falls back to analysis client if not configured)
    _init_llm_client(
        session_key="extraction_llm_client",
        config=config.llm.extraction,
        cache_config=config.llm.cache,
        verbose=config.llm.defaults.verbose,
        fallback=st.session_state.analysis_llm_client,
    )


def create_spread_chart(
    spread_data: Sequence[SpreadData],
    moving_averages: Sequence[MovingAverages],
) -> go.Figure:
    """Create an interactive Plotly chart for spread data."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3],
        subplot_titles=("WTI-Brent Spread", "Spread Value"),
    )

    sorted_spread = sorted(spread_data, key=lambda x: x.date)
    sorted_ma = sorted(moving_averages, key=lambda x: x.date)

    dates = [d.date for d in sorted_spread]
    wti_prices = [d.wti_close for d in sorted_spread]
    brent_prices = [d.brent_close for d in sorted_spread]
    spreads = [d.spread for d in sorted_spread]

    # WTI price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=wti_prices,
            name="WTI",
            line=dict(color="#48bb78", width=2),
            hovertemplate="WTI: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Brent price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=brent_prices,
            name="Brent",
            line=dict(color="#4299e1", width=2),
            hovertemplate="Brent: $%{y:.2f}<extra></extra>",
        ),
        row=1, col=1,
    )

    # Spread line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=spreads,
            name="Spread",
            line=dict(color="#ecc94b", width=2),
            hovertemplate="Spread: $%{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )

    # Moving averages
    ma_5_dates = [d.date for d in sorted_ma if d.ma_5 is not None]
    ma_5_values = [d.ma_5 for d in sorted_ma if d.ma_5 is not None]
    ma_20_dates = [d.date for d in sorted_ma if d.ma_20 is not None]
    ma_20_values = [d.ma_20 for d in sorted_ma if d.ma_20 is not None]

    if ma_5_values:
        fig.add_trace(
            go.Scatter(
                x=ma_5_dates,
                y=ma_5_values,
                name="5-day MA",
                line=dict(color="#fc8181", width=1, dash="dash"),
                hovertemplate="5-day MA: $%{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

    if ma_20_values:
        fig.add_trace(
            go.Scatter(
                x=ma_20_dates,
                y=ma_20_values,
                name="20-day MA",
                line=dict(color="#9f7aea", width=1, dash="dash"),
                hovertemplate="20-day MA: $%{y:.2f}<extra></extra>",
            ),
            row=2, col=1,
        )

    # Zero line for spread
    fig.add_hline(
        y=0, line_dash="dot", line_color="#a0aec0",
        row=2, col=1, opacity=0.5,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#fafafa"),
            bgcolor="rgba(0,0,0,0)",
        ),
        hovermode="x unified",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(gridcolor="#2d3748", showgrid=True, tickfont=dict(color="#a0aec0"), title_font=dict(color="#fafafa"))
    fig.update_yaxes(gridcolor="#2d3748", showgrid=True, tickfont=dict(color="#a0aec0"), title_font=dict(color="#fafafa"))
    fig.update_yaxes(title_text="Price ($/barrel)", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($)", row=2, col=1)

    # Update subplot titles (annotations) color
    for annotation in fig.layout.annotations:
        annotation.font.color = "#fafafa"

    return fig


def display_signal_card(signal: MarketSignal) -> None:
    """Display the market signal card."""
    signal_class = f"signal-{signal.signal}"

    st.markdown(f"""
    <div class="metric-card">
        <h3>AI Market Signal</h3>
        <p class="{signal_class}">
            Signal: {signal.signal.upper()}
        </p>
        <p>Confidence: {signal.confidence:.0%}</p>
        <p>Generated: {signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View Full Analysis"):
        st.markdown(signal.analysis)

    st.markdown(f"""
    <div class="disclaimer">
        {signal.disclaimer}
    </div>
    """, unsafe_allow_html=True)


def _render_data_tab(
    data: Sequence,
    display_columns: list[tuple[str, callable]],
    csv_columns: list[tuple[str, callable]],
    stats: list[tuple[str, str]],
    filename: str,
    download_key: str,
) -> None:
    """Render a data explorer tab with table, stats, and download button.

    Args:
        data: Sequence of data objects to display (must have .date attribute)
        display_columns: List of (column_name, formatter_func) for display DataFrame
        csv_columns: List of (column_name, formatter_func) for CSV download
        stats: List of (label, formatted_value) for summary metrics
        filename: Display name for download (e.g., "WTI Data (CSV)")
        download_key: Unique key for the download button widget
    """
    # Create display DataFrame
    display_df = pd.DataFrame([
        {name: fmt(d) for name, fmt in display_columns}
        for d in sorted(data, key=lambda x: x.date, reverse=True)
    ])

    # Summary stats row
    cols = st.columns(len(stats))
    for col, (label, value) in zip(cols, stats):
        with col:
            st.metric(label, value)

    # Display table
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Download button
    csv_df = pd.DataFrame([
        {name: fmt(d) for name, fmt in csv_columns}
        for d in sorted(data, key=lambda x: x.date, reverse=True)
    ]).to_csv(index=False)

    st.download_button(
        f"📥 Download {filename}",
        csv_df,
        filename.lower().replace(" ", "_").replace("(csv)", "").strip() + ".csv",
        "text/csv",
        key=download_key,
    )


def create_data_explorer(
    wti_data: Sequence[MarketData],
    brent_data: Sequence[MarketData],
    spread_data: Sequence[SpreadData],
    moving_averages: Sequence[MovingAverages],
) -> None:
    """Create the data explorer section with tabbed views."""
    st.divider()
    st.subheader("📊 Data Explorer")

    tab_wti, tab_brent, tab_spread, tab_analysis = st.tabs([
        "WTI Data", "Brent Data", "Spread Data", "Analysis Summary"
    ])

    # Common column definitions for market data (WTI/Brent)
    market_display_columns = [
        ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
        ("Open", lambda d: f"${d.open:.2f}"),
        ("High", lambda d: f"${d.high:.2f}"),
        ("Low", lambda d: f"${d.low:.2f}"),
        ("Close", lambda d: f"${d.close:.2f}"),
        ("Volume", lambda d: f"{int(d.volume):,}" if d.volume else "N/A"),
    ]
    market_csv_columns = [
        ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
        ("Open", lambda d: d.open),
        ("High", lambda d: d.high),
        ("Low", lambda d: d.low),
        ("Close", lambda d: d.close),
        ("Volume", lambda d: d.volume or ""),
    ]

    # WTI Data Tab
    with tab_wti:
        closes = [d.close for d in wti_data]
        _render_data_tab(
            data=wti_data,
            display_columns=market_display_columns,
            csv_columns=market_csv_columns,
            stats=[
                ("Records", str(len(wti_data))),
                ("Min", f"${min(closes):.2f}"),
                ("Max", f"${max(closes):.2f}"),
                ("Avg", f"${sum(closes)/len(closes):.2f}"),
            ],
            filename="WTI Data (CSV)",
            download_key="download_wti",
        )

    # Brent Data Tab
    with tab_brent:
        closes = [d.close for d in brent_data]
        _render_data_tab(
            data=brent_data,
            display_columns=market_display_columns,
            csv_columns=market_csv_columns,
            stats=[
                ("Records", str(len(brent_data))),
                ("Min", f"${min(closes):.2f}"),
                ("Max", f"${max(closes):.2f}"),
                ("Avg", f"${sum(closes)/len(closes):.2f}"),
            ],
            filename="Brent Data (CSV)",
            download_key="download_brent",
        )

    # Spread Data Tab
    with tab_spread:
        spreads = [d.spread for d in spread_data]
        _render_data_tab(
            data=spread_data,
            display_columns=[
                ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
                ("WTI Close", lambda d: f"${d.wti_close:.2f}"),
                ("Brent Close", lambda d: f"${d.brent_close:.2f}"),
                ("Spread", lambda d: f"${d.spread:.2f}"),
            ],
            csv_columns=[
                ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
                ("WTI_Close", lambda d: d.wti_close),
                ("Brent_Close", lambda d: d.brent_close),
                ("Spread", lambda d: d.spread),
            ],
            stats=[
                ("Records", str(len(spread_data))),
                ("Min Spread", f"${min(spreads):.2f}"),
                ("Max Spread", f"${max(spreads):.2f}"),
                ("Avg Spread", f"${sum(spreads)/len(spreads):.2f}"),
            ],
            filename="Spread Data (CSV)",
            download_key="download_spread",
        )

    # Analysis Summary Tab (has different stats pattern - trend counts)
    with tab_analysis:
        trends = [d.trend_signal for d in moving_averages if d.trend_signal]
        bullish_count = sum(1 for t in trends if t == "bullish")
        bearish_count = sum(1 for t in trends if t == "bearish")
        neutral_count = sum(1 for t in trends if t == "neutral")

        _render_data_tab(
            data=moving_averages,
            display_columns=[
                ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
                ("Spread", lambda d: f"${d.spread:.2f}"),
                ("5-Day MA", lambda d: f"${d.ma_5:.2f}" if d.ma_5 else "N/A"),
                ("20-Day MA", lambda d: f"${d.ma_20:.2f}" if d.ma_20 else "N/A"),
                ("Trend", lambda d: d.trend_signal.title() if d.trend_signal else "N/A"),
            ],
            csv_columns=[
                ("Date", lambda d: d.date.strftime("%Y-%m-%d")),
                ("Spread", lambda d: d.spread),
                ("MA_5", lambda d: d.ma_5 or ""),
                ("MA_20", lambda d: d.ma_20 or ""),
                ("Trend", lambda d: d.trend_signal or ""),
            ],
            stats=[
                ("Records", str(len(moving_averages))),
                ("📈 Bullish Days", str(bullish_count)),
                ("📉 Bearish Days", str(bearish_count)),
                ("➡️ Neutral Days", str(neutral_count)),
            ],
            filename="Analysis Data (CSV)",
            download_key="download_analysis",
        )


def _generate_sample_data(
    days: int,
    ui_config,
) -> tuple[list[MarketData], list[MarketData]]:
    """Generate sample market data for demo when real data unavailable.

    Args:
        days: Number of days of sample data to generate
        ui_config: UIConfig with sample price settings

    Returns:
        Tuple of (wti_data, brent_data) lists
    """
    import random

    base_date = datetime.now()
    wti_data, brent_data = [], []
    wti_base = ui_config.sample_wti_base_price
    brent_base = ui_config.sample_brent_base_price
    variation = ui_config.sample_price_variation

    for i in range(days, 0, -1):
        date = base_date - timedelta(days=i)
        wti_base += random.uniform(-1, 1)
        brent_base += random.uniform(-1, 1)

        wti_data.append(MarketData(
            date=date,
            open=wti_base - variation,
            high=wti_base + variation,
            low=wti_base - variation * 1.4,
            close=wti_base,
            ticker="WTI",
        ))
        brent_data.append(MarketData(
            date=date,
            open=brent_base - variation,
            high=brent_base + variation,
            low=brent_base - variation * 1.4,
            close=brent_base,
            ticker="BRENT",
        ))

    return wti_data, brent_data


def main():
    """Main application entry point."""
    setup_logging(console_output=False)
    init_services()

    # Header
    st.title("🛢️ Brent/WTI (West Texas Intermediate) Intelligence Copilot")
    st.markdown("*AI-powered WTI-Brent crude oil spread analysis*")

    # Get UI config
    ui_config = st.session_state.app_config.ui

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        days = st.slider(
            "Historical Days",
            min_value=ui_config.min_historical_days,
            max_value=ui_config.max_historical_days,
            value=ui_config.default_historical_days,
            step=1,
        )

        use_llm = st.checkbox(
            "Enable AI Analysis",
            value=True,
            help="Generate AI-powered market signals",
        )

        st.divider()

        if st.button("Refresh Data", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        # Health Status
        st.divider()
        with st.expander("System Health", expanded=False):
            health_checker = HealthChecker(
                market_client=st.session_state.market_client,
                llm_client=st.session_state.analysis_llm_client,
            )
            health_result = health_checker.check_all()

            status_emoji = {
                HealthStatus.HEALTHY: "🟢",
                HealthStatus.DEGRADED: "🟡",
                HealthStatus.UNHEALTHY: "🔴",
            }

            st.markdown(f"**Overall:** {status_emoji.get(health_result.status, '⚪')} {health_result.status.value.title()}")

            for component in health_result.components:
                emoji = status_emoji.get(component.status, "⚪")
                latency = f" ({component.latency_ms:.0f}ms)" if component.latency_ms else ""
                st.markdown(f"- {emoji} **{component.name}**{latency}")

            # Show model types
            st.markdown("---")
            st.markdown("**Model Configuration:**")
            analysis_type = get_client_type_name(st.session_state.analysis_llm_client)
            extraction_type = get_client_type_name(st.session_state.extraction_llm_client)
            st.markdown(f"- Analysis: {analysis_type}")
            st.markdown(f"- Extraction: {extraction_type}")

    # Main content
    with st.spinner("Fetching market data..."):
        try:
            fetch_use_case = FetchMarketDataUseCase(
                st.session_state.market_client,
                buffer_days=st.session_state.app_config.market_data.buffer_days,
            )
            wti_data, brent_data = fetch_use_case.execute(days=days)
        except Exception as e:
            st.error(f"Failed to fetch market data: {str(e)}")
            st.info("This may be due to market data unavailability. Using sample data for demonstration.")
            wti_data, brent_data = _generate_sample_data(days, ui_config)

    with st.spinner("Analyzing spread..."):
        try:
            analyze_use_case = AnalyzeSpreadUseCase(st.session_state.spread_calculator)
            spread_data, moving_averages, trend_summary = analyze_use_case.execute(
                wti_data, brent_data
            )
            stats = analyze_use_case.get_spread_statistics(spread_data)
        except Exception as e:
            st.error(f"Failed to analyze spread: {str(e)}")
            st.stop()

    # Data Explorer (moved up - display right after data is ready)
    create_data_explorer(wti_data, brent_data, spread_data, moving_averages)

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_spread = trend_summary.get("current_spread")
        st.metric(
            "Current Spread",
            f"${current_spread:.2f}" if current_spread else "N/A",
            delta=f"${trend_summary.get('spread_change_5d', 0):.2f} (5d)"
            if trend_summary.get("spread_change_5d") else None,
        )

    with col2:
        st.metric(
            "5-Day MA",
            f"${trend_summary.get('ma_5'):.2f}" if trend_summary.get("ma_5") else "N/A",
        )

    with col3:
        st.metric(
            "20-Day MA",
            f"${trend_summary.get('ma_20'):.2f}" if trend_summary.get("ma_20") else "N/A",
        )

    with col4:
        trend = trend_summary.get("trend", "N/A")
        trend_emoji = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}.get(trend, "")
        st.metric("Trend", f"{trend_emoji} {trend.title() if trend else 'N/A'}")

    # Chart
    st.plotly_chart(
        create_spread_chart(spread_data, moving_averages),
        use_container_width=True,
    )

    # Statistics
    with st.expander("Spread Statistics"):
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            st.metric("Min Spread", f"${stats['min']:.2f}" if stats["min"] else "N/A")
        with stat_col2:
            st.metric("Max Spread", f"${stats['max']:.2f}" if stats["max"] else "N/A")
        with stat_col3:
            st.metric("Mean Spread", f"${stats['mean']:.2f}" if stats["mean"] else "N/A")
        with stat_col4:
            st.metric("Std Dev", f"${stats['std']:.2f}" if stats["std"] else "N/A")

    # AI Signal
    if use_llm:
        st.divider()
        st.subheader("🤖 AI Market Analysis")

        with st.spinner("Generating AI analysis..."):
            try:
                app_config = st.session_state.app_config

                # Clients use their own config for inference parameters
                signal_use_case = GenerateSignalUseCase(
                    llm_client=st.session_state.analysis_llm_client,
                    extraction_client=st.session_state.extraction_llm_client,
                    signal_config=app_config.signal,
                )
                signal = signal_use_case.execute(
                    spread_data, moving_averages, trend_summary
                )
                display_signal_card(signal)
            except Exception as e:
                st.error(f"Failed to generate AI signal: {str(e)}")

    # Footer
    st.divider()
    st.caption(
        "Data sourced from Yahoo Finance. "
        "This application is for informational purposes only."
    )


if __name__ == "__main__":
    main()
