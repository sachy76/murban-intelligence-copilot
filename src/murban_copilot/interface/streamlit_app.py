"""Streamlit dashboard for Murban Crude Intelligence Copilot."""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Sequence


from murban_copilot.domain.entities import MovingAverages, SpreadData, MarketSignal
from murban_copilot.domain.spread_calculator import SpreadCalculator
from murban_copilot.infrastructure.market_data.yahoo_client import YahooFinanceClient
from murban_copilot.infrastructure.llm.llm_client import LlamaClient, MockLlamaClient
from murban_copilot.infrastructure.logging import setup_logging
from murban_copilot.application.fetch_market_data import FetchMarketDataUseCase
from murban_copilot.application.analyze_spread import AnalyzeSpreadUseCase
from murban_copilot.application.generate_signal import GenerateSignalUseCase


# Page config
st.set_page_config(
    page_title="Murban Intelligence Copilot",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stSidebar {
        background-color: #1a1f2c;
    }
    .metric-card {
        background-color: #1a1f2c;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #2d3748;
    }
    .signal-bullish {
        color: #48bb78;
        font-weight: bold;
    }
    .signal-bearish {
        color: #fc8181;
        font-weight: bold;
    }
    .signal-neutral {
        color: #ecc94b;
        font-weight: bold;
    }
    .disclaimer {
        font-size: 0.75rem;
        color: #a0aec0;
        padding: 1rem;
        background-color: #1a1f2c;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


def init_services():
    """Initialize services with caching."""
    if "market_client" not in st.session_state:
        st.session_state.market_client = YahooFinanceClient()
    if "spread_calculator" not in st.session_state:
        st.session_state.spread_calculator = SpreadCalculator()
    if "llm_client" not in st.session_state:
        # Use mock client by default for demo; can be replaced with real LlamaClient
        st.session_state.llm_client = MockLlamaClient()


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
        subplot_titles=("Murban-Brent Spread", "Spread Value"),
    )

    sorted_spread = sorted(spread_data, key=lambda x: x.date)
    sorted_ma = sorted(moving_averages, key=lambda x: x.date)

    dates = [d.date for d in sorted_spread]
    murban_prices = [d.murban_close for d in sorted_spread]
    brent_prices = [d.brent_close for d in sorted_spread]
    spreads = [d.spread for d in sorted_spread]

    # Murban price line
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=murban_prices,
            name="Murban",
            line=dict(color="#48bb78", width=2),
            hovertemplate="Murban: $%{y:.2f}<extra></extra>",
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
        height=600,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    fig.update_xaxes(gridcolor="#2d3748", showgrid=True)
    fig.update_yaxes(gridcolor="#2d3748", showgrid=True)
    fig.update_yaxes(title_text="Price ($/barrel)", row=1, col=1)
    fig.update_yaxes(title_text="Spread ($)", row=2, col=1)

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


def main():
    """Main application entry point."""
    setup_logging(console_output=False)
    init_services()

    # Header
    st.title("🛢️ Murban Intelligence Copilot")
    st.markdown("*AI-powered Murban-Brent crude oil spread analysis*")

    # Sidebar
    with st.sidebar:
        st.header("Settings")

        ticker_option = st.selectbox(
            "Base Ticker",
            options=["Murban vs Brent", "Custom"],
            index=0,
        )

        days = st.slider(
            "Historical Days",
            min_value=7,
            max_value=90,
            value=30,
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

    # Main content
    with st.spinner("Fetching market data..."):
        try:
            fetch_use_case = FetchMarketDataUseCase(st.session_state.market_client)
            murban_data, brent_data = fetch_use_case.execute(days=days)
        except Exception as e:
            st.error(f"Failed to fetch market data: {str(e)}")
            st.info("This may be due to market data unavailability. Using sample data for demonstration.")

            # Create sample data for demo
            from murban_copilot.domain.entities import MarketData
            import random

            base_date = datetime.now()
            murban_data = []
            brent_data = []

            murban_base = 85.0
            brent_base = 82.0

            for i in range(days, 0, -1):
                date = base_date - timedelta(days=i)
                murban_base += random.uniform(-1, 1)
                brent_base += random.uniform(-1, 1)

                murban_data.append(MarketData(
                    date=date,
                    open=murban_base - 0.5,
                    high=murban_base + 0.5,
                    low=murban_base - 0.7,
                    close=murban_base,
                    ticker="MURBAN",
                ))
                brent_data.append(MarketData(
                    date=date,
                    open=brent_base - 0.5,
                    high=brent_base + 0.5,
                    low=brent_base - 0.7,
                    close=brent_base,
                    ticker="BRENT",
                ))

    with st.spinner("Analyzing spread..."):
        try:
            analyze_use_case = AnalyzeSpreadUseCase(st.session_state.spread_calculator)
            spread_data, moving_averages, trend_summary = analyze_use_case.execute(
                murban_data, brent_data
            )
            stats = analyze_use_case.get_spread_statistics(spread_data)
        except Exception as e:
            st.error(f"Failed to analyze spread: {str(e)}")
            st.stop()

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
                signal_use_case = GenerateSignalUseCase(st.session_state.llm_client)
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
