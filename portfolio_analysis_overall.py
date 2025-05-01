# portfolio_analysis.py
import pandas as pd
import plotly.express as px
import numpy as np
from pandas.tseries.offsets import DateOffset


def create_portfolio(price_data, static_data, weights=None, tickers=None):
    """Create a portfolio"""
    # Remove duplicate (date, code) combinations and keep the first record
    price_data = price_data.drop_duplicates(subset=['date', 'code'], keep='first')

    if weights is not None and isinstance(weights, list) and tickers is not None and isinstance(tickers, list):
        # Reorganize the price data, with code as columns and date as index
        price_data_pivot = price_data.pivot(index='date', columns='code', values='value')
        # Filter the price data corresponding to the user-specified tickers
        price_data_pivot = price_data_pivot[tickers]
        portfolio_returns = (price_data_pivot.pct_change() * weights).sum(axis=1)
    else:
        # Filter the stocks that exist in the price data
        available_codes = price_data['code'].unique()
        static_data = static_data[static_data.index.isin(available_codes)]

        if weights is None:
            # Equal weights
            num_stocks = len(static_data)
            weights = np.ones(num_stocks) / num_stocks

        # Reorganize the price data, with code as columns and date as index
        price_data_pivot = price_data.pivot(index='date', columns='code', values='value')
        # Ensure that the columns of the price data are consistent with the index of the static data
        price_data_pivot = price_data_pivot[static_data.index]

        portfolio_returns = (price_data_pivot.pct_change() * weights).sum(axis=1)

    return portfolio_returns


def factor_exposure_analysis(portfolio_returns, factor_exposures, factor_cov, tickers, static_data, price_data, weights):
    """Factor exposure analysis"""
    import streamlit as st
    st.subheader("Factor Exposure Analysis")
    st.write("Visualize the portfolio's exposure to key market factors")

    if tickers:
        # Assume that the index of factor_exposures is ticker, filter the data according to ticker
        selected_factor_exposures = factor_exposures.loc[tickers]
        # Sum the filtered data by factor (assuming the columns are factors)
        exposure_sum = selected_factor_exposures.sum(axis=0).sort_values(ascending=False)
    else:
        # Keep the original logic here. When no ticker-related file is uploaded, display in the original way
        exposure_sum = factor_exposures.sum(axis=0).sort_values(ascending=False)

    exposure_sum = exposure_sum[exposure_sum != 0]

    # Draw a bar chart of factor exposure
    fig = px.bar(exposure_sum, title="Portfolio Factor Exposure")
    fig.update_layout(xaxis_tickangle=-45)  # Arrange the labels diagonally from bottom to top, set to -45 degrees
    st.plotly_chart(fig)

    # Return the names of the exposed factors (for the heatmap)
    return exposure_sum.index.tolist()


def return_analysis(portfolio_returns, benchmark_returns):
    """Modified return analysis function with state preservation support"""
    import streamlit as st

    # State initialization
    if 'return_analysis_state' not in st.session_state:
        st.session_state.return_analysis_state = {
            'selected_period': 'YTD',
            'last_processed_data': None,
            'active_tab': True
        }

    # Data validity check
    if portfolio_returns is None or benchmark_returns is None:
        st.warning("Waiting for input data...")
        return

    # Automatically maintain tab focus
    if st.session_state.return_analysis_state['active_tab']:
        st.query_params = {"tab": "return_analysis"}

    st.subheader("Return Analysis")

    # Time period selector (with state memory)
    time_periods = ['YTD', '1Y', '3Y', '5Y', 'All']
    selected_period = st.selectbox(
        "Select the analysis time period",
        time_periods,
        index=time_periods.index(st.session_state.return_analysis_state['selected_period']),
        key='return_period_selector'
    )

    # Update the state when the selection changes
    if selected_period != st.session_state.return_analysis_state['selected_period']:
        st.session_state.return_analysis_state['selected_period'] = selected_period
        st.rerun()

    # Calculate the time range
    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max()
    now = pd.Timestamp.now()

    # Dynamic time range calculation
    period_map = {
        'YTD': end_date.replace(month=1, day=1),
        '1Y': end_date - pd.DateOffset(years=1),
        '3Y': end_date - pd.DateOffset(years=3),
        '5Y': end_date - pd.DateOffset(years=5),
        'All': start_date
    }
    start_date = period_map[selected_period]

    # Data slicing (with cache)
    @st.cache_data(ttl=300)
    def get_period_data(_portfolio, _benchmark, start, end):
        port = _portfolio[(portfolio_returns.index >= start) & (portfolio_returns.index <= end)]
        bench = _benchmark[(benchmark_returns.index >= start) & (benchmark_returns.index <= end)]
        common_idx = port.index.intersection(bench.index)
        return port.loc[common_idx], bench.loc[common_idx]

    port_ret, bench_ret = get_period_data(portfolio_returns, benchmark_returns, start_date, end_date)

    # Indicator calculation
    with st.container():
        cols = st.columns(3)
        abs_return = (1 + port_ret).prod() - 1
        bench_return = (1 + bench_ret).prod() - 1
        rel_return = abs_return - bench_return

        cols[0].metric("Portfolio Absolute Return",
                       f"{abs_return * 100:.2f}%",
                       help="Total portfolio return within the selected time period")
        cols[1].metric("Benchmark Absolute Return",
                       f"{bench_return * 100:.2f}%",
                       help="Total return of the benchmark index during the same period")
        cols[2].metric("Excess Return",
                       f"{rel_return * 100:.2f}%",
                       delta_color="off",
                       help="Excess return of the portfolio relative to the benchmark")

    # Visualization part
    with st.container(border=True):
        st.write(f"**{selected_period} Return Trend**")

        # Cumulative return curve
        cumulative_df = pd.DataFrame({
            'Portfolio': (1 + port_ret).cumprod(),
            'Benchmark': (1 + bench_ret).cumprod()
        }).reset_index()

        fig = px.line(cumulative_df, x='date', y=['Portfolio', 'Benchmark'],
                      labels={'value': 'Cumulative Return', 'date': 'Date'},
                      height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Drawdown analysis
        st.write("**Drawdown Analysis**")

        def calculate_drawdown(series):
            cumulative = (1 + series).cumprod()
            peak = cumulative.expanding().max()
            return (cumulative / peak) - 1

        drawdown_df = pd.DataFrame({
            'Portfolio Drawdown': calculate_drawdown(port_ret),
            'Benchmark Drawdown': calculate_drawdown(bench_ret)
        }).reset_index()

        fig_drawdown = px.area(drawdown_df, x='date', y=['Portfolio Drawdown', 'Benchmark Drawdown'],
                               labels={'value': 'Drawdown Percentage', 'date': 'Date'},
                               line_shape='hv',
                               height=400)
        fig_drawdown.update_yaxes(tickformat=".1%")
        st.plotly_chart(fig_drawdown, use_container_width=True)

    # Save the state of the latest processed data
    st.session_state.return_analysis_state['last_processed_data'] = {
        'portfolio': portfolio_returns,
        'benchmark': benchmark_returns
    }