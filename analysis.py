# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 12:34:44 2025

@author: 24802
"""

import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
import yfinance as yf
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from pandas.tseries.offsets import DateOffset
import plotly.graph_objects as go
from openai import OpenAI
from statsmodels.tsa.arima.model import ARIMA
import warnings
import index

def return_analysis(portfolio_returns, benchmark_returns):
    st.subheader("Return Analysis")
    time_periods = ['YTD', '1Y', '3Y', '5Y', 'All']
    selected_period = st.selectbox("Select return analysis time period", time_periods)

    st.write(f"### {selected_period} Return Analysis")
    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max()
    # Determine the time period
    now = pd.Timestamp.now()
    if selected_period == 'YTD':
        start_date = end_date.replace(month=1, day=1)
    elif selected_period == 'All':
        start_date = portfolio_returns.index.min()
    else:
        years = int(selected_period[0])
        start_date = end_date - DateOffset(years=years)

    # Data filtering
    port_ret = portfolio_returns[portfolio_returns.index >= start_date]
    bench_ret = benchmark_returns[benchmark_returns.index >= start_date]

    # Index alignment
    common_idx = port_ret.index.intersection(bench_ret.index)
    port_ret = port_ret.loc[common_idx]
    bench_ret = bench_ret.loc[common_idx]

    # Empty data check
    if len(port_ret) == 0 or len(bench_ret) == 0:
        st.warning(f"No valid data available for the {selected_period} time period, skipping analysis")
        return

    # Calculate returns (ensure scalar)
    abs_return = (1 + port_ret).prod() - 1
    bench_return = (1 + bench_ret).prod() - 1
    rel_return = abs_return - bench_return

    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Portfolio Absolute Return", f"{abs_return * 100:.2f}%")
    with col2:
        st.metric("Benchmark Absolute Return", f"{bench_return * 100:.2f}%")
    with col3:
        st.metric("Relative Return", f"{rel_return * 100:.2f}%", delta_color="inverse")

    # Prepare data for plotting
    df_plot = pd.DataFrame({
        'Portfolio': (1 + port_ret).cumprod(),
        'Benchmark': (1 + bench_ret).cumprod()
    }).reset_index()

    # Ensure data is valid
    if not df_plot.empty:
        fig = px.line(df_plot, x='date', y=['Portfolio', 'Benchmark'],
                      title=f"{selected_period} Cumulative Return Trend",
                      labels={'value': 'Cumulative Return', 'date': 'Date'})
        st.plotly_chart(fig)

        # Calculate portfolio drawdown
        cum_returns_port = (1 + port_ret).cumprod()
        running_max_port = cum_returns_port.cummax()
        drawdowns_port = cum_returns_port / running_max_port - 1

        # Calculate benchmark drawdown
        cum_returns_bench = (1 + bench_ret).cumprod()
        running_max_bench = cum_returns_bench.cummax()
        drawdowns_bench = cum_returns_bench / running_max_bench - 1

        # Prepare data for drawdown plot
        df_drawdowns = pd.DataFrame({
            'Portfolio': drawdowns_port,
            'Benchmark': drawdowns_bench
        }).reset_index()

        # Plot drawdown line chart
        fig_drawdowns = px.line(df_drawdowns, x='date', y=['Portfolio', 'Benchmark'],
                                title=f"{selected_period} Drawdown Trend",
                                labels={'value': 'Drawdown', 'date': 'Date'})
        st.plotly_chart(fig_drawdowns)
    else:
        st.warning("Unable to generate trend chart: No valid data points")

