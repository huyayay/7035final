#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px

# def rolling_volatility_analysis(portfolio_returns, benchmark_returns):
#     """Rolling volatility analysis"""
#     st.subheader("Rolling Volatility Analysis")
    
#     # Add window size selection
#     window_options = {
#         'One Week': 5,
#         'One Month': 21,
#         'One Quarter': 63,
#         'One Year': 252
#     }
#     selected_window = st.selectbox("Select window size", list(window_options.keys()), key='volatility_window')
#     window = window_options[selected_window]
    
#     # Add time period selection
#     time_periods = ['1Y', '3Y', '5Y', 'All']
#     selected_period = st.selectbox("Select analysis time period", time_periods, key='volatility_period')
    
#     # Filter data based on the selected time period
#     end_date = portfolio_returns.index.max()
    
#     if selected_period == '1Y':
#         start_date = end_date - pd.DateOffset(years=1)
#     elif selected_period == '3Y':
#         start_date = end_date - pd.DateOffset(years=3)
#     elif selected_period == '5Y':
#         start_date = end_date - pd.DateOffset(years=5)
#     else:  # All
#         start_date = portfolio_returns.index.min()
    
#     port_ret = portfolio_returns[(portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)]
#     bench_ret = benchmark_returns[(benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]
    
#     # Calculate rolling volatility (annualized)
#     port_vol = port_ret.rolling(window).std() * np.sqrt(252 / window)
#     bench_vol = bench_ret.rolling(window).std() * np.sqrt(252 / window)
    
#     # Create a DataFrame
#     df_vol = pd.DataFrame({
#         'Portfolio': port_vol,
#         'Benchmark': bench_vol
#     }).dropna()
    
#     if len(df_vol) > 0:
#         fig = px.line(df_vol, 
#                       title=f"{selected_period} Rolling Annualized Volatility ({selected_window} Window)",
#                       labels={'value': 'Volatility', 'date': 'Date'},
#                       color_discrete_map={'Portfolio': '#1f77b4', 'Benchmark': '#ff7f0e'})
#         fig.update_layout(hovermode="x unified")
#         st.plotly_chart(fig, use_container_width=True)
        
#         # Add statistical indicators
#         st.write("Volatility Statistical Indicators:")
#         col1, col2 = st.columns(2)
#         with col1:
#             st.metric("Portfolio Average Volatility", f"{port_vol.mean()*100:.2f}%")
#             st.metric("Portfolio Maximum Volatility", f"{port_vol.max()*100:.2f}%")
#         with col2:
#             st.metric("Benchmark Average Volatility", f"{bench_vol.mean()*100:.2f}%")
#             st.metric("Benchmark Maximum Volatility", f"{bench_vol.max()*100:.2f}%")
#     else:
#         st.warning("Insufficient data to calculate rolling volatility")


# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import norm

def rolling_volatility_analysis(portfolio_returns, benchmark_returns):
    """Rolling volatility analysis"""
    st.subheader("Rolling Volatility Analysis")
    
    # Add window size selection
    window_options = {
        'One Week': 5,
        'One Month': 21,
        'One Quarter': 63,
        'One Year': 252
    }
    selected_window = st.selectbox("Select window size", list(window_options.keys()), key='volatility_window')
    window = window_options[selected_window]
    
    # Add time period selection
    time_periods = ['1Y', '3Y', '5Y', 'All']
    selected_period = st.selectbox("Select analysis time period", time_periods, key='volatility_period')
    
    # Filter data based on the selected time period
    end_date = portfolio_returns.index.max()
    
    if selected_period == '1Y':
        start_date = end_date - pd.DateOffset(years=1)
    elif selected_period == '3Y':
        start_date = end_date - pd.DateOffset(years=3)
    elif selected_period == '5Y':
        start_date = end_date - pd.DateOffset(years=5)
    else:  # All
        start_date = portfolio_returns.index.min()
    
    port_ret = portfolio_returns[(portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)]
    bench_ret = benchmark_returns[(benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]
    
    # Calculate rolling volatility (annualized)
    port_vol = port_ret.rolling(window).std() * np.sqrt(252 / window)
    bench_vol = bench_ret.rolling(window).std() * np.sqrt(252 / window)
    
    # Create a DataFrame
    df_vol = pd.DataFrame({
        'Portfolio': port_vol,
        'Benchmark': bench_vol
    }).dropna()
    
    if len(df_vol) > 0:
        fig = px.line(df_vol, 
                      title=f"{selected_period} Rolling Annualized Volatility ({selected_window} Window)",
                      labels={'value': 'Volatility', 'date': 'Date'},
                      color_discrete_map={'Portfolio': '#1f77b4', 'Benchmark': '#ff7f0e'})
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistical indicators
        st.write("Volatility Statistical Indicators:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Portfolio Average Volatility", f"{port_vol.mean()*100:.2f}%")
            st.metric("Portfolio Maximum Volatility", f"{port_vol.max()*100:.2f}%")
        with col2:
            st.metric("Benchmark Average Volatility", f"{bench_vol.mean()*100:.2f}%")
            st.metric("Benchmark Maximum Volatility", f"{bench_vol.max()*100:.2f}%")

        # Calculate risk-adjusted return (Sharpe ratio)
        risk_free_rate = st.number_input("Enter the risk-free rate (%)", value=2.0) / 100
        port_sharpe = (port_ret.mean() - risk_free_rate) / port_vol.mean()
        bench_sharpe = (bench_ret.mean() - risk_free_rate) / bench_vol.mean()

        st.write("Risk-Adjusted Return Indicators:")
        col3, col4 = st.columns(2)
        with col3:
            st.metric("Portfolio Sharpe Ratio", f"{port_sharpe:.2f}")
        with col4:
            st.metric("Benchmark Sharpe Ratio", f"{bench_sharpe:.2f}")

        # Scenario simulation
        st.subheader("Scenario Simulation")
        st.write("Simulate extreme market conditions to test portfolio resilience.")
        shock_percentage = st.slider("Select the shock percentage (%)", min_value=1, max_value=50, value=20)
        shock_duration = st.slider("Select the shock duration (days)", min_value=1, max_value=30, value=5)

        # Apply the shock to the portfolio returns
        shocked_port_ret = port_ret.copy()
        shock_start_index = np.random.randint(0, len(shocked_port_ret) - shock_duration)
        shock_end_index = shock_start_index + shock_duration
        shocked_port_ret.iloc[shock_start_index:shock_end_index] *= (1 - shock_percentage / 100)

        # Recalculate volatility and Sharpe ratio after the shock
        shocked_port_vol = shocked_port_ret.rolling(window).std() * np.sqrt(252 / window)
        shocked_port_sharpe = (shocked_port_ret.mean() - risk_free_rate) / shocked_port_vol.mean()

        st.write("Portfolio Performance after Shock:")
        st.metric("Shocked Portfolio Average Volatility", f"{shocked_port_vol.mean()*100:.2f}%")
        st.metric("Shocked Portfolio Sharpe Ratio", f"{shocked_port_sharpe:.2f}")

        # Plot the shocked portfolio returns
        df_shocked = pd.DataFrame({
            'Original Portfolio': port_ret,
            'Shocked Portfolio': shocked_port_ret
        })
        fig_shocked = px.line(df_shocked, 
                              title=f"Portfolio Returns before and after {shock_percentage}% Shock",
                              labels={'value': 'Returns', 'date': 'Date'},
                              color_discrete_map={'Original Portfolio': '#1f77b4', 'Shocked Portfolio': '#ff7f0e'})
        fig_shocked.update_layout(hovermode="x unified")
        st.plotly_chart(fig_shocked, use_container_width=True)

    else:
        st.warning("Insufficient data to calculate rolling volatility")

