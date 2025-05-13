# combined999_v4.py
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import numpy as np
from pandas.tseries.offsets import DateOffset
import index 
import return_prediction as predict
import volatility_analysis as vola
import chatwithai as ai
import asyncio
from input import process_user_input
from archive_manager import PortfolioArchive, show_archive_management_ui
from portfolio_analysis_overall import create_portfolio, factor_exposure_analysis, return_analysis
from datetime import datetime
import seaborn as sns  # New import
import matplotlib.pyplot as plt  # New import

@st.cache_data
def load_data():
    """Load all datasets"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_paths = {
        "factor_exposures": "Factor_Exposures.xlsx",
        "factor_cov": "Factor_Covariance_Matrix.xlsx",
        "static_data": "Static_Data.xlsx",
        "price_data": "Constituent_Price_History.csv",
        "bench_data": "benchdata.xlsx",
        "stock_cap": "stock_cap.csv"  # New file path
    }

    data = {}
    for key, filename in file_paths.items():
        filepath = os.path.join(base_dir, filename)
        try:
            if filename.endswith('.xlsx'):
                data[key] = pd.read_excel(filepath, index_col=0, engine='openpyxl')
            elif filename.endswith('.csv'):
                data[key] = pd.read_csv(filepath, parse_dates=['date'], infer_datetime_format=True) if key != "stock_cap" else pd.read_csv(filepath)
        except FileNotFoundError:
            st.error(f"File {filename} not found. Please check the file path.")
            return None, None, None, None, None, None
        except Exception as e:
            st.error(f"Failed to load {filename}: {str(e)}")
            return None, None, None, None, None, None

    return data['factor_exposures'], data['factor_cov'], data['static_data'], data['price_data'], data['bench_data'], data['stock_cap']

async def process_data(tickers, weights, submit, archive_data):
    st.subheader("Investment Portfolio Analysis")
    session = st.session_state
    session.setdefault('cached_portfolio', None)
    session.setdefault('cached_benchmark', None)
    session.setdefault('analysis_period', 'YTD')
    session.setdefault('analysis_confirmed', False)  # Add confirmation analysis flag
    session.setdefault('current_tickers', [])  # New
    session.setdefault('current_weights', [])  

    # Load basic data (regardless of submission)
    factor_exposures, factor_cov, static_data, price_data, bench_data, stock_cap = load_data()
    if any(data is None for data in [factor_exposures, factor_cov, static_data, price_data, bench_data, stock_cap]):
        st.stop()

    if tickers and weights and submit:
        session.analysis_confirmed = True  # Set confirmation analysis flag to True

        # Data cache check
        if (session.cached_portfolio is None or 
            session.cached_benchmark is None or 
            session.get('current_tickers') != tickers):
            
            # Stock ticker validity verification
            valid_tickers = []
            valid_weights = []
            for ticker, weight in zip(tickers, weights):
                if ticker in static_data.index:
                    valid_tickers.append(ticker)
                    valid_weights.append(weight)
                else:
                    st.error(f"Invalid stock ticker: {ticker}. Please check your input.")
            
            if not valid_tickers:
                return

            tickers, weights = valid_tickers, valid_weights
            weights = np.array(weights) / sum(weights)

            # Create portfolio returns
            portfolio_returns = create_portfolio(price_data, static_data, weights.tolist(), tickers)
            
            # Benchmark data processing
            bench_data = bench_data.reset_index()
            common_dates = set(price_data['date'].dt.date.unique()) & set(bench_data['date'].dt.date.unique())
            benchmark_data = bench_data[bench_data['date'].dt.date.isin(common_dates)].set_index('date')
            benchmark_returns = benchmark_data['close'].pct_change()

            # Time series alignment
            portfolio_returns = portfolio_returns.reindex(portfolio_returns.index.union(benchmark_returns.index), method='ffill')
            benchmark_returns = benchmark_returns.reindex(benchmark_returns.index.union(portfolio_returns.index), method='ffill')
            common_index = portfolio_returns.index.intersection(benchmark_returns.index)
            portfolio_returns = portfolio_returns.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]

            # Cache calculation results
            session.cached_portfolio = portfolio_returns
            session.cached_benchmark = benchmark_returns
            session.current_tickers = tickers.copy()
            session.current_tickers = tickers.copy()
            session.current_weights = np.array(weights).copy()  
        else:
            # Use cached data
            portfolio_returns = session.cached_portfolio
            benchmark_returns = session.cached_benchmark

        # Set active state before creating tabs
        session.active_tab = "Return Analysis"  # Default active tab is Return Analysis

        # Keep editing state when there is archived data
        if archive_data and session.get('editing_existing'):
            session.editing_portfolio = archive_data
            session.input_rows = [
                {"ticker": t, "weight": w} 
                for t, w in zip(archive_data['tickers'], archive_data['weights'])
            ]

    # Prioritize using cached data for display
    portfolio_returns = session.cached_portfolio
    benchmark_returns = session.cached_benchmark

    # Only display analysis content after clicking the confirm button
    if session.analysis_confirmed:
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["Portfolio Composition", "Factor Exposure Analysis", "Return Analysis"])

        with tab1:
            # Display archived information
            if archive_data:
                st.info(f"The current analysis is for the archived portfolio: {archive_data['name']}")
                cols = st.columns([3, 1, 1])
                with cols[1]:
                    if st.button("📝 Edit Portfolio", key="edit_portfolio"):
                        st.session_state.editing_existing = archive_data['name']
                        st.session_state.input_rows = [
                            {"ticker": t, "weight": w} 
                            for t, w in zip(archive_data['tickers'], archive_data['weights'])
                        ]
                        st.rerun()
                with cols[2]:
                    if st.button("🗑️ Delete Portfolio", key="delete_portfolio"):
                        archive = PortfolioArchive()
                        archive.delete_portfolio(archive_data['name'])
                        st.success(f"Portfolio '{archive_data['name']}' has been deleted!")
                        st.rerun()
            
            # Portfolio composition table
            portfolio_df = pd.DataFrame({
                "Ticker": session.current_tickers,
                "Weight (%)": [f"{w*100:.2f}%" for w in session.current_weights]
            })
            st.dataframe(portfolio_df, use_container_width=True)

            # Detailed information of constituent stocks
            selected_static_data_list = []
            for ticker in tickers:
                try:
                    selected_static_data_list.append(static_data.loc[ticker].to_frame().T)
                except KeyError:
                    continue
            if selected_static_data_list:
                selected_static_data = pd.concat(selected_static_data_list)
                st.write("Detailed information of constituent stocks:")
                st.dataframe(selected_static_data)
            else:
                st.write("No relevant stock information found.")

            # Sector weight distribution chart
            st.subheader("Sector Weight Distribution")
            try:
                sector_data = static_data.loc[session.current_tickers, 'sector']
                sector_weights = pd.Series(session.current_weights, index=sector_data.index).groupby(sector_data).sum()

                top_sectors = sector_weights.nlargest(10).sort_values(ascending=True)
                fig = px.bar(top_sectors, orientation='h', title="Top 10 Sector Weight Distribution",
                            labels={'value': 'Weight Proportion', 'y': 'Sector'}, color_discrete_sequence=['#1f77b4']*len(top_sectors))
                fig.update_layout(height=max(400, 40*len(top_sectors)), xaxis_tickformat=".1%", margin=dict(l=150, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating sector weight distribution chart: {str(e)}")

            # Market capitalization distribution pie chart
            st.subheader("Stock Market Cap Distribution")
            try:
                cap_data = stock_cap.set_index('code').loc[session.current_tickers, 'cap_category']
                cap_weights = pd.Series(session.current_weights, index=cap_data.index).groupby(cap_data).sum()
                fig = px.pie(cap_weights, values=cap_weights, names=cap_weights.index, title="Stock Market Cap Distribution")
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating market capitalization distribution chart: {str(e)}")

        with tab2:
            # Factor exposure analysis
            selected_factors = factor_exposure_analysis(portfolio_returns, factor_exposures, factor_cov, tickers, static_data, price_data, weights)

            # Factor covariance heatmap
            st.subheader("Factor Covariance Heatmap")
            try:
                # Filter the corresponding factors in the covariance matrix
                factor_cov_filtered = factor_cov.loc[selected_factors, selected_factors]

                # Custom color map: normalize to [0, 1]
                color_scale = [
                    [0.0, "blue"],       # Dark blue for values less than -20
                    [0.25, "lightblue"], # Light blue for values between -20 and -1
                    [0.5, "white"],      # White for values between -1 and 1
                    [0.75, "lightcoral"],# Light red for values between 1 and 20
                    [1.0, "red"]         # Dark red for values greater than 20
                ]

                # Draw the heatmap
                fig_cov = px.imshow(
                    factor_cov_filtered,
                    color_continuous_scale=color_scale,
                    zmin=-max(abs(factor_cov_filtered.min().min()), abs(factor_cov_filtered.max().max())),  # Set the minimum value of the color map
                    zmax=max(abs(factor_cov_filtered.min().min()), abs(factor_cov_filtered.max().max())),  # Set the maximum value of the color map
                    labels=dict(x="Factor", y="Factor", color="Covariance"),
                    title="Factor Covariance Matrix"
                )

                # Update layout and color bar
                fig_cov.update_layout(
                    height=500,
                    margin=dict(l=50, r=50, t=50, b=50),
                    coloraxis_colorbar=dict(
                        title="Covariance",
                        tickvals=[
                            -max(abs(factor_cov_filtered.min().min()), abs(factor_cov_filtered.max().max())),  # Minimum value
                            0.5,  # Middle value
                            max(abs(factor_cov_filtered.min().min()), abs(factor_cov_filtered.max().max()))   # Maximum value
                        ],
                        ticktext=[
                            "Strong Negative",  # Minimum value corresponds to dark blue
                            "Neutral",          # Middle value corresponds to white
                            "Strong Positive"   # Maximum value corresponds to dark red
                        ],
                        lenmode="fraction",  # Set the length mode of the color bar
                        len=0.8,             # Adjust the length of the color bar
                        yanchor="middle",    # Set the vertical alignment of the color bar
                        y=0.5,               # Adjust the position of the color bar
                        ticks="outside",     # Place the tick marks outside the color bar
                        tickfont=dict(size=12, family="Arial")  # Adjust the tick font size and font
                    )
                )

                # Display the heatmap
                st.plotly_chart(fig_cov, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating factor covariance heatmap: {str(e)}")

        with tab3:
            # Return analysis
            return_analysis(portfolio_returns, benchmark_returns)

            # AI summary function
            API_TOKEN = ai.read_token_json(r"C:\Users\80580\Desktop\MFIN7035\new\PLB_HKU_Factor_data\AI\token.json")
            if API_TOKEN:
                analysis_results = index.performance_metrics(portfolio_returns, benchmark_returns)
                st.subheader("AI Investment Portfolio Performance Summary")
                with st.spinner("Generating AI summary..."):
                    summary = await ai.chat_with_AI(analysis_results, "Analyze the performance difference between the investment portfolio and the benchmark", API_TOKEN)
                    st.write(summary)
            else:
                st.info("AI function skipped. API key is missing.")

            # Volatility analysis
            st.markdown("---")
            if st.session_state.rolling_volatility_results is None:
                st.session_state.rolling_volatility_results = vola.rolling_volatility_analysis(portfolio_returns, benchmark_returns)
            else:
                st.write(st.session_state.rolling_volatility_results)

            # Return prediction
            st.markdown("---")
            if st.session_state.return_prediction_results is None:
                st.session_state.return_prediction_results = predict.return_prediction(portfolio_returns, benchmark_returns)
            else:
                st.write(st.session_state.return_prediction_results)
    else:
        st.info("ℹ️ Please enter stock tickers and weights, upload a weight file, or select an archived portfolio, then click the Confirm button.")

async def main():
    st.title("Investment Portfolio Analysis Dashboard")
    
    # Initialize session_state
    session = st.session_state
    archive = PortfolioArchive()
    session.setdefault('archive_data', archive.data)
    session.setdefault('editing_portfolio', None)
    session.setdefault('editing_existing', None)
    session.setdefault('input_rows', [{"ticker": "", "weight": 0.0}])
    session.setdefault('rolling_volatility_results', None)
    session.setdefault('return_prediction_results', None)

    # Main page layout adjustment
    col_main_left, col_main_middle, col_main_right = st.columns([0.1, 9.8, 0.1])

    with col_main_left:
        # Sidebar input function area
        with st.sidebar:
            st.subheader("Portfolio Input")
            saved_portfolios = archive.data["portfolios"]
            
            # Archived selection logic
            selected_archive = st.selectbox(
                "Select an existing portfolio",
                ["Create New Portfolio"] + [p["name"] for p in saved_portfolios],
                key="archive_selector"
            )
            
            if selected_archive != "Create New Portfolio":
                # Load archived data
                archive_data = next(p for p in saved_portfolios if p["name"] == selected_archive)
                session['editing_portfolio'] = archive_data
                
                # Rename function area
                cols_rename = st.columns([2, 2])
                with cols_rename[0]:
                    new_name = st.text_input("Rename Portfolio", value=archive_data['name'])
                with cols_rename[1]:
                    if st.button("Rename Confirm"):
                        if new_name.strip() != archive_data['name']:
                            archive.rename_portfolio(archive_data['name'], new_name.strip())
                            st.success("Name updated!")
                            st.rerun()
                
                # Operation buttons
                col_load, col_del = st.columns(2)
                with col_load:
                    if st.button("📥 Load Portfolio", key="load_archive"):
                        session.input_rows = [
                            {"ticker": t, "weight": w} 
                            for t, w in zip(archive_data['tickers'], archive_data['weights'])
                        ]
                        st.rerun()
                with col_del:
                    if st.button("🗑️ Delete Portfolio", key="del_archive"):
                        archive.delete_portfolio(archive_data['name'])
                        st.success(f"Portfolio '{archive_data['name']}' has been deleted")
                        st.rerun()
            else:
                session.pop('editing_portfolio', None)

            # Process user input
            tickers, weights, submit = process_user_input()

    with col_main_middle:
        # Add container style
        with st.container():
            st.markdown("""
                <style>
                .main-container {
                    padding: 2rem;
                    border-radius: 10px;
                    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                    margin: 1rem 0;
                }
                </style>
            """, unsafe_allow_html=True)
            show_archive_management_ui(archive)

    # Get archived state
    archive_data = session.get('editing_portfolio', None)
    await process_data(tickers, weights, submit, archive_data)


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            loop.run_until_complete(main())
        else:
            asyncio.run(main())
    except RuntimeError:
        asyncio.run(main())