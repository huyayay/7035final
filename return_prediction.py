#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.linear_model import LinearRegression
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# import warnings

# def return_prediction(portfolio_returns, benchmark_returns):
#     """Combined return prediction function, including linear regression, SARIMAX and ARIMA model predictions"""
#     st.subheader("Return Prediction Analysis")
    
#     # Select the prediction model
#     model_type = st.radio("Select the prediction model", ["Linear Regression", "ARIMA", "SARIMAX"], horizontal=True)
#     periods = st.slider("Select the number of prediction days", 10, 60, 30)
    
#     # Prepare data based on the model type
#     end_date = portfolio_returns.index.max()
#     if model_type == "Linear Regression":
#         # Use data from the last 5 months for linear regression
#         start_date = end_date - pd.DateOffset(months=5)
#     else:  # ARIMA and SARIMAX
#         # Use data from the last 12 months for ARIMA and SARIMAX
#         start_date = end_date - pd.DateOffset(months=12)
    
#     # Filter portfolio and benchmark data
#     port_ret = portfolio_returns[(portfolio_returns.index >= start_date)].dropna()
#     bench_ret = benchmark_returns[(benchmark_returns.index >= start_date)].dropna()
    
#     if len(port_ret) < 10 or len(bench_ret) < 10:
#         st.warning("Insufficient data (at least 10 trading days of data are required) to make a prediction")
#         return
    
#     if model_type == "Linear Regression":
#         # Linear regression prediction
#         def linear_predict(returns):
#             X = np.arange(len(returns)).reshape(-1, 1)
#             y = returns.values
#             model = LinearRegression()
#             model.fit(X, y)
#             future_X = np.arange(len(returns), len(returns)+periods).reshape(-1, 1)
#             return model.predict(future_X)
        
#         # Predict portfolio and benchmark
#         port_pred = linear_predict(port_ret)
#         bench_pred = linear_predict(bench_ret)
    
#     elif model_type == "ARIMA":
#         # Suppress warning messages
#         warnings.filterwarnings("ignore")
        
#         def arima_predict(returns):
#             # Simple ARIMA(2, 1, 2) model
#             model = ARIMA(returns, order=(2, 1, 2))
#             model_fit = model.fit()
#             forecast = model_fit.get_forecast(steps=periods)
#             return forecast.predicted_mean.values
        
#         # Predict portfolio and benchmark
#         port_pred = arima_predict(port_ret)
#         bench_pred = arima_predict(bench_ret)
        
#     else:  # SARIMAX model
#         # Suppress warning messages
#         warnings.filterwarnings("ignore")
        
#         def sarimax_predict(returns):
#             # Simple SARIMAX(1,0,1)(0,0,0,0) model
#             model = SARIMAX(returns, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
#             model_fit = model.fit()
#             forecast = model_fit.get_forecast(steps=periods)
#             return forecast.predicted_mean.values
        
#         # Predict portfolio and benchmark
#         port_pred = sarimax_predict(port_ret)
#         bench_pred = sarimax_predict(bench_ret)
    
#     # Create a date index
#     future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=periods)
    
#     # Prepare data for plotting
#     def prepare_plot_data(returns, pred, name):
#         history_df = pd.DataFrame({
#             'Date': returns.index,
#             'Return': returns,
#             'Type': f'{name} Historical Data'
#         })
        
#         prediction_df = pd.DataFrame({
#             'Date': future_dates,
#             'Return': pred,
#             'Type': f'{name} Forecast Data'
#         })
        
#         return pd.concat([history_df, prediction_df])
    
#     # Combine portfolio and benchmark data
#     port_plot = prepare_plot_data(port_ret, port_pred, "Portfolio")
#     bench_plot = prepare_plot_data(bench_ret, bench_pred, "Benchmark")
#     combined_plot = pd.concat([port_plot, bench_plot])
    
#     # Plot the chart
#     fig = px.line(combined_plot, x='Date', y='Return', color='Type',
#                  title=f"Return Forecast for the Next {periods} Days ({model_type} Model)",
#                  labels={'Return': 'Daily Return', 'Date': 'Date'},
#                  color_discrete_map={
#                      'Portfolio Historical Data': '#1f77b4',
#                      'Portfolio Forecast Data': '#6495ED',  # Light blue
#                      'Benchmark Historical Data': '#FFA500', # Orange
#                      'Benchmark Forecast Data': '#FFD700'    # Gold
#                  })
    
#     # Add a zero line and a current date line
#     fig.add_hline(y=0, line_dash="dash", line_color="gray")
#     fig.add_vline(x=end_date, line_dash="dash", line_color="gray")
    
#     # Adjust the layout
#     fig.update_layout(
#         hovermode="x unified",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         yaxis_tickformat=".2%"
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # Display forecast statistics
#     st.subheader("Forecast Statistics")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Portfolio Forecast Average Daily Return", f"{port_pred.mean()*100:.2f}%")
#         st.metric("Portfolio Forecast Volatility", f"{port_pred.std()*100:.2f}%")
#     with col2:
#         st.metric("Benchmark Forecast Average Daily Return", f"{bench_pred.mean()*100:.2f}%")
#         st.metric("Benchmark Forecast Volatility", f"{bench_pred.std()*100:.2f}%")


# In[ ]:


# import streamlit as st
# import pandas as pd
# import numpy as np
# import plotly.express as px
# from sklearn.linear_model import LinearRegression
# from statsmodels.tsa.statespace.sarimax import SARIMAX
# from statsmodels.tsa.arima.model import ARIMA
# from prophet import Prophet
# import pmdarima as pm
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# import warnings

# def model_introduction(model_type):
#     if model_type == "Linear Regression":
#         st.markdown("### 线性回归模型介绍")
#         st.write("**原理**：线性回归是一种基本的统计模型，它假设因变量和自变量之间存在线性关系。在时间序列预测中，通常将时间作为自变量，收益率作为因变量，通过最小化残差平方和来拟合一条直线，从而进行预测。")
#         st.write("**适用场景**：适用于数据具有明显的线性趋势，且不存在复杂的季节性和周期性变化的情况。")
#         st.write("**优点**：模型简单易懂，计算速度快，可解释性强。")
#         st.write("**缺点**：无法捕捉数据中的非线性特征和季节性变化，预测精度可能较低。")
#     elif model_type == "ARIMA":
#         st.markdown("### ARIMA模型介绍")
#         st.write("**原理**：ARIMA（自回归积分滑动平均模型）是一种广泛应用于时间序列预测的统计模型。它结合了自回归（AR）、差分（I）和滑动平均（MA）三个部分。自回归部分考虑了时间序列的历史值对当前值的影响，差分部分用于处理非平稳数据，滑动平均部分考虑了随机误差项的影响。")
#         st.write("**适用场景**：适用于具有一定自相关性和趋势性的非平稳时间序列数据。")
#         st.write("**优点**：能够处理非平稳数据，考虑了时间序列的自相关性，预测精度相对较高。")
#         st.write("**缺点**：需要手动确定模型的阶数（p, d, q），对于复杂的时间序列数据，模型选择可能比较困难。")
#     elif model_type == "SARIMAX":
#         st.markdown("### SARIMAX模型介绍")
#         st.write("**原理**：SARIMAX（季节性自回归积分滑动平均外生回归模型）是 ARIMA 模型的扩展，它在 ARIMA 模型的基础上考虑了季节性因素和外生变量的影响。通过引入季节性差分和季节性自回归、滑动平均项，能够更好地捕捉时间序列中的季节性变化。")
#         st.write("**适用场景**：适用于具有明显季节性变化的时间序列数据，并且可以考虑外生变量的影响。")
#         st.write("**优点**：能够处理具有季节性和外生变量的复杂时间序列数据，预测精度较高。")
#         st.write("**缺点**：模型参数较多，需要更多的数据进行训练，计算复杂度较高。")
#     elif model_type == "Prophet":
#         st.markdown("### Prophet模型介绍")
#         st.write("**原理**：Prophet 是 Facebook 开发的一个开源时间序列预测库，它基于加法模型，将时间序列分解为趋势、季节性和节假日效应三个部分。通过对每个部分进行建模，能够很好地处理时间序列中的趋势变化、季节性变化和节假日效应。")
#         st.write("**适用场景**：适用于具有明显季节性和趋势变化的时间序列数据，特别是在处理包含节假日等特殊日期的数据集时表现出色。")
#         st.write("**优点**：模型简单易用，自动处理季节性和趋势变化，对异常值有较好的鲁棒性。")
#         st.write("**缺点**：对于复杂的非线性关系和高频数据的处理能力相对较弱。")
#     elif model_type == "Auto - ARIMA":
#         st.markdown("### Auto - ARIMA模型介绍")
#         st.write("**原理**：Auto - ARIMA 是一种自动化的 ARIMA 模型选择方法，它通过网格搜索和信息准则（如 AIC、BIC）自动选择最优的 ARIMA 模型阶数（p, d, q）。在搜索过程中，会尝试不同的阶数组合，并选择使信息准则最小的模型。")
#         st.write("**适用场景**：适用于各种类型的时间序列数据，特别是在不知道合适的 ARIMA 模型阶数时。")
#         st.write("**优点**：自动选择最优模型，节省了手动调参的时间和精力，提高了模型选择的效率。")
#         st.write("**缺点**：搜索过程可能比较耗时，特别是在数据量较大或搜索空间较大时。")

# def return_prediction(portfolio_returns, benchmark_returns):
#     """Combined return prediction function, including linear regression, SARIMAX and ARIMA model predictions"""
#     st.subheader("Return Prediction Analysis")
    
#     # 选择预测模型
#     model_types = ["Linear Regression", "ARIMA", "SARIMAX", "Prophet", "Auto - ARIMA"]
#     model_type = st.radio("Select the prediction model", model_types, horizontal=True)
#     periods = st.slider("Select the number of prediction days", 10, 60, 30)
    
#     # 显示模型介绍
#     model_introduction(model_type)
    
#     # 准备数据
#     end_date = portfolio_returns.index.max()
#     if model_type == "Linear Regression":
#         # 线性回归使用最近5个月的数据
#         start_date = end_date - pd.DateOffset(months=5)
#     else:  # ARIMA、SARIMAX、Prophet、Auto - ARIMA
#         # 其他模型使用最近12个月的数据
#         start_date = end_date - pd.DateOffset(months=12)
    
#     # 过滤投资组合和基准数据
#     port_ret = portfolio_returns[(portfolio_returns.index >= start_date)].dropna()
#     bench_ret = benchmark_returns[(benchmark_returns.index >= start_date)].dropna()
    
#     if len(port_ret) < 10 or len(bench_ret) < 10:
#         st.warning("Insufficient data (at least 10 trading days of data are required) to make a prediction")
#         return
    
#     # 划分训练集和测试集
#     train_size = int(len(port_ret) * 0.8)
#     port_train = port_ret[:train_size]
#     port_test = port_ret[train_size:]
#     bench_train = bench_ret[:train_size]
#     bench_test = bench_ret[train_size:]
    
#     models = []
#     port_preds = []
#     bench_preds = []
#     port_errors = []
#     bench_errors = []
    
#     for model_name in model_types:
#         if model_name == "Linear Regression":
#             def linear_predict(returns):
#                 X = np.arange(len(returns)).reshape(-1, 1)
#                 y = returns.values
#                 model = LinearRegression()
#                 model.fit(X, y)
#                 future_X = np.arange(len(returns), len(returns)+len(port_test)).reshape(-1, 1)
#                 return model.predict(future_X)
            
#             port_pred = linear_predict(port_train)
#             bench_pred = linear_predict(bench_train)
#         elif model_name == "ARIMA":
#             def arima_predict(returns):
#                 warnings.filterwarnings("ignore")
#                 model = ARIMA(returns, order=(2, 1, 2))
#                 model_fit = model.fit()
#                 forecast = model_fit.get_forecast(steps=len(port_test))
#                 return forecast.predicted_mean.values
            
#             port_pred = arima_predict(port_train)
#             bench_pred = arima_predict(bench_train)
#         elif model_name == "SARIMAX":
#             def sarimax_predict(returns):
#                 warnings.filterwarnings("ignore")
#                 model = SARIMAX(returns, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
#                 model_fit = model.fit()
#                 forecast = model_fit.get_forecast(steps=len(port_test))
#                 return forecast.predicted_mean.values
            
#             port_pred = sarimax_predict(port_train)
#             bench_pred = sarimax_predict(bench_train)
#         elif model_name == "Prophet":
#             def prophet_predict(returns):
#                 df = pd.DataFrame({'ds': returns.index, 'y': returns.values})
#                 model = Prophet()
#                 model.fit(df)
#                 future = model.make_future_dataframe(periods=len(port_test))
#                 forecast = model.predict(future)
#                 return forecast['yhat'][-len(port_test):].values
            
#             port_pred = prophet_predict(port_train)
#             bench_pred = prophet_predict(bench_train)
#         elif model_name == "Auto - ARIMA":
#             def auto_arima_predict(returns):
#                 model = pm.auto_arima(returns, seasonal=True, m=12)
#                 forecast, conf_int = model.predict(n_periods=len(port_test), return_conf_int=True)
#                 return forecast
            
#             port_pred = auto_arima_predict(port_train)
#             bench_pred = auto_arima_predict(bench_train)
        
#         models.append(model_name)
#         port_preds.append(port_pred)
#         bench_preds.append(bench_pred)
        
#         port_rmse = np.sqrt(mean_squared_error(port_test, port_pred))
#         port_mae = mean_absolute_error(port_test, port_pred)
#         bench_rmse = np.sqrt(mean_squared_error(bench_test, bench_pred))
#         bench_mae = mean_absolute_error(bench_test, bench_pred)
        
#         port_errors.append((port_rmse, port_mae))
#         bench_errors.append((bench_rmse, bench_mae))
    
#     # 选择最佳模型
#     port_best_model_index = np.argmin([rmse for rmse, _ in port_errors])
#     bench_best_model_index = np.argmin([rmse for rmse, _ in bench_errors])
    
#     port_best_model = models[port_best_model_index]
#     bench_best_model = models[bench_best_model_index]
    
#     st.markdown("### 最佳模型选择")
#     st.write(f"投资组合的最佳预测模型是: {port_best_model}")
#     st.write(f"基准的最佳预测模型是: {bench_best_model}")
    
#     # 使用最佳模型进行预测
#     if model_type == "Linear Regression":
#         def linear_predict(returns):
#             X = np.arange(len(returns)).reshape(-1, 1)
#             y = returns.values
#             model = LinearRegression()
#             model.fit(X, y)
#             future_X = np.arange(len(returns), len(returns)+periods).reshape(-1, 1)
#             return model.predict(future_X)
        
#         port_pred = linear_predict(port_ret)
#         bench_pred = linear_predict(bench_ret)
#     elif model_type == "ARIMA":
#         def arima_predict(returns):
#             warnings.filterwarnings("ignore")
#             model = ARIMA(returns, order=(2, 1, 2))
#             model_fit = model.fit()
#             forecast = model_fit.get_forecast(steps=periods)
#             return forecast.predicted_mean.values
        
#         port_pred = arima_predict(port_ret)
#         bench_pred = arima_predict(bench_ret)
#     elif model_type == "SARIMAX":
#         def sarimax_predict(returns):
#             warnings.filterwarnings("ignore")
#             model = SARIMAX(returns, order=(2, 1, 2), seasonal_order=(1, 1, 1, 12))
#             model_fit = model.fit()
#             forecast = model_fit.get_forecast(steps=periods)
#             return forecast.predicted_mean.values
        
#         port_pred = sarimax_predict(port_ret)
#         bench_pred = sarimax_predict(bench_ret)
#     elif model_type == "Prophet":
#         def prophet_predict(returns):
#             df = pd.DataFrame({'ds': returns.index, 'y': returns.values})
#             model = Prophet()
#             model.fit(df)
#             future = model.make_future_dataframe(periods=periods)
#             forecast = model.predict(future)
#             return forecast['yhat'][-periods:].values
        
#         port_pred = prophet_predict(port_ret)
#         bench_pred = prophet_predict(bench_ret)
#     elif model_type == "Auto - ARIMA":
#         def auto_arima_predict(returns):
#             model = pm.auto_arima(returns, seasonal=True, m=12)
#             forecast, conf_int = model.predict(n_periods=periods, return_conf_int=True)
#             return forecast
        
#         port_pred = auto_arima_predict(port_ret)
#         bench_pred = auto_arima_predict(bench_ret)
    
#     # 创建日期索引
#     future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=periods)
    
#     # 准备绘图数据
#     def prepare_plot_data(returns, pred, name):
#         history_df = pd.DataFrame({
#             'Date': returns.index,
#             'Return': returns,
#             'Type': f'{name} Historical Data'
#         })
        
#         prediction_df = pd.DataFrame({
#             'Date': future_dates,
#             'Return': pred,
#             'Type': f'{name} Forecast Data'
#         })
        
#         return pd.concat([history_df, prediction_df])
    
#     # 合并投资组合和基准数据
#     port_plot = prepare_plot_data(port_ret, port_pred, "Portfolio")
#     bench_plot = prepare_plot_data(bench_ret, bench_pred, "Benchmark")
#     combined_plot = pd.concat([port_plot, bench_plot])
    
#     # 绘制图表
#     fig = px.line(combined_plot, x='Date', y='Return', color='Type',
#                  title=f"Return Forecast for the Next {periods} Days ({model_type} Model)",
#                  labels={'Return': 'Daily Return', 'Date': 'Date'},
#                  color_discrete_map={
#                      'Portfolio Historical Data': '#1f77b4',
#                      'Portfolio Forecast Data': '#6495ED',  # Light blue
#                      'Benchmark Historical Data': '#FFA500', # Orange
#                      'Benchmark Forecast Data': '#FFD700'    # Gold
#                  })
    
#     # 添加零值线和当前日期线
#     fig.add_hline(y=0, line_dash="dash", line_color="gray")
#     fig.add_vline(x=end_date, line_dash="dash", line_color="gray")
    
#     # 调整布局
#     fig.update_layout(
#         hovermode="x unified",
#         legend=dict(
#             orientation="h",
#             yanchor="bottom",
#             y=1.02,
#             xanchor="right",
#             x=1
#         ),
#         yaxis_tickformat=".2%"
#     )
    
#     st.plotly_chart(fig, use_container_width=True)
    
#     # 显示预测统计信息
#     st.subheader("Forecast Statistics")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Portfolio Forecast Average Daily Return", f"{port_pred.mean()*100:.2f}%")
#         st.metric("Portfolio Forecast Volatility", f"{port_pred.std()*100:.2f}%")
#     with col2:
#         st.metric("Benchmark Forecast Average Daily Return", f"{bench_pred.mean()*100:.2f}%")
#         st.metric("Benchmark Forecast Volatility", f"{bench_pred.std()*100:.2f}%")


# In[ ]:


'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
from math import sqrt

def model_description(model_type):
    """Return detailed description of each model type"""
    descriptions = {
        "Linear Regression": {
            "principle": "Finds a linear relationship between time (independent variable) and returns (dependent variable).",
            "strengths": "Simple to implement, fast computation, easy to interpret.",
            "weaknesses": "Assumes linear relationship, cannot capture complex patterns or seasonality.",
            "best_for": "Short-term predictions when data shows clear linear trend."
        },
        "ARIMA": {
            "principle": "Autoregressive Integrated Moving Average model that combines autoregression (AR), differencing (I), and moving average (MA) components.",
            "strengths": "Can capture trends and basic patterns in time series data.",
            "weaknesses": "Requires stationary data, struggles with seasonality and complex patterns.",
            "best_for": "Non-seasonal time series with clear trends."
        },
        "SARIMAX": {
            "principle": "Seasonal ARIMA with eXogenous factors, extends ARIMA to handle seasonality.",
            "strengths": "Can model both trend and seasonality in time series data.",
            "weaknesses": "Complex parameter tuning, computationally intensive.",
            "best_for": "Time series with both trend and seasonal components."
        },
        "Exponential Smoothing": {
            "principle": "Weighted averages of past observations, with weights decaying exponentially.",
            "strengths": "Good for data with trend and/or seasonality, relatively simple.",
            "weaknesses": "Not suitable for long-term forecasts.",
            "best_for": "Short-term forecasting with trend and seasonality."
        },
        "Prophet": {
            "principle": "Additive model developed by Facebook that fits non-linear trends with yearly, weekly, and daily seasonality.",
            "strengths": "Handles missing data, outliers, and holidays well. Automatic changepoint detection.",
            "weaknesses": "Requires more data for good performance, slower than simpler models.",
            "best_for": "Time series with strong seasonal effects and multiple seasons."
        }
    }
    return descriptions.get(model_type, {})

def evaluate_model(actual, predicted):
    """Calculate evaluation metrics for model performance"""
    return {
        "RMSE": sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted)
    }

def make_predictions(train_data, test_data, model_type, periods):
    """Make predictions using the selected model type"""
    if model_type == "Linear Regression":
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Evaluate on test set
        X_test = np.arange(len(train_data), len(train_data)+len(test_data)).reshape(-1, 1)
        test_pred = model.predict(X_test)
        
        # Make future predictions
        future_X = np.arange(len(train_data)+len(test_data), 
                           len(train_data)+len(test_data)+periods).reshape(-1, 1)
        future_pred = model.predict(future_X)
        
        return test_pred, future_pred
    
    elif model_type == "ARIMA":
        model = ARIMA(train_data, order=(2, 1, 2))
        model_fit = model.fit()
        
        # Evaluate on test set
        test_pred = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
        
        # Make future predictions
        forecast = model_fit.get_forecast(steps=periods)
        future_pred = forecast.predicted_mean.values
        
        return test_pred, future_pred
    
    elif model_type == "SARIMAX":
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
        model_fit = model.fit()
        
        # Evaluate on test set
        test_pred = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
        
        # Make future predictions
        forecast = model_fit.get_forecast(steps=periods)
        future_pred = forecast.predicted_mean.values
        
        return test_pred, future_pred
    
    elif model_type == "Exponential Smoothing":
        model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=5)
        model_fit = model.fit()
        
        # Evaluate on test set
        test_pred = model_fit.forecast(len(test_data))
        
        # Make future predictions
        future_pred = model_fit.forecast(periods)
        
        return test_pred, future_pred
    
    elif model_type == "Prophet":
        # Prepare data for Prophet
        df = train_data.reset_index()
        df.columns = ['ds', 'y']
        
        model = Prophet(daily_seasonality=False)
        model.fit(df)
        
        # Create future dataframe for test and prediction
        future = pd.DataFrame({
            'ds': pd.date_range(start=train_data.index[0], 
                               periods=len(train_data)+len(test_data)+periods)
        })
        
        forecast = model.predict(future)
        
        # Extract predictions
        test_pred = forecast.iloc[len(train_data):len(train_data)+len(test_data)]['yhat'].values
        future_pred = forecast.iloc[-periods:]['yhat'].values
        
        return test_pred, future_pred

def run_all_models(port_train, port_test, bench_train, bench_test, periods):
    """Run all models and return evaluation results"""
    model_options = ["Linear Regression", "ARIMA", "SARIMAX", "Exponential Smoothing", "Prophet"]
    all_results = []
    
    for model_type in model_options:
        try:
            # Portfolio predictions
            port_test_pred, _ = make_predictions(port_train, port_test, model_type, periods)
            port_metrics = evaluate_model(port_test, port_test_pred)
            
            # Benchmark predictions
            bench_test_pred, _ = make_predictions(bench_train, bench_test, model_type, periods)
            bench_metrics = evaluate_model(bench_test, bench_test_pred)
            
            # Calculate combined score (weighted average of RMSE)
            combined_score = 0.7 * port_metrics['RMSE'] + 0.3 * bench_metrics['RMSE']
            
            all_results.append({
                'Model': model_type,
                'Portfolio RMSE': port_metrics['RMSE'],
                'Portfolio MAE': port_metrics['MAE'],
                'Benchmark RMSE': bench_metrics['RMSE'],
                'Benchmark MAE': bench_metrics['MAE'],
                'Combined Score': combined_score
            })
            
        except Exception as e:
            st.warning(f"Could not evaluate {model_type} model: {str(e)}")
            continue
    
    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        best_model = results_df.loc[results_df['Combined Score'].idxmin(), 'Model']
        return results_df, best_model
    return None, None


def return_prediction(portfolio_returns, benchmark_returns):
    """Enhanced return prediction function with multiple models and automatic model selection"""
    st.subheader("Return Prediction Analysis")
    
    # Initialize session state for model selection
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    
    # Model selection
    model_options = ["Linear Regression", "ARIMA", "SARIMAX", "Exponential Smoothing", "Prophet"]
    model_type = st.radio("Select prediction model", model_options, horizontal=True)
    
    # Show model description
    with st.expander(f"ℹ️ {model_type} Model Details"):
        desc = model_description(model_type)
        st.markdown(f"""
        **Model Principle**: {desc.get('principle', 'N/A')}  
        **Strengths**: {desc.get('strengths', 'N/A')}  
        **Weaknesses**: {desc.get('weaknesses', 'N/A')}  
        **Best For**: {desc.get('best_for', 'N/A')}
        """)
    
    periods = st.slider("Select prediction horizon (days)", 10, 90, 30)
    
    # Data preparation
    end_date = portfolio_returns.index.max()
    
    # Determine lookback period based on model
    lookback = {
        "Linear Regression": pd.DateOffset(months=3),
        "ARIMA": pd.DateOffset(months=12),
        "SARIMAX": pd.DateOffset(months=18),
        "Exponential Smoothing": pd.DateOffset(months=12),
        "Prophet": pd.DateOffset(months=24)
    }[model_type]
    
    start_date = end_date - lookback
    port_ret = portfolio_returns[(portfolio_returns.index >= start_date)].dropna()
    bench_ret = benchmark_returns[(benchmark_returns.index >= start_date)].dropna()
    
    if len(port_ret) < 10 or len(bench_ret) < 10:
        st.warning("Insufficient data (minimum 10 trading days required)")
        return
    
    # Split data into train and test sets for evaluation
    split_idx = int(len(port_ret) * 0.8)
    port_train, port_test = port_ret.iloc[:split_idx], port_ret.iloc[split_idx:]
    bench_train, bench_test = bench_ret.iloc[:split_idx], bench_ret.iloc[split_idx:]
    
    # Automatic model selection
    if st.checkbox("Run automatic model selection (may take longer)"):
        with st.spinner("Evaluating all models..."):
            model_results, best_model = run_all_models(
                port_train, port_test, 
                bench_train, bench_test, 
                periods
            )
            
            if model_results is not None:
                st.session_state.model_results = model_results
                st.session_state.best_model = best_model
                
                st.write("### Model Comparison Results")
                st.dataframe(model_results.style.highlight_min(axis=0, color='lightgreen'))
                
                st.success(f"🎯 Best model: **{best_model}**")
                
                # If the best model is different from selected one, offer to switch
                if best_model != model_type:
                    if st.button(f"Switch to {best_model} model"):
                        model_type = best_model
                        st.session_state.best_model = best_model
                        st.rerun()
    
    # Use the selected model (or best model if available)
    current_model = st.session_state.best_model if st.session_state.best_model else model_type
    
    # Suppress warnings
    warnings.filterwarnings("ignore")
    
    # Make predictions with current model
    port_test_pred, port_future_pred = make_predictions(port_train, port_test, current_model, periods)
    port_metrics = evaluate_model(port_test, port_test_pred)
    
    bench_test_pred, bench_future_pred = make_predictions(bench_train, bench_test, current_model, periods)
    bench_metrics = evaluate_model(bench_test, bench_test_pred)
    
    # Create visualization
    future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=periods)
    
    def prepare_plot_data(returns, test_pred, future_pred, name):
        # Historical data
        history_df = pd.DataFrame({
            'Date': returns.index,
            'Return': returns,
            'Type': f'{name} Historical'
        })
        
        # Test period predictions
        test_df = pd.DataFrame({
            'Date': returns.index[-len(test_pred):],
            'Return': test_pred,
            'Type': f'{name} Test Prediction'
        })
        
        # Future predictions
        future_df = pd.DataFrame({
            'Date': future_dates,
            'Return': future_pred,
            'Type': f'{name} Forecast'
        })
        
        return pd.concat([history_df, test_df, future_df])
    
    # Combine data for plotting
    port_plot = prepare_plot_data(port_ret, port_test_pred, port_future_pred, "Portfolio")
    bench_plot = prepare_plot_data(bench_ret, bench_test_pred, bench_future_pred, "Benchmark")
    combined_plot = pd.concat([port_plot, bench_plot])
    
    # Create figure
    fig = px.line(combined_plot, x='Date', y='Return', color='Type',
                 title=f"Return Forecast ({current_model} Model)",
                 labels={'Return': 'Daily Return'},
                 color_discrete_map={
                     'Portfolio Historical': '#1f77b4',
                     'Portfolio Test Prediction': '#6495ED',
                     'Portfolio Forecast': '#00008B',
                     'Benchmark Historical': '#FFA500',
                     'Benchmark Test Prediction': '#FFD700',
                     'Benchmark Forecast': '#8B0000'
                 })
    
    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=end_date, line_dash="dash", line_color="gray")
    
    # Highlight test period
    if len(port_test) > 0:
        fig.add_vrect(
            x0=port_test.index[0], x1=port_test.index[-1],
            fillcolor="lightgray", opacity=0.2,
            layer="below", line_width=0
        )
    
    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_tickformat=".2%"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio RMSE", f"{port_metrics['RMSE']:.4f}")
        st.metric("Portfolio MAE", f"{port_metrics['MAE']:.4f}")
    with col2:
        st.metric("Benchmark RMSE", f"{bench_metrics['RMSE']:.4f}")
        st.metric("Benchmark MAE", f"{bench_metrics['MAE']:.4f}")
    
    # Forecast statistics
    st.subheader("Forecast Statistics (Next {} Days)".format(periods))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio Forecast Avg Return", f"{port_future_pred.mean()*100:.2f}%")
        st.metric("Portfolio Forecast Volatility", f"{port_future_pred.std()*100:.2f}%")
    with col2:
        st.metric("Benchmark Forecast Avg Return", f"{bench_future_pred.mean()*100:.2f}%")
        st.metric("Benchmark Forecast Volatility", f"{bench_future_pred.std()*100:.2f}%")
'''


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import streamlit as st
import math
import warnings
from pandas.tseries.offsets import DateOffset
import plotly.express as px

"""Calculate evaluation metrics for model performance"""
def evaluate_model(actual, predicted):
    """Calculate evaluation metrics for model performance"""
    return {
        "RMSE": math.sqrt(mean_squared_error(actual, predicted)),
        "MAE": mean_absolute_error(actual, predicted)
    }

"""Return detailed description of each model type"""
def model_description(model_type):
    """Return detailed description of each model type"""
    descriptions = {
        "Linear Regression": {
            "principle": "Finds a linear relationship between time (independent variable) and returns (dependent variable).",
            "strengths": "Simple to implement, fast computation, easy to interpret.",
            "weaknesses": "Assumes linear relationship, cannot capture complex patterns or seasonality.",
            "best_for": "Short-term predictions when data shows clear linear trend."
        },
        "ARIMA": {
            "principle": "Autoregressive Integrated Moving Average model that combines autoregression (AR), differencing (I), and moving average (MA) components.",
            "strengths": "Can capture trends and basic patterns in time series data.",
            "weaknesses": "Requires stationary data, struggles with seasonality and complex patterns.",
            "best_for": "Non-seasonal time series with clear trends."
        },
        "SARIMAX": {
            "principle": "Seasonal ARIMA with eXogenous factors, extends ARIMA to handle seasonality.",
            "strengths": "Can model both trend and seasonality in time series data.",
            "weaknesses": "Complex parameter tuning, computationally intensive.",
            "best_for": "Time series with both trend and seasonal components."
        },
        "Exponential Smoothing": {
            "principle": "Weighted averages of past observations, with weights decaying exponentially.",
            "strengths": "Good for data with trend and/or seasonality, relatively simple.",
            "weaknesses": "Not suitable for long-term forecasts.",
            "best_for": "Short-term forecasting with trend and seasonality."
        },
        "Prophet": {
            "principle": "Additive model developed by Facebook that fits non-linear trends with yearly, weekly, and daily seasonality.",
            "strengths": "Handles missing data, outliers, and holidays well. Automatic changepoint detection.",
            "weaknesses": "Requires more data for good performance, slower than simpler models.",
            "best_for": "Time series with strong seasonal effects and multiple seasons."
        }
    }
    return descriptions.get(model_type, {})

"""Make predictions using the selected model type"""
def make_predictions(train_data, test_data, model_type, periods):
    """Make predictions using the selected model type"""
    if model_type == "Linear Regression":
        X_train = np.arange(len(train_data)).reshape(-1, 1)
        y_train = train_data.values
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Evaluate on test set
        X_test = np.arange(len(train_data), len(train_data)+len(test_data)).reshape(-1, 1)
        test_pred = model.predict(X_test)

        # Make future predictions
        future_X = np.arange(len(train_data)+len(test_data), 
                           len(train_data)+len(test_data)+periods).reshape(-1, 1)
        future_pred = model.predict(future_X)

        return test_pred, future_pred

    elif model_type == "ARIMA":
        model = ARIMA(train_data, order=(2, 1, 2))
        model_fit = model.fit()

        # Evaluate on test set
        test_pred = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

        # Make future predictions
        forecast = model_fit.get_forecast(steps=periods)
        future_pred = forecast.predicted_mean.values

        return test_pred, future_pred

    elif model_type == "SARIMAX":
        model = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(1, 1, 1, 5))
        model_fit = model.fit()

        # Evaluate on test set
        test_pred = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)

        # Make future predictions
        forecast = model_fit.get_forecast(steps=periods)
        future_pred = forecast.predicted_mean.values

        return test_pred, future_pred

    elif model_type == "Exponential Smoothing":
        model = ExponentialSmoothing(train_data, trend='add', seasonal='add', seasonal_periods=5)
        model_fit = model.fit()

        # Evaluate on test set
        test_pred = model_fit.forecast(len(test_data))

        # Make future predictions
        future_pred = model_fit.forecast(periods)

        return test_pred, future_pred

    elif model_type == "Prophet":
        # Prepare data for Prophet
        df = train_data.reset_index()
        df.columns = ['ds', 'y']

        model = Prophet(daily_seasonality=False)
        model.fit(df)

        # Create future dataframe for test and prediction
        future = pd.DataFrame({
            'ds': pd.date_range(start=train_data.index[0], 
                               periods=len(train_data)+len(test_data)+periods)
        })

        forecast = model.predict(future)

        # Extract predictions
        test_pred = forecast.iloc[len(train_data):len(train_data)+len(test_data)]['yhat'].values
        future_pred = forecast.iloc[-periods:]['yhat'].values

        return test_pred, future_pred

"""Run all models and return evaluation results"""
def run_all_models(port_train, port_test, bench_train, bench_test, periods):
    """Run all models and return evaluation results"""
    model_options = ["Linear Regression", "ARIMA", "SARIMAX", "Exponential Smoothing", "Prophet"]
    all_results = []

    for model_type in model_options:
        try:
            # Portfolio predictions
            port_test_pred, _ = make_predictions(port_train, port_test, model_type, periods)
            port_metrics = evaluate_model(port_test, port_test_pred)

            # Benchmark predictions
            bench_test_pred, _ = make_predictions(bench_train, bench_test, model_type, periods)
            bench_metrics = evaluate_model(bench_test, bench_test_pred)

            # Calculate combined score (weighted average of RMSE)
            combined_score = 0.7 * port_metrics['RMSE'] + 0.3 * bench_metrics['RMSE']

            all_results.append({
                'Model': model_type,
                'Portfolio RMSE': port_metrics['RMSE'],
                'Portfolio MAE': port_metrics['MAE'],
                'Benchmark RMSE': bench_metrics['RMSE'],
                'Benchmark MAE': bench_metrics['MAE'],
                'Combined Score': combined_score
            })

        except Exception as e:
            st.warning(f"Could not evaluate {model_type} model: {str(e)}")
            continue

    results_df = pd.DataFrame(all_results)
    if not results_df.empty:
        best_model = results_df.loc[results_df['Combined Score'].idxmin(), 'Model']
        return results_df, best_model
    return None, None

def prepare_plot_data(returns, test_pred, future_pred, name, end_date, periods):
    future_dates = pd.date_range(start=end_date + pd.Timedelta(days=1), periods=periods)
    # Historical data
    history_df = pd.DataFrame({
        'Date': returns.index,
        'Return': returns,
        'Type': f'{name} Historical'
    })

    # Test period predictions
    test_df = pd.DataFrame({
        'Date': returns.index[-len(test_pred):],
        'Return': test_pred,
        'Type': f'{name} Test Prediction'
    })

    # Future predictions
    future_df = pd.DataFrame({
        'Date': future_dates,
        'Return': future_pred,
        'Type': f'{name} Forecast'
    })

    return pd.concat([history_df, test_df, future_df])

"""Enhanced return prediction function with multiple models and automatic model selection"""
def return_prediction(portfolio_returns, benchmark_returns):
    """Enhanced return prediction function with multiple models and automatic model selection"""
    st.subheader("Return Prediction Analysis")

    selection_method = st.radio("Model selection method", ["Manually select model", "Automatically select best model"])

    # Initialize session state for model selection
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None

    # Model selection
    model_options = ["Linear Regression", "ARIMA", "SARIMAX", "Exponential Smoothing", "Prophet"]
    if selection_method == "Manually select model":
        model_type = st.radio("Select prediction model", model_options, horizontal=True)
        # Show model description
        with st.expander(f"ℹ️ {model_type} Model Details"):
            desc = model_description(model_type)
            st.markdown(f"""
            **Model Principle**: {desc.get('principle', 'N/A')}  
            **Strengths**: {desc.get('strengths', 'N/A')}  
            **Weaknesses**: {desc.get('weaknesses', 'N/A')}  
            **Best For**: {desc.get('best_for', 'N/A')}
            """)
    elif selection_method == "Automatically select best model":
        model_type = None

    periods = st.slider("Select prediction horizon (days)", 10, 90, 30)

    # Data preparation
    end_date = portfolio_returns.index.max()

    # Determine lookback period based on model
    lookback = {
        "Linear Regression": pd.DateOffset(months=3),
        "ARIMA": pd.DateOffset(months=12),
        "SARIMAX": pd.DateOffset(months=18),
        "Exponential Smoothing": pd.DateOffset(months=12),
        "Prophet": pd.DateOffset(months=24)
    }

    if model_type:
        start_date = end_date - lookback[model_type]
    else:
        start_date = end_date - lookback["Prophet"]

    port_ret = portfolio_returns[(portfolio_returns.index >= start_date)].dropna()
    bench_ret = benchmark_returns[(benchmark_returns.index >= start_date)].dropna()

    if len(port_ret) < 10 or len(bench_ret) < 10:
        st.warning("Insufficient data (minimum 10 trading days required)")
        return

    # Split data into train and test sets for evaluation
    split_idx = int(len(port_ret) * 0.8)
    port_train, port_test = port_ret.iloc[:split_idx], port_ret.iloc[split_idx:]
    bench_train, bench_test = bench_ret.iloc[:split_idx], bench_ret.iloc[split_idx:]

    # Automatic model selection
    if selection_method == "Automatically select best model":
        with st.spinner("Evaluating all models..."):
            model_results, best_model = run_all_models(
                port_train, port_test, 
                bench_train, bench_test, 
                periods
            )

            if model_results is not None:
                st.session_state.model_results = model_results
                st.session_state.best_model = best_model

                st.write("### Model Comparison Results")
                st.dataframe(model_results.style.highlight_min(axis=0, color='lightgreen'))

                st.success(f"🎯 Best model: **{best_model}**")
                model_type = best_model
    else:
        st.session_state.best_model = None
        st.session_state.model_results = None

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Make predictions with current model
    port_test_pred, port_future_pred = make_predictions(port_train, port_test, model_type, periods)
    port_metrics = evaluate_model(port_test, port_test_pred)

    bench_test_pred, bench_future_pred = make_predictions(bench_train, bench_test, model_type, periods)
    bench_metrics = evaluate_model(bench_test, bench_test_pred)

    # Create visualization
    port_plot = prepare_plot_data(port_ret, port_test_pred, port_future_pred, "Portfolio", end_date, periods)
    bench_plot = prepare_plot_data(bench_ret, bench_test_pred, bench_future_pred, "Benchmark", end_date, periods)
    combined_plot = pd.concat([port_plot, bench_plot])

    # Create figure
    fig = px.line(combined_plot, x='Date', y='Return', color='Type',
                 title=f"Return Forecast ({model_type} Model)",
                 labels={'Return': 'Daily Return'},
                 color_discrete_map={
                     'Portfolio Historical': '#1f77b4',
                     'Portfolio Test Prediction': '#6495ED',
                     'Portfolio Forecast': '#00008B',
                     'Benchmark Historical': '#FFA500',
                     'Benchmark Test Prediction': '#FFD700',
                     'Benchmark Forecast': '#8B0000'
                 })

    # Add reference lines
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.add_vline(x=end_date, line_dash="dash", line_color="gray")

    # Highlight test period
    if len(port_test) > 0:
        fig.add_vrect(
            x0=port_test.index[0], x1=port_test.index[-1],
            fillcolor="lightgray", opacity=0.2,
            layer="below", line_width=0
        )

    fig.update_layout(
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis_tickformat=".2%"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Display metrics
    st.subheader("Model Performance Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio RMSE", f"{port_metrics['RMSE']:.4f}")
        st.metric("Portfolio MAE", f"{port_metrics['MAE']:.4f}")
    with col2:
        st.metric("Benchmark RMSE", f"{bench_metrics['RMSE']:.4f}")
        st.metric("Benchmark MAE", f"{bench_metrics['MAE']:.4f}")

    # Forecast statistics
    st.subheader("Forecast Statistics (Next {} Days)".format(periods))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Portfolio Forecast Avg Return", f"{port_future_pred.mean()*100:.2f}%")
        st.metric("Portfolio Forecast Volatility", f"{port_future_pred.std()*100:.2f}%")
    with col2:
        st.metric("Benchmark Forecast Avg Return", f"{bench_future_pred.mean()*100:.2f}%")
        st.metric("Benchmark Forecast Volatility", f"{bench_future_pred.std()*100:.2f}%")

