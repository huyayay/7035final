# -*- coding: utf-8 -*-
""" 
Created on Sun Apr 13 12:17:46 2025

@author: 24802
"""
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from pandas.tseries.offsets import DateOffset
from sklearn.linear_model import LinearRegression
import uuid

def get_selected_period():
    """
    获取用户选择的回报分析时间段
    :return: 用户选择的时间段
    """
  
    time_periods = ['1Y', '3Y', '5Y', 'All']
    #unique_key = str(uuid.uuid4())  # 生成唯一的 UUID 作为 key
    return st.selectbox(" ", time_periods, key='unique_key')


def filter_returns(portfolio_returns, benchmark_returns, selected_period):
    """
    根据用户选择的时间段筛选投资组合和基准的收益率数据
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :param selected_period: 用户选择的时间段
    :return: 筛选后的投资组合和基准收益率
    """
    end_date = portfolio_returns.index.max()
    if selected_period == 'All':
        start_date = portfolio_returns.index.min()
    else:
        years = int(selected_period[0])
        start_date = end_date - DateOffset(years=years)

    portfolio_returns = portfolio_returns[
        (portfolio_returns.index >= start_date) & (portfolio_returns.index <= end_date)]
    benchmark_returns = benchmark_returns[
        (benchmark_returns.index >= start_date) & (benchmark_returns.index <= end_date)]
    return portfolio_returns, benchmark_returns


def calculate_annualized_return(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的年度化回报
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 年度化回报指标结果
    """
    annualized_return_port = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
    annualized_return_bench = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
    return annualized_return_port, annualized_return_bench


def calculate_sharpe_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的夏普比率
    :param annualized_return_port: 投资组合年度化回报
    :param annualized_return_bench: 基准年度化回报
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 夏普比率指标结果
    """
    risk_free_rate = 0.02
    annualized_std_port = portfolio_returns.std() * np.sqrt(252)
    annualized_std_bench = benchmark_returns.std() * np.sqrt(252)
    if annualized_std_port != 0 and annualized_std_bench != 0:
        sharpe_ratio_port = (annualized_return_port - risk_free_rate) / annualized_std_port
        sharpe_ratio_bench = (annualized_return_bench - risk_free_rate) / annualized_std_bench
        return [f"{sharpe_ratio_port:.2f}", f"{sharpe_ratio_bench:.2f}"]
    return ["无法计算（标准差为 0）", "无法计算（标准差为 0）"]


def calculate_max_drawdown(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的最大回撤
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 最大回撤指标结果
    """
    cum_returns_port = (1 + portfolio_returns).cumprod()
    running_max_port = cum_returns_port.cummax()
    drawdown_port = cum_returns_port / running_max_port - 1
    max_drawdown_port = drawdown_port.min()

    cum_returns_bench = (1 + benchmark_returns).cumprod()
    running_max_bench = cum_returns_bench.cummax()
    drawdown_bench = cum_returns_bench / running_max_bench - 1
    max_drawdown_bench = drawdown_bench.min()
    return max_drawdown_port, max_drawdown_bench


def calculate_arithmetic_mean(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的算术平均月收益率和年化收益率
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 算术平均月收益率和年化收益率指标结果
    """
    monthly_return_port = portfolio_returns.groupby(portfolio_returns.index.strftime('%Y-%m')).mean()
    arithmetic_mean_monthly_port = monthly_return_port.mean()
    monthly_return_bench = benchmark_returns.groupby(benchmark_returns.index.strftime('%Y-%m')).mean()
    arithmetic_mean_monthly_bench = monthly_return_bench.mean()

    arithmetic_mean_annualized_port = arithmetic_mean_monthly_port * 12
    arithmetic_mean_annualized_bench = arithmetic_mean_monthly_bench * 12

    return [
        [f"{arithmetic_mean_monthly_port * 100:.2f}%", f"{arithmetic_mean_monthly_bench * 100:.2f}%"],
        [f"{arithmetic_mean_annualized_port * 100:.2f}%", f"{arithmetic_mean_annualized_bench * 100:.2f}%"]
    ]


def calculate_geometric_mean(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的几何平均月收益率和年化收益率
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 几何平均月收益率和年化收益率指标结果
    """
    monthly_return_port = portfolio_returns.groupby(portfolio_returns.index.strftime('%Y-%m')).mean()
    geometric_mean_monthly_port = (1 + monthly_return_port).prod() ** (1 / len(monthly_return_port)) - 1
    monthly_return_bench = benchmark_returns.groupby(benchmark_returns.index.strftime('%Y-%m')).mean()
    geometric_mean_monthly_bench = (1 + monthly_return_bench).prod() ** (1 / len(monthly_return_bench)) - 1

    geometric_mean_annualized_port = (1 + geometric_mean_monthly_port) ** 12 - 1
    geometric_mean_annualized_bench = (1 + geometric_mean_monthly_bench) ** 12 - 1

    return [
        [f"{geometric_mean_monthly_port * 100:.2f}%", f"{geometric_mean_monthly_bench * 100:.2f}%"],
        [f"{geometric_mean_annualized_port * 100:.2f}%", f"{geometric_mean_annualized_bench * 100:.2f}%"]
    ]


def calculate_std_deviation(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的月收益率标准差和年化收益率标准差
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 月收益率标准差和年化收益率标准差指标结果
    """
    monthly_return_port = portfolio_returns.groupby(portfolio_returns.index.strftime('%Y-%m')).mean()
    std_dev_monthly_port = monthly_return_port.std()
    monthly_return_bench = benchmark_returns.groupby(benchmark_returns.index.strftime('%Y-%m')).mean()
    std_dev_monthly_bench = monthly_return_bench.std()

    std_dev_annualized_port = std_dev_monthly_port * np.sqrt(12)
    std_dev_annualized_bench = std_dev_monthly_bench * np.sqrt(12)

    return [
        [f"{std_dev_monthly_port * 100:.2f}%", f"{std_dev_monthly_bench * 100:.2f}%"],
        [f"{std_dev_annualized_port * 100:.2f}%", f"{std_dev_annualized_bench * 100:.2f}%"]
    ]


def calculate_downside_deviation(portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的月下行偏差
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 月下行偏差指标结果
    """
    monthly_return_port = portfolio_returns.groupby(portfolio_returns.index.strftime('%Y-%m')).mean()
    downside_returns_port = monthly_return_port[monthly_return_port < 0]
    if len(downside_returns_port) > 0:
        downside_deviation_monthly_port = downside_returns_port.std()
        downside_deviation_monthly_port_str = f"{downside_deviation_monthly_port * 100:.2f}%"
    else:
        downside_deviation_monthly_port_str = "无法计算（无负回报数据）"

    monthly_return_bench = benchmark_returns.groupby(benchmark_returns.index.strftime('%Y-%m')).mean()
    downside_returns_bench = monthly_return_bench[monthly_return_bench < 0]
    if len(downside_returns_bench) > 0:
        downside_deviation_monthly_bench = downside_returns_bench.std()
        downside_deviation_monthly_bench_str = f"{downside_deviation_monthly_bench * 100:.2f}%"
    else:
        downside_deviation_monthly_bench_str = "无法计算（无负回报数据）"

    return [downside_deviation_monthly_port_str, downside_deviation_monthly_bench_str]


def calculate_benchmark_correlation(portfolio_returns, benchmark_returns):
    """
    计算投资组合与基准的相关性
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 与基准相关性指标结果
    """
    if len(portfolio_returns) == len(benchmark_returns):
        correlation_port_bench = portfolio_returns.corr(benchmark_returns)
        correlation_bench_bench = benchmark_returns.corr(benchmark_returns)
        return [f"{correlation_port_bench:.2f}", f"{correlation_bench_bench:.2f}"]
    return ["无法计算（数据长度不一致）", "无法计算（数据长度不一致）"]





def calculate_sortino_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的索提诺比率
    :param annualized_return_port: 投资组合年度化回报
    :param annualized_return_bench: 基准年度化回报
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 索提诺比率指标结果
    """
    risk_free_rate = 0.02
    downside_std_port = np.sqrt(((portfolio_returns[portfolio_returns < 0]) ** 2).sum() / len(portfolio_returns))
    if downside_std_port != 0:
        sortino_ratio_port = (annualized_return_port - risk_free_rate) / downside_std_port
        sortino_ratio_port_str = f"{sortino_ratio_port:.2f}"
    else:
        sortino_ratio_port_str = "无法计算（下行标准差为 0）"

    downside_std_bench = np.sqrt(((benchmark_returns[benchmark_returns < 0]) ** 2).sum() / len(benchmark_returns))
    if downside_std_bench != 0:
        sortino_ratio_bench = (annualized_return_bench - risk_free_rate) / downside_std_bench
        sortino_ratio_bench_str = f"{sortino_ratio_bench:.2f}"
    else:
        sortino_ratio_bench_str = "无法计算（下行标准差为 0）"

    return [sortino_ratio_port_str, sortino_ratio_bench_str]


def calculate_treynor_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns):
    """
    计算投资组合和基准的特雷诺比率
    :param annualized_return_port: 投资组合年度化回报
    :param annualized_return_bench: 基准年度化回报
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 特雷诺比率指标结果
    """
    risk_free_rate = 0.02
    annualized_std_port = portfolio_returns.std() * np.sqrt(252)
    annualized_std_bench = benchmark_returns.std() * np.sqrt(252)
    if annualized_std_port != 0:
        beta_port = annualized_std_port / annualized_std_bench
        treynor_ratio_port = (annualized_return_port - risk_free_rate) / beta_port
        treynor_ratio_port_str = f"{treynor_ratio_port * 100:.2f}%"
    else:
        treynor_ratio_port_str = "无法计算（标准差为 0）"

    if annualized_std_bench != 0:
        beta_bench = 1
        treynor_ratio_bench = (annualized_return_bench - risk_free_rate) / beta_bench
        treynor_ratio_bench_str = f"{treynor_ratio_bench * 100:.2f}%"
    else:
        treynor_ratio_bench_str = "无法计算（标准差为 0）"

    return [treynor_ratio_port_str, treynor_ratio_bench_str]


def calculate_calmar_ratio(annualized_return_port, annualized_return_bench, max_drawdown_port, max_drawdown_bench):
    """
    计算投资组合和基准的卡玛比率
    :param annualized_return_port: 投资组合年度化回报
    :param annualized_return_bench: 基准年度化回报
    :param max_drawdown_port: 投资组合最大回撤
    :param max_drawdown_bench: 基准最大回撤
    :return: 卡玛比率指标结果
    """
    if max_drawdown_port != 0:
        calmar_ratio_port = annualized_return_port / (-max_drawdown_port)
        calmar_ratio_port_str = f"{calmar_ratio_port:.2f}"
    else:
        calmar_ratio_port_str = "无法计算（最大回撤为 0）"

    if max_drawdown_bench != 0:
        calmar_ratio_bench = annualized_return_bench / (-max_drawdown_bench)
        calmar_ratio_bench_str = f"{calmar_ratio_bench:.2f}"
    else:
        calmar_ratio_bench_str = "无法计算（最大回撤为 0）"

    return [calmar_ratio_port_str, calmar_ratio_bench_str]

def performance_metrics(portfolio_returns, benchmark_returns):
    st.subheader("Performance Metrics")
    st.write("Calculate and display key performance indicators")

    selected_period = get_selected_period()
    portfolio_returns, benchmark_returns = filter_returns(portfolio_returns, benchmark_returns, selected_period)

    annualized_return_port, annualized_return_bench = calculate_annualized_return(portfolio_returns, benchmark_returns)
    max_drawdown_port, max_drawdown_bench = calculate_max_drawdown(portfolio_returns, benchmark_returns)

    basic_metrics = {
        "Annualized Return": [f"{annualized_return_port * 100:.2f}%", f"{annualized_return_bench * 100:.2f}%"],
        "Sharpe Ratio": calculate_sharpe_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns),
        "Max Drawdown": [f"{max_drawdown_port * 100:.2f}%", f"{max_drawdown_bench * 100:.2f}%"],
        "Arithmetic Mean Monthly Return": calculate_arithmetic_mean(portfolio_returns, benchmark_returns)[0],
        "Arithmetic Mean Annualized Return": calculate_arithmetic_mean(portfolio_returns, benchmark_returns)[1],
        "Geometric Mean Monthly Return": calculate_geometric_mean(portfolio_returns, benchmark_returns)[0],
        "Geometric Mean Annualized Return": calculate_geometric_mean(portfolio_returns, benchmark_returns)[1],
        "Monthly Return Standard Deviation": calculate_std_deviation(portfolio_returns, benchmark_returns)[0],
        "Annualized Return Standard Deviation": calculate_std_deviation(portfolio_returns, benchmark_returns)[1],
        "Monthly Downside Deviation": calculate_downside_deviation(portfolio_returns, benchmark_returns),
        "Correlation with Benchmark": calculate_benchmark_correlation(portfolio_returns, benchmark_returns),
        
        "Sortino Ratio": calculate_sortino_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns),
        "Treynor Ratio": calculate_treynor_ratio(annualized_return_port, annualized_return_bench, portfolio_returns, benchmark_returns),
        "Calmar Ratio": calculate_calmar_ratio(annualized_return_port, annualized_return_bench, max_drawdown_port, max_drawdown_bench)
    }

    basic_df = pd.DataFrame({
        "Metrics": list(basic_metrics.keys()),
        "Portfolio": [x[0] for x in basic_metrics.values()],
        "Benchmark": [x[1] for x in basic_metrics.values()]
    }).set_index("Metrics")

    st.table(basic_df.style.hide(axis="index"))
    # 返回 DataFrame 而不是显示表格
    return basic_df
    
def calculate_basic_metrics(portfolio_returns, benchmark_returns):
    """
    计算基本性能指标
    :param portfolio_returns: 投资组合收益率
    :param benchmark_returns: 基准收益率
    :return: 基本性能指标字典
    """
    basic_metrics = {}
    if len(portfolio_returns) > 0 and len(benchmark_returns) > 0:
        annualized_return_port = (1 + portfolio_returns).prod() ** (252 / len(portfolio_returns)) - 1
        annualized_return_bench = (1 + benchmark_returns).prod() ** (252 / len(benchmark_returns)) - 1
        basic_metrics["Annualized Return"] = [f"{annualized_return_port * 100:.2f}%", f"{annualized_return_bench * 100:.2f}%"]

        risk_free_rate = 0.02
        annualized_std_port = portfolio_returns.std() * np.sqrt(252)
        annualized_std_bench = benchmark_returns.std() * np.sqrt(252)
        if annualized_std_port != 0 and annualized_std_bench != 0:
            sharpe_ratio_port = (annualized_return_port - risk_free_rate) / annualized_std_port
            sharpe_ratio_bench = (annualized_return_bench - risk_free_rate) / annualized_std_bench
            basic_metrics["Sharpe Ratio"] = [f"{sharpe_ratio_port:.2f}", f"{sharpe_ratio_bench:.2f}"]
        else:
            basic_metrics["Sharpe Ratio"] = ["Cannot be calculated (standard deviation is 0)",
                                             "Cannot be calculated (standard deviation is 0)"]

        cum_returns_port = (1 + portfolio_returns).cumprod()
        running_max_port = cum_returns_port.cummax()
        drawdown_port = cum_returns_port / running_max_port - 1
        max_drawdown_port = drawdown_port.min()

        cum_returns_bench = (1 + benchmark_returns).cumprod()
        running_max_bench = cum_returns_bench.cummax()
        drawdown_bench = cum_returns_bench / running_max_bench - 1
        max_drawdown_bench = drawdown_bench.min()

        basic_metrics["Maximum Drawdown"] = [f"{max_drawdown_port * 100:.2f}%", f"{max_drawdown_bench * 100:.2f}%"]
    return basic_metrics


def calculate_confidence_metrics(portfolio_returns):
    """
    计算置信度相关指标（VaR 和 CVaR）
    :param portfolio_returns: 投资组合收益率
    :return: 置信度相关指标数据框
    """
    if len(portfolio_returns) > 0:
        conf_levels = [0.95, 0.99]
        var_metrics = []
        cvar_metrics = []
        for conf_level in conf_levels:
            var_port = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            cvar_port = portfolio_returns[portfolio_returns <= var_port].mean()
            var_metrics.append(f"{var_port * 100:.2f}%")
            cvar_metrics.append(f"{cvar_port * 100:.2f}%")

        confidence_df = pd.DataFrame({
            "Confidence Level": [f"{conf * 100:.0f}%" for conf in conf_levels],
            "VaR": var_metrics,
            "CVaR": cvar_metrics
        })
        return confidence_df
    return None
 
def plot_annual_returns(portfolio_returns, benchmark_returns):
    """
    绘制年度收益率柱状图
    :param portfolio_returns: 投资组合收益率（需为DatetimeIndex）
    :param benchmark_returns: 基准收益率（需为DatetimeIndex）
    """
    # 确保索引为DatetimeIndex
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)
    
    # 计算年化收益率（保留数值类型）
    portfolio_annual = portfolio_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)
    benchmark_annual = benchmark_returns.resample('Y').apply(lambda x: (1 + x).prod() - 1)

    # 创建DataFrame时直接保留数值
    annual_returns_df = pd.DataFrame({
        'Year': portfolio_annual.index.year,
        'Portfolio': portfolio_annual.values,
        'Benchmark': benchmark_annual.values
    })

    # 转换数据格式为长表（无需转换百分比符号）
    annual_returns_df = pd.melt(
        annual_returns_df, 
        id_vars=['Year'], 
        var_name='Type', 
        value_name='Return'
    )

    # 绘图（y轴使用数值，通过Plotly格式化显示百分比）
    fig = px.bar(
        annual_returns_df, 
        x='Year', 
        y='Return', 
        color='Type', 
        barmode='group',
        title='Annual Returns',
        labels={'Return': 'Annual Return'},
        color_discrete_map={
            'Portfolio': 'rgb(0, 0, 139)',  # 深蓝色
            'Benchmark': 'rgb(50, 205, 50)' # 亮绿色
        }
    )

    # 格式化y轴显示为百分比
    fig.update_layout(
        yaxis_tickformat='.2%',  # 显示两位小数的百分比
        font=dict(color='black'),
        plot_bgcolor='white',
        xaxis=dict(
            type='category',  # 确保年份作为分类数据
            showgrid=True, 
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True, 
            gridcolor='lightgray'
        )
    )

    st.plotly_chart(fig)
    