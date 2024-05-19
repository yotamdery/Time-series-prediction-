###################################################
# THIS SCRIPT IS TO CONTAIN VISUALIZATION FUNCTIONS
###################################################

# Imports
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import acf, pacf
from pandas import DataFrame


def plot_feature_over_time(df: DataFrame, feature: str) -> any:
    # Plot revenue, subscribers or marketing over time
    fig_trend = px.line(df, x='month', y=feature, title=f'Monthly {feature} Over Time')
    # Update layout for clarity
    fig_trend.update_layout(
                      width=1200,
                      height=600,
                      title_x=0.5)  # Center the title
    fig_trend.show()

def plot_combined_trends(df: DataFrame) -> any:
    fig_combined = go.Figure()
    for feature in df.select_dtypes(include='number'): 
        fig_combined.add_trace(go.Scatter(x=df.index, y=df[feature], mode='lines', name=feature.capitalize()))

    fig_combined.update_layout(title='Revenue, Subscriptions, and Marketing Spend Over Time')
    fig_combined.show()

def plot_corr(correlation_matrix: DataFrame) -> any:
    # Create a heatmap using Plotly
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",  # Color scale for better distinction
        labels=dict(x="Feature", y="Feature", color="Correlation"),
        x=correlation_matrix.columns,
        y=correlation_matrix.columns)

    # Update layout for clarity
    fig.update_layout(title="Correlation Matrix of Numerical Variables",
                      xaxis_title="Features",
                      yaxis_title="Features",
                      width=600,
                      height=500,
                      title_x=0.5)  # Center the title

    # Show the plot
    fig.show()

def box_plot(df: DataFrame, feature: str) -> any :
    # Box plot for Revenue
    fig_box = px.box(df, y='revenue', title=f'{feature} Box Plot')
    fig_box.update_layout(width=900, height=700)
    fig_box.show()

def plot_seasonal_decomposition(result: DataFrame, df: DataFrame, feature: str) -> any:
    # Create subplots
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                        subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))

    # Observed plot
    fig.add_trace(go.Scatter(x=df.index, y=result.observed, mode='lines', name='Observed'), row=1, col=1)
    # Trend plot
    fig.add_trace(go.Scatter(x=df.index, y=result.trend, mode='lines', name='Trend'), row=2, col=1)
    # Seasonal plot
    fig.add_trace(go.Scatter(x=df.index, y=result.seasonal, mode='lines+markers', name='Seasonal'), row=3, col=1)
    # Residual plot
    fig.add_trace(go.Scatter(x=df.index, y=result.resid, mode='lines+markers', name='Residual'), row=4, col=1)

    # Update layout
    fig.update_layout(height=800, width=1000, title_text=f"Seasonal Decomposition of {feature}",
                    showlegend=False)

    # Update x-axis labels
    fig.update_xaxes(tickangle=45)
    # Show plot
    fig.show()

def plot_lag_plots(df: DataFrame, feature: str, lag: int) -> any:


    # Function to create lagged data for a given feature
    def create_lagged_data(df, feature, lag):
        lagged_df = df.copy()
        lagged_df[f'{feature}_lag'] = lagged_df[feature].shift(lag)
        return lagged_df.dropna()

    # Data to plot
    lagged_data = create_lagged_data(df, feature, lag)

    # Create lag plot using Plotly
    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=lagged_data[feature], 
        y=lagged_data[f'{feature}_lag'], 
        mode='markers',
        marker=dict(color='orange', size=8),
        name='Lag Plot'
    ))

    # Update layout to match Matplotlib style
    fig.update_layout(
        title=f'Lag Plot for {feature.capitalize()}',
        xaxis_title='y(t)',
        yaxis_title='y(t + 1)',
        plot_bgcolor='rgba(230, 230, 230, 0.5)',
        width=800,
        height=600,
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="black"
        ),
        xaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinecolor='lightgrey'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgrey',
            zeroline=True,
            zerolinecolor='lightgrey'
        )
    )
    fig.show()

def plot_mean_std_plots(df: DataFrame, window_size: int) -> any:
    # Calculate rolling mean and standard deviation
    rolling_stats = pd.DataFrame()
    for feature in df.select_dtypes(include='number'):
        rolling_stats[f'{feature}_mean'] = df[feature].rolling(window=window_size).mean()
        rolling_stats[f'{feature}_std'] = df[feature].rolling(window=window_size).std()

    # Plotting rolling statistics for each feature using Plotly
    for feature in df.select_dtypes(include='number'):
        fig = go.Figure()
        
        # Add rolling mean to the plot
        fig.add_trace(go.Scatter(x=rolling_stats.index, y=rolling_stats[f'{feature}_mean'],
                                mode='lines', name=f'Rolling Mean - {feature.capitalize()}'))
        
        # Add rolling standard deviation to the plot
        fig.add_trace(go.Scatter(x=rolling_stats.index, y=rolling_stats[f'{feature}_std'],
                                mode='lines', name=f'Rolling Std Dev - {feature.capitalize()}'))
        
        # Update layout
        fig.update_layout(title=f'Rolling Mean and Standard Deviation of {feature.capitalize()}',
                        xaxis_title='Month',
                        yaxis_title=feature.capitalize(),
                        height= 400,
                        width= 1400, 
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255, 255, 255, 0)', bordercolor='rgba(255, 255, 255, 0)'))
        
        # Show plot
        fig.show()

def plot_acf_plots(df: DataFrame, feature: str) -> any:
    # Calculate ACF values for the 'spend' column
    acf_values, confint = acf(df[feature], alpha=0.05)

    # Create the ACF plot
    fig = go.Figure()

    # Add bars for ACF values
    fig.add_trace(go.Bar(
        x=np.arange(len(acf_values)),
        y=acf_values,
        name='ACF'
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)),
        y=confint[:, 0] - acf_values,
        fill=None,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.3)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(acf_values)),
        y=confint[:, 1] - acf_values,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.3)'),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Autocorrelation Function (ACF) Plot for feature {feature.capitalize()}',
        xaxis_title='Lag',
        yaxis_title='Autocorrelation',
        template='plotly_white'
    )

    fig.show()

def plot_pacf_plots(df: DataFrame, feature: str) -> any:
    # Calculate PACF values for the 'spend' column
    pacf_values, confint = pacf(df[feature], alpha=0.05)

    # Create the PACF plot
    fig = go.Figure()

    # Add bars for PACF values
    fig.add_trace(go.Bar(
        x=np.arange(len(pacf_values)),
        y=pacf_values,
        name='PACF'
    ))

    # Add confidence intervals
    fig.add_trace(go.Scatter(
        x=np.arange(len(pacf_values)),
        y=confint[:, 0] - pacf_values,
        fill=None,
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.3)'),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=np.arange(len(pacf_values)),
        y=confint[:, 1] - pacf_values,
        fill='tonexty',
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.3)'),
        showlegend=False
    ))

    # Update layout
    fig.update_layout(
        title=f'Partial Autocorrelation Function (PACF) Plot for {feature}',
        xaxis_title='Lag',
        yaxis_title='Partial Autocorrelation',
        template='plotly_white'
    )

    fig.show()


def create_pred_observed_plot(df, forecasted_values, confidence_intervals, forecast_steps) -> any:
    # Create the plot
    fig = go.Figure()

    # Add observed data
    fig.add_trace(go.Scatter(x=df.index, y=df['revenue'], mode='lines', name='Observed'))

    # Add forecasted data
    fig.add_trace(go.Scatter(x=forecasted_values.index, y=forecasted_values, mode='lines', name='Forecasted'))

    # Add confidence intervals
    # fig.add_trace(go.Scatter(x=confidence_intervals.index, y=confidence_intervals.iloc[:, 0], fill=None, mode='lines', line_color='lightgrey', name='Lower CI'))
    # fig.add_trace(go.Scatter(x=confidence_intervals.index, y=confidence_intervals.iloc[:, 1], fill='tonexty', mode='lines', line_color='lightgrey', name='Upper CI'))

    # Add vertical line to indicate end of training data
    fig.add_vline(x=df.index[-forecast_steps], line_dash='dash', line_color='gray')

    # Customize layout
    fig.update_layout(
        title='Revenue Forecast',
        xaxis_title='Month',
        yaxis_title='Revenue Difference',
        legend_title='Series',
        template='plotly_white'
    )

    # Show the plot
    fig.show()