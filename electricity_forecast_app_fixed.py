import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMAResults
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# Set page configuration
st.set_page_config(
    page_title="Electricity Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Apply custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #333;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .card {
        border-radius: 5px;
        padding: 20px;
        background-color: #f9f9f9;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f8ff;
        border-left: 5px solid #1E88E5;
    }
    .highlight {
        background-color: #e1f5fe;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Title and introduction
st.markdown('<div class="main-header">⚡ Electricity Consumption Forecasting</div>', unsafe_allow_html=True)

# Sidebar for navigation and controls
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Forecasting", "Model Comparison", "Usage Insights", "About"])

# Function to load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('dataset.csv', sep=',', 
                       parse_dates={'dt': ['Date', 'Time']}, 
                       infer_datetime_format=True, 
                       low_memory=False, 
                       na_values=['nan', '?'], 
                       index_col='dt')
        # Handle missing values
        df_hourly = df.resample('H').mean()
        df_hourly_interp = df_hourly.interpolate(method='time')
        df_hourly_interp = df_hourly_interp.interpolate(method='linear', limit_direction='both')
        return df_hourly_interp
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# Function to load models
@st.cache_resource
def load_models():
    models = {}
    
    # Define custom_objects to handle 'mse' metric issue
    custom_objects = {'mse': tf.keras.losses.MeanSquaredError()}
    
    # Try to load Enhanced LSTM model
    try:
        # First try with enhanced_lstm_model.h5
        try:
            models["enhanced_lstm"] = {
                "model": load_model("enhanced_lstm_model.h5", custom_objects=custom_objects),
                "scaler": joblib.load("enhanced_lstm_scaler.pkl")
            }
            st.sidebar.success("✅ Enhanced LSTM model loaded")
        except Exception as e1:
            # Try alternative filename
            try:
                models["enhanced_lstm"] = {
                    "model": load_model("best_enhanced_lstm_model.h5", custom_objects=custom_objects),
                    "scaler": joblib.load("enhanced_lstm_scaler.pkl")
                }
                st.sidebar.success("✅ Enhanced LSTM model loaded (from best_enhanced_lstm_model.h5)")
            except Exception as e2:
                st.sidebar.warning(f"⚠️ Enhanced LSTM model not loaded: {str(e1)} or {str(e2)}")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Enhanced LSTM model not loaded: {str(e)}")
    
    # Try to load Basic LSTM model
    try:
        models["basic_lstm"] = {
            "model": load_model("best_lstm_model.h5", custom_objects=custom_objects),
            "scaler": joblib.load("lstm_scaler.pkl")
        }
        st.sidebar.success("✅ Basic LSTM model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Basic LSTM model not loaded: {str(e)}")
    
    # Try to load SARIMA model
    try:
        models["sarima"] = {
            "model": joblib.load("best_sarima_tuned_model.pkl")
        }
        st.sidebar.success("✅ SARIMA model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ SARIMA model not loaded: {str(e)}")
    
    # Try to load ARIMA model
    try:
        models["arima"] = {
            "model": joblib.load("best_arima_model.pkl")
        }
        st.sidebar.success("✅ ARIMA model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ ARIMA model not loaded: {str(e)}")
    
    return models

# Function to create time features for LSTM
def create_time_features(dates):
    # Extract time features
    hour_of_day = np.array([d.hour for d in dates]).reshape(-1, 1)
    day_of_week = np.array([d.dayofweek for d in dates]).reshape(-1, 1)
    
    # Cyclic encoding for time
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24)
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)
    
    # Combine features
    time_features = np.concatenate([hour_sin, hour_cos, day_sin, day_cos], axis=1)
    return time_features

# Function to create sequences for LSTM models
def create_lstm_dataset(dataset, time_steps=24):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), 0])
        y.append(dataset[i + time_steps, 0])
    return np.array(X), np.array(y)

# Function to forecast with LSTM
def forecast_lstm(model_type, df, forecast_days=7):
    models = load_models()
    
    if model_type not in models:
        st.error(f"Model '{model_type}' not available. Please check model files.")
        st.info("Looking for files: enhanced_lstm_model.h5 or best_enhanced_lstm_model.h5, and enhanced_lstm_scaler.pkl")
        return None, None
    
    try:
        model_info = models[model_type]
        if "model" not in model_info or "scaler" not in model_info:
            st.error(f"Model {model_type} is missing required components (model or scaler)")
            return None, None
        
        model = model_info["model"]
        scaler = model_info["scaler"]
          # Detect the correct time_steps by checking the model's input shape
        # Enhanced LSTM model uses 48 time steps, while Basic LSTM uses 24
        input_shape = model.input_shape
        if isinstance(input_shape, list):  # Enhanced LSTM has multiple inputs
            time_steps = input_shape[0][1]  # Get time_steps from the first input
        else:
            time_steps = input_shape[1]  # Basic LSTM has a single input
            
        st.info(f"Using time_steps={time_steps} for {model_type} model based on model input shape")
        target_col = 'Global_active_power'
        
        if len(df) < time_steps:
            st.error(f"Not enough data points. Need at least {time_steps} hours of data for {model_type} model.")
            return None, None
        
        last_data = df[target_col].iloc[-time_steps:].values.reshape(-1, 1)
        last_scaled = scaler.transform(last_data)
        
        if model_type == "enhanced_lstm":
            # For enhanced LSTM, need time features
            last_dates = df.index[-time_steps:]
            last_time_features = create_time_features(last_dates)
            
            # Reshape for model input
            last_values = last_scaled.reshape(1, time_steps, 1)
            last_time_features = last_time_features.reshape(1, time_steps, 4)
            
            # Forecast future values
            forecasted_values = []
            current_sequence = last_values.copy()
            current_time_features = last_time_features.copy()
            
            future_steps = 24 * forecast_days
            forecast_dates = [df.index[-1] + timedelta(hours=i+1) for i in range(future_steps)]
            
            for i in range(future_steps):
                # Generate time features for next step
                next_date = forecast_dates[i]
                next_time_features = create_time_features(np.array([next_date]))
                next_time_features_reshaped = next_time_features.reshape(1, 1, 4)
                  # Predict next value
                try:
                    next_value = model.predict([current_sequence, current_time_features], verbose=0)[0, 0]
                    forecasted_values.append(next_value)
                    
                    # Update sequences for next prediction
                    next_value_reshaped = np.array([[[next_value]]])
                    current_sequence = np.concatenate((current_sequence[:, 1:, :], next_value_reshaped), axis=1)
                    current_time_features = np.concatenate((current_time_features[:, 1:, :], next_time_features_reshaped), axis=1)
                except Exception as e:
                    st.error(f"Prediction error for {model_type}: {str(e)}")
                    st.info(f"Current sequence shape: {current_sequence.shape}, time features shape: {current_time_features.shape}")
                    return None, None
        
        else:  # Basic LSTM
            # Reshape for model input
            last_sequence = last_scaled.reshape(1, time_steps, 1)
            
            # Forecast future values
            forecasted_values = []
            current_sequence = last_sequence.copy()
            
            future_steps = 24 * forecast_days
            forecast_dates = [df.index[-1] + timedelta(hours=i+1) for i in range(future_steps)]
            for _ in range(future_steps):
                # Predict next value
                try:
                    next_value = model.predict(current_sequence, verbose=0)[0, 0]
                    forecasted_values.append(next_value)
                    
                    # Update sequence for next prediction
                    next_value_reshaped = np.array([[[next_value]]])
                    current_sequence = np.concatenate((current_sequence[:, 1:, :], next_value_reshaped), axis=1)
                except Exception as e:
                    st.error(f"Prediction error for {model_type}: {str(e)}")
                    st.info(f"Current sequence shape: {current_sequence.shape}")
                    return None, None
        
        # Inverse transform to get actual values
        forecasted_values_inv = scaler.inverse_transform(np.array(forecasted_values).reshape(-1, 1))
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecasted_values_inv, index=forecast_dates, columns=[target_col])
        
        return forecast_df, forecast_dates
    
    except Exception as e:
        st.error(f"Error in {model_type} forecasting: {str(e)}")
        return None, None

# Function to forecast with SARIMA
def forecast_sarima(df, forecast_days=7):
    models = load_models()
    
    if "sarima" not in models:
        st.error("SARIMA model not available")
        return None, None
    
    try:
        if "model" not in models["sarima"]:
            st.error("SARIMA model is missing required components")
            return None, None
            
        model = models["sarima"]["model"]
        future_steps = 24 * forecast_days
        
        # Generate forecast
        forecast = model.get_forecast(steps=future_steps)
        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()
        
        # Fix negative values
        forecast_mean = forecast_mean.apply(lambda x: max(0, x))
        forecast_ci.iloc[:, 0] = forecast_ci.iloc[:, 0].apply(lambda x: max(0, x))
        
        forecast_dates = forecast_mean.index
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame({
            'Global_active_power': forecast_mean,
            'lower_ci': forecast_ci.iloc[:, 0],
            'upper_ci': forecast_ci.iloc[:, 1]
        })
        
        return forecast_df, forecast_dates
    
    except Exception as e:
        st.error(f"Error in SARIMA forecasting: {str(e)}")
        return None, None

# Function to forecast with ARIMA
def forecast_arima(df, forecast_days=7):
    models = load_models()
    
    if "arima" not in models:
        st.error("ARIMA model not available")
        return None, None
    
    try:
        if "model" not in models["arima"]:
            st.error("ARIMA model is missing required components")
            return None, None
            
        model = models["arima"]["model"]
        future_steps = 24 * forecast_days
        
        # Generate forecast
        forecast = model.forecast(steps=future_steps)
        
        # Fix negative values
        forecast = forecast.apply(lambda x: max(0, x))
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=future_steps, freq='H')
        
        # Create forecast DataFrame
        forecast_df = pd.DataFrame(forecast.values, index=forecast_dates, columns=['Global_active_power'])
        
        return forecast_df, forecast_dates
    
    except Exception as e:
        st.error(f"Error in ARIMA forecasting: {str(e)}")
        return None, None

# Dashboard page
def show_dashboard(df):
    st.markdown('<div class="sub-header">📊 Current Consumption Overview</div>', unsafe_allow_html=True)
    
    try:
        # Get recent data
        recent_data = df.iloc[-24*7:]  # Last 7 days
        
        # Create three columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
            current_consumption = df['Global_active_power'].iloc[-1]
            st.metric("Current Consumption", f"{current_consumption:.2f} kW", 
                     delta=f"{(current_consumption - df['Global_active_power'].iloc[-2]):.2f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col2:
            st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
            daily_avg = recent_data['Global_active_power'].resample('D').mean().mean()
            st.metric("Daily Average", f"{daily_avg:.2f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col3:
            st.markdown('<div class="card metric-card">', unsafe_allow_html=True)
            peak_consumption = recent_data['Global_active_power'].max()
            peak_time = recent_data['Global_active_power'].idxmax().strftime('%Y-%m-%d %H:%M')
            st.metric("Peak Consumption", f"{peak_consumption:.2f} kW", 
                     delta_color="off", delta=f"at {peak_time}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Daily consumption pattern for the last week
        st.markdown('<div class="sub-header">📈 Recent Consumption Patterns</div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Last 7 Days", "Hourly Pattern"])
        
        with tab1:
            # Plot last 7 days consumption
            fig = px.line(recent_data, y='Global_active_power', 
                         title='Electricity Consumption - Last 7 Days')
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Global Active Power (kW)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Plot hourly pattern
            hourly_pattern = recent_data.groupby(recent_data.index.hour)['Global_active_power'].mean()
            
            fig = px.line(x=hourly_pattern.index, y=hourly_pattern.values, markers=True,
                        title='Average Hourly Consumption Pattern')
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Global Active Power (kW)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekly pattern
        st.markdown('<div class="sub-header">📅 Weekly Consumption Pattern</div>', unsafe_allow_html=True)
        
        # Get weekly pattern
        daily_pattern = recent_data.groupby(recent_data.index.day_name())['Global_active_power'].mean()
        # Reorder days
        days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        daily_pattern = daily_pattern.reindex(days_order)
        
        fig = px.bar(x=daily_pattern.index, y=daily_pattern.values,
                    title='Average Daily Consumption by Weekday')
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Average Global Active Power (kW)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error displaying dashboard: {str(e)}")

# Forecasting page
def show_forecasting(df):
    st.markdown('<div class="sub-header">🔮 Electricity Consumption Forecast</div>', unsafe_allow_html=True)
    
    # Forecasting options
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        model_type = st.selectbox("Select Model", 
                                ["Enhanced LSTM", "Basic LSTM", "SARIMA", "ARIMA"], 
                                index=0)
        
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=30, value=7)
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Generating forecast..."):
                forecast_df = None
                forecast_dates = None
                confidence_interval = False
                
                try:
                    if model_type == "Enhanced LSTM":
                        st.info("Attempting to load Enhanced LSTM model...")
                        forecast_df, forecast_dates = forecast_lstm("enhanced_lstm", df, forecast_days)
                        confidence_interval = False
                    elif model_type == "Basic LSTM":
                        st.info("Attempting to load Basic LSTM model...")
                        forecast_df, forecast_dates = forecast_lstm("basic_lstm", df, forecast_days)
                        confidence_interval = False
                    elif model_type == "SARIMA":
                        st.info("Attempting to load SARIMA model...")
                        forecast_df, forecast_dates = forecast_sarima(df, forecast_days)
                        confidence_interval = True
                    else:  # ARIMA
                        st.info("Attempting to load ARIMA model...")
                        forecast_df, forecast_dates = forecast_arima(df, forecast_days)
                        confidence_interval = False
                except Exception as e:
                    st.error(f"Error generating forecast: {str(e)}")
                    forecast_df = None
                
                # Check if forecast was successful
                if forecast_df is not None:
                    st.session_state.forecast_df = forecast_df
                    st.session_state.forecast_dates = forecast_dates
                    st.session_state.model_type = model_type
                    st.session_state.confidence_interval = confidence_interval
                    st.session_state.show_forecast = True
                else:
                    st.error(f"Failed to generate forecast with {model_type} model. Please check if model files exist and are valid.")
                    if "show_forecast" in st.session_state:
                        st.session_state.show_forecast = False
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Display consumption insights
        if hasattr(st.session_state, "show_forecast") and st.session_state.show_forecast:
            st.markdown('<div class="card highlight">', unsafe_allow_html=True)
            st.markdown("### Consumption Insights")
            
            try:
                forecast_data = st.session_state.forecast_df
                
                # Calculate and display statistics
                avg_consumption = forecast_data['Global_active_power'].mean()
                peak_consumption = forecast_data['Global_active_power'].max()
                peak_time = forecast_data['Global_active_power'].idxmax().strftime('%Y-%m-%d %H:%M')
                
                st.markdown(f"**Average Consumption:** {avg_consumption:.2f} kW")
                st.markdown(f"**Peak Consumption:** {peak_consumption:.2f} kW at {peak_time}")
                
                # Get daily patterns from forecast
                daily_pattern = forecast_data.groupby(forecast_data.index.day_name())['Global_active_power'].mean()
                highest_day = daily_pattern.idxmax()
                lowest_day = daily_pattern.idxmin()
                
                st.markdown(f"**Highest Consumption Day:** {highest_day}")
                st.markdown(f"**Lowest Consumption Day:** {lowest_day}")
                
                # Get hourly patterns
                hourly_pattern = forecast_data.groupby(forecast_data.index.hour)['Global_active_power'].mean()
                high_usage_threshold = hourly_pattern.mean() * 1.2
                high_usage_hours = hourly_pattern[hourly_pattern > high_usage_threshold].index.tolist()
                high_usage_hours.sort()
                
                low_usage_threshold = hourly_pattern.mean() * 0.8
                low_usage_hours = hourly_pattern[hourly_pattern < low_usage_threshold].index.tolist()
                low_usage_hours.sort()
                
                st.markdown("**High Usage Hours:** " + ", ".join([f"{h}:00" for h in high_usage_hours]))
                st.markdown("**Low Usage Hours:** " + ", ".join([f"{h}:00" for h in low_usage_hours]))
            except Exception as e:
                st.error(f"Error displaying consumption insights: {str(e)}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Display forecast if available
        if hasattr(st.session_state, "show_forecast") and st.session_state.show_forecast:
            try:
                # Get data from session state
                forecast_df = st.session_state.forecast_df
                model_type = st.session_state.model_type
                confidence_interval = st.session_state.confidence_interval
                
                # Get historical data for plotting
                historical_data = df.iloc[-24*7:]
                
                # Create plot
                fig = go.Figure()
                
                # Add historical data
                fig.add_trace(go.Scatter(
                    x=historical_data.index,
                    y=historical_data['Global_active_power'],
                    name='Historical Data',
                    line=dict(color='blue', width=1.5),
                    opacity=0.7
                ))
                
                # Add forecast
                fig.add_trace(go.Scatter(
                    x=forecast_df.index,
                    y=forecast_df['Global_active_power'],
                    name=f'{model_type} Forecast',
                    line=dict(color='red', width=2)
                ))
                
                # Add confidence intervals if available
                if confidence_interval:
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['upper_ci'],
                        name='Upper CI',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        showlegend=False
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=forecast_df.index,
                        y=forecast_df['lower_ci'],
                        name='Lower CI',
                        line=dict(color='rgba(255,0,0,0.2)', width=0),
                        fill='tonexty',
                        fillcolor='rgba(255,0,0,0.1)',
                        showlegend=False
                    ))
                
                # Add weekend shading
                for date in pd.date_range(start=forecast_df.index[0], end=forecast_df.index[-1], freq='D'):
                    if date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                        fig.add_vrect(
                            x0=date,
                            x1=date + pd.Timedelta(days=1),
                            fillcolor="lightgrey",
                            opacity=0.2,
                            line_width=0,
                            layer="below"
                        )
                
                # Update layout
                fig.update_layout(
                    title=f'Electricity Consumption Forecast ({model_type})',
                    xaxis_title='Date',
                    yaxis_title='Global Active Power (kW)',
                    height=600,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display forecast data table
                with st.expander("Show Forecast Data"):
                    st.dataframe(forecast_df.reset_index().rename(columns={'index': 'datetime'}))
                    
                    # Download button
                    csv = forecast_df.to_csv().encode('utf-8')
                    st.download_button(
                        label="Download Forecast Data",
                        data=csv,
                        file_name=f'electricity_forecast_{model_type}_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                    )
            except Exception as e:
                st.error(f"Error displaying forecast: {str(e)}")
        else:
            st.info("Select a model and click 'Generate Forecast' to see predictions.")

# Model comparison page
def show_model_comparison(df):
    st.markdown('<div class="sub-header">🔍 Model Comparison</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        forecast_days = st.slider("Forecast Days", min_value=1, max_value=14, value=7, key="comp_days")
        
        models_to_compare = st.multiselect(
            "Select Models to Compare",
            ["Enhanced LSTM", "Basic LSTM", "SARIMA", "ARIMA"],
            default=["Enhanced LSTM", "SARIMA"]
        )
        
        if st.button("Compare Models", type="primary"):
            if len(models_to_compare) == 0:
                st.error("Please select at least one model to compare")
            else:
                forecasts = {}
                
                with st.spinner("Generating forecasts for comparison..."):
                    try:
                        for model in models_to_compare:
                            if model == "Enhanced LSTM":
                                forecast_df, _ = forecast_lstm("enhanced_lstm", df, forecast_days)
                                if forecast_df is not None:
                                    forecasts["Enhanced LSTM"] = forecast_df
                            elif model == "Basic LSTM":
                                forecast_df, _ = forecast_lstm("basic_lstm", df, forecast_days)
                                if forecast_df is not None:
                                    forecasts["Basic LSTM"] = forecast_df
                            elif model == "SARIMA":
                                forecast_df, _ = forecast_sarima(df, forecast_days)
                                if forecast_df is not None:
                                    forecasts["SARIMA"] = forecast_df[['Global_active_power']]
                            elif model == "ARIMA":
                                forecast_df, _ = forecast_arima(df, forecast_days)
                                if forecast_df is not None:
                                    forecasts["ARIMA"] = forecast_df
                    except Exception as e:
                        st.error(f"Error generating forecasts: {str(e)}")
                
                if forecasts:
                    st.session_state.comparison_forecasts = forecasts
                    st.session_state.show_comparison = True
                else:
                    st.error("Failed to generate forecasts. Please check if model files exist and are valid.")
                    if "show_comparison" in st.session_state:
                        st.session_state.show_comparison = False
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Show model descriptions
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Model Descriptions")
        
        st.markdown("""
        **Enhanced LSTM**: A deep learning model that uses both historical consumption data and temporal features (hour of day, day of week) to capture complex patterns.
        
        **Basic LSTM**: A simpler deep learning model that only uses historical consumption data without additional features.
        
        **SARIMA**: Statistical model that captures seasonal patterns (hourly, daily, weekly) and trends in the data.
        
        **ARIMA**: Statistical model that captures trends and autocorrelation in the data without seasonal components.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        if hasattr(st.session_state, "show_comparison") and st.session_state.show_comparison:
            try:
                forecasts = st.session_state.comparison_forecasts
                
                if len(forecasts) > 0:
                    # Get historical data for context
                    historical_data = df.iloc[-24*7:]
                    
                    # Create comparison plot
                    fig = go.Figure()
                    
                    # Add historical data
                    fig.add_trace(go.Scatter(
                        x=historical_data.index,
                        y=historical_data['Global_active_power'],
                        name='Historical Data',
                        line=dict(color='black', width=1.5),
                        opacity=0.7
                    ))
                    
                    # Add each model's forecast
                    colors = ['red', 'blue', 'green', 'purple']
                    for i, (model_name, forecast_df) in enumerate(forecasts.items()):
                        fig.add_trace(go.Scatter(
                            x=forecast_df.index,
                            y=forecast_df['Global_active_power'],
                            name=f'{model_name}',
                            line=dict(color=colors[i % len(colors)], width=2)
                        ))
                    
                    # Update layout
                    fig.update_layout(
                        title='Model Comparison: Forecasted Electricity Consumption',
                        xaxis_title='Date',
                        yaxis_title='Global Active Power (kW)',
                        height=600,
                        hovermode="x unified"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display forecast statistics
                    st.markdown('<div class="sub-header">Forecast Statistics</div>', unsafe_allow_html=True)
                    
                    # Create statistics table
                    stats = []
                    for model_name, forecast_df in forecasts.items():
                        stats.append({
                            'Model': model_name,
                            'Average (kW)': forecast_df['Global_active_power'].mean(),
                            'Min (kW)': forecast_df['Global_active_power'].min(),
                            'Max (kW)': forecast_df['Global_active_power'].max(),
                            'Std Dev': forecast_df['Global_active_power'].std()
                        })
                    
                    stats_df = pd.DataFrame(stats)
                    st.dataframe(stats_df)
                else:
                    st.warning("No forecasts available for comparison. Please check if models are properly loaded.")
            except Exception as e:
                st.error(f"Error displaying model comparison: {str(e)}")
        else:
            st.info("Select models to compare and click 'Compare Models' to see comparison.")

# Usage insights page
def show_usage_insights(df):
    st.markdown('<div class="sub-header">💡 Usage Insights & Recommendations</div>', unsafe_allow_html=True)
    
    try:
        # Get recent data for analysis
        recent_data = df.iloc[-24*30:]  # Last 30 days for more robust patterns
        
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Daily Consumption Patterns")
            
            # Calculate daily patterns
            daily_pattern = recent_data.groupby(recent_data.index.day_name())['Global_active_power'].mean()
            days_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
            daily_pattern = daily_pattern.reindex(days_order)
            
            # Find highest and lowest consumption days
            highest_day = daily_pattern.idxmax()
            lowest_day = daily_pattern.idxmin()
            
            # Plot daily pattern
            fig = px.bar(
                x=daily_pattern.index, 
                y=daily_pattern.values,
                title='Average Consumption by Day of Week'
            )
            fig.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Average Global Active Power (kW)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Highest consumption day:** {highest_day}")
            st.markdown(f"**Lowest consumption day:** {lowest_day}")
            
            if highest_day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]:
                st.markdown("👉 **Insight:** Consumption is higher on weekdays, suggesting work-related activities drive electricity usage.")
            else:
                st.markdown("👉 **Insight:** Consumption is higher on weekends, suggesting home-based activities drive electricity usage.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Hourly Consumption Patterns")
            
            # Calculate hourly patterns
            hourly_pattern = recent_data.groupby(recent_data.index.hour)['Global_active_power'].mean()
            
            # Identify peak and off-peak hours
            high_usage_threshold = hourly_pattern.mean() * 1.2
            high_usage_hours = hourly_pattern[hourly_pattern > high_usage_threshold].index.tolist()
            high_usage_hours.sort()
            
            low_usage_threshold = hourly_pattern.mean() * 0.8
            low_usage_hours = hourly_pattern[hourly_pattern < low_usage_threshold].index.tolist()
            low_usage_hours.sort()
            
            # Plot hourly pattern
            fig = px.line(
                x=hourly_pattern.index, 
                y=hourly_pattern.values,
                markers=True,
                title='Average Consumption by Hour of Day'
            )
            
            # Add threshold lines
            fig.add_hline(
                y=high_usage_threshold, 
                line_dash="dash", 
                line_color="red",
                annotation_text="High Usage Threshold"
            )
            
            fig.add_hline(
                y=low_usage_threshold, 
                line_dash="dash", 
                line_color="green",
                annotation_text="Low Usage Threshold"
            )
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Average Global Active Power (kW)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"**Peak hours:** {', '.join([f'{h}:00' for h in high_usage_hours])}")
            st.markdown(f"**Off-peak hours:** {', '.join([f'{h}:00' for h in low_usage_hours])}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Recommendations section
        st.markdown('<div class="sub-header">🎯 Energy Optimization Recommendations</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card highlight">', unsafe_allow_html=True)
            st.markdown("### Cost Saving Recommendations")
            
            st.markdown("""
            #### 💰 Shift High-Power Activities
            - Schedule high-power appliances (washing machines, dryers, dishwashers) during off-peak hours:
              {}
            
            #### ⚡ Optimize HVAC Usage
            - Program your thermostat to reduce heating/cooling during peak hours
            - Pre-cool or pre-heat your home during off-peak hours
            
            #### 🔌 Manage Standby Power
            - Use smart power strips to eliminate standby power during peak hours
            - Unplug non-essential electronics when not in use
            """.format(", ".join([f"{h}:00" for h in low_usage_hours])))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card highlight">', unsafe_allow_html=True)
            st.markdown("### Potential Savings Estimation")
            
            # Calculate potential savings
            avg_consumption = hourly_pattern.mean()
            high_consumption = hourly_pattern[high_usage_hours].mean()
            potential_savings_pct = ((high_consumption - avg_consumption) / high_consumption) * 100
            
            # Estimated savings calculation
            electricity_rate = st.slider("Electricity Rate ($/kWh)", 0.10, 0.40, 0.20, 0.01)
            daily_kwh = recent_data['Global_active_power'].resample('D').mean().mean() * 24
            monthly_kwh = daily_kwh * 30
            monthly_cost = monthly_kwh * electricity_rate
            
            potential_monthly_savings = monthly_cost * potential_savings_pct / 100
            
            st.markdown(f"""
            #### Current Usage Estimate
            - Daily average: {daily_kwh:.2f} kWh
            - Monthly average: {monthly_kwh:.2f} kWh
            - Estimated monthly cost: ${monthly_cost:.2f}
            
            #### Potential Savings
            - If you shift usage from peak to off-peak hours, you could save approximately:
              **{potential_savings_pct:.1f}%** of your electricity costs.
            
            - Estimated monthly savings: **${potential_monthly_savings:.2f}**
            - Estimated annual savings: **${potential_monthly_savings*12:.2f}**
            """)
            st.markdown('</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying usage insights: {str(e)}")

# About page
def show_about():
    st.markdown('<div class="sub-header">ℹ️ About This App</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### Electricity Consumption Forecasting App
    
    This application is designed to help users understand and forecast their electricity consumption patterns. 
    It uses advanced time series forecasting models to predict future electricity usage.
    
    #### Features:
    - **Dashboard**: View current consumption metrics and patterns
    - **Forecasting**: Generate predictions using multiple models
    - **Model Comparison**: Compare different forecasting approaches
    - **Usage Insights**: Get recommendations for optimizing electricity usage
    
    #### Models Implemented:
    - **Enhanced LSTM**: Deep learning model with temporal features
    - **Basic LSTM**: Simple deep learning approach
    - **SARIMA**: Statistical model with seasonal components
    - **ARIMA**: Statistical model for time series data
    
    #### Data Source:
    The application uses household electricity consumption data, which includes measurements of global active power.
    
    #### Credits:
    Developed using Streamlit, TensorFlow, Statsmodels, and Plotly.
    """)
    
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### How to Use This App")
    
    st.markdown("""
    1. **Dashboard**: View your current consumption patterns and metrics.
    2. **Forecasting**: Select a model and forecast horizon to predict future consumption.
    3. **Model Comparison**: Compare different models to see which one works best for your data.
    4. **Usage Insights**: Get personalized recommendations to reduce your electricity consumption and costs.
    
    The sidebar navigation allows you to move between these different sections of the app.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Main app logic
def main():
    # Load data
    df = load_data()
    
    # Load models
    models = load_models()
    
    if df is None:
        st.error("Error loading data. Please check if 'dataset.csv' exists in the correct location.")
        return
    
    try:
        # Display selected page
        if page == "Dashboard":
            show_dashboard(df)
        elif page == "Forecasting":
            show_forecasting(df)
        elif page == "Model Comparison":
            show_model_comparison(df)
        elif page == "Usage Insights":
            show_usage_insights(df)
        elif page == "About":
            show_about()
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
