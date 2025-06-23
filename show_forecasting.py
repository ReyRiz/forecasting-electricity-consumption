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
                        forecast_df, forecast_dates = forecast_lstm("enhanced_lstm", df, forecast_days)
                        confidence_interval = False
                    elif model_type == "Basic LSTM":
                        forecast_df, forecast_dates = forecast_lstm("basic_lstm", df, forecast_days)
                        confidence_interval = False
                    elif model_type == "SARIMA":
                        forecast_df, forecast_dates = forecast_sarima(df, forecast_days)
                        confidence_interval = True
                    else:  # ARIMA
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
