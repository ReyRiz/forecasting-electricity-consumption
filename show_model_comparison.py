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
