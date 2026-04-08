import streamlit as st
import pandas as pd
import yfinance as yf
from prophet import Prophet
import plotly.graph_objs as go
from datetime import date, timedelta

st.set_page_config(page_title="Financial Forecasting Tool", page_icon="📈", layout="wide")

st.title("📈 Financial Forecasting Web Tool")
st.markdown("Forecast business revenue, expenses, or stock prices using Meta's Prophet algorithm.")

# Initialize session state for data
if 'df' not in st.session_state:
    st.session_state['df'] = None

# Sidebar components
st.sidebar.header("Data Source Setup")
data_source = st.sidebar.radio("Select Data Source", ["Fetch Stock Data (yfinance)", "Upload CSV"])

@st.cache_data
def load_stock_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    # yfinance sometimes returns a MultiIndex when downloading. We handle it by grabbing the close or just flattening.
    # The new yfinance returns columns with MultiIndex if downloaded multiple or singular. 
    # For a single ticker, it usually returns standard index but sometimes MultiIndex columns.
    # Let's ensure it's a standard dataframe
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    df.reset_index(inplace=True)
    # yfinance index is usually Date, rename if necessary to make it clean
    if 'Date' in df.columns:
        df.rename(columns={'Date': 'ds'}, inplace=True)
    return df

df = None

if data_source == "Fetch Stock Data (yfinance)":
    st.sidebar.markdown("---")
    ticker = st.sidebar.text_input("Enter Ticker Symbol", value="AAPL")
    start_date = st.sidebar.date_input("Start Date", date.today() - timedelta(days=365*5))
    end_date = st.sidebar.date_input("End Date", date.today())
    
    if st.sidebar.button("Fetch Data", type="primary"):
        with st.spinner(f"Fetching data for {ticker}..."):
            try:
                fetched_df = load_stock_data(ticker, start_date, end_date)
                if not fetched_df.empty:
                    st.session_state['df'] = fetched_df
                    st.sidebar.success("Data fetched successfully!")
                else:
                    st.sidebar.error("Failed to fetch data or empty dataset.")
            except Exception as e:
                st.sidebar.error(f"Error fetching data: {e}")

elif data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your historical data (CSV format)", type=["csv"])
    if uploaded_file is not None:
        try:
            st.session_state['df'] = pd.read_csv(uploaded_file)
            st.sidebar.success("CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

# Application Content Body
df = st.session_state['df']

if df is not None and not df.empty:
    st.subheader("1. Data Exploration")
    
    col1, col2 = st.columns([1, 1.5])
    with col1:
        st.write("Raw Data Snapshot:")
        st.dataframe(df.head(20), use_container_width=True)
    with col2:
        st.write("Summary Statistics:")
        st.dataframe(df.describe(), use_container_width=True)

    st.markdown("---")
    st.subheader("2. Forecasting Setup")
    col3, col4 = st.columns(2)
    with col3:
        # Auto-select columns if possible to make UX better
        date_cols = [c for c in df.columns if 'date' in str(c).lower() or 'time' in str(c).lower() or c == 'ds']
        target_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        default_date_idx = df.columns.tolist().index(date_cols[0]) if date_cols else 0
        date_column = st.selectbox("Select Date Column (must be a date/timestamp)", df.columns, index=default_date_idx)
        
    with col4:
        # If it's yfinance, try to default to Close
        target_default = 'Close' if 'Close' in target_cols else target_cols[0] if target_cols else 0
        default_target_idx = df.columns.tolist().index(target_default) if target_default in df.columns else 0
        target_column = st.selectbox("Select Target Column to Forecast", df.columns, index=default_target_idx)
    
    forecast_horizon = st.slider("Forecast Horizon (Days from End Date)", min_value=30, max_value=365*5, value=365, step=30)
    
    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Training Prophet model and generating forecast (this may take a moment)..."):
            try:
                # Prepare data for Prophet: needs 'ds' and 'y' columns
                prophet_df = df[[date_column, target_column]].copy()
                prophet_df = prophet_df.rename(columns={date_column: 'ds', target_column: 'y'})
                
                # Clean Data
                prophet_df = prophet_df.dropna(subset=['y', 'ds'])
                prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
                if pd.api.types.is_datetime64tz_dtype(prophet_df['ds']):
                    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
                
                # Initialize and train Prophet
                m = Prophet(daily_seasonality=False, yearly_seasonality=True, weekly_seasonality=True)
                m.fit(prophet_df)
                
                # Make future dataframe
                future = m.make_future_dataframe(periods=forecast_horizon)
                forecast = m.predict(future)
                
                st.success("✅ Forecast generated successfully!")
                
                st.subheader("3. Forecast Visualization")
                
                # Plotly Chart
                fig = go.Figure()
                
                # Historical Data Trace
                fig.add_trace(go.Scatter(
                    x=prophet_df['ds'], y=prophet_df['y'], 
                    mode='lines', name='Historical Data', 
                    line=dict(color='#1f77b4')
                ))
                
                # Forecast Trace
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat'], 
                    mode='lines', name='Forecasted Trend', 
                    line=dict(color='#ff7f0e', dash='dot')
                ))
                
                # Confidence Intervals
                fig.add_trace(go.Scatter(
                    x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                    y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='Confidence Interval (95%)',
                    showlegend=True
                ))
                
                fig.update_layout(
                    title=f"Forecast of {target_column} for next {forecast_horizon} days",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    hovermode="x unified",
                    template='plotly_white',
                    height=600
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Show forecast components
                st.subheader("Forecast Components (Seasonality & Trend)")
                components_fig = m.plot_components(forecast)
                st.pyplot(components_fig)
                
                # Download forecast
                st.markdown("---")
                st.subheader("4. Data Export")
                
                def convert_df(df):
                    return df.to_csv(index=False).encode('utf-8')

                export_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'ds': 'Date', 'yhat': 'Predicted Value', 'yhat_lower': 'Lower Confidence Interval', 'yhat_upper': 'Upper Confidence Interval'
                })
                csv = convert_df(export_df)
                
                st.download_button(
                    label="📥 Download Forecast as CSV",
                    data=csv,
                    file_name='forecast_data.csv',
                    mime='text/csv',
                    type="primary"
                )
                
            except Exception as e:
                st.error(f"An error occurred during forecasting: {str(e)}")
else:
    st.info("👈 Please load data from the sidebar to begin.")
