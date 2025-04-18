import streamlit as st
import pandas as pd
import numpy as np
import yaml
import plotly.graph_objects as go
from datetime import datetime, timedelta
from bitcoin_prediction import BitcoinPricePredictor
import ta
from pathlib import Path

class CryptoApp:
    def __init__(self):
        self.config = self.load_config()
        self.predictor = BitcoinPricePredictor()
        
    @staticmethod
    def load_config():
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    
    def plot_price_history(self, df):
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Bitcoin Price',
            hoverlabel=dict(
                bgcolor='#1E1E1E',
                bordercolor='#FF9900',
                font=dict(color='#FFFFFF', size=14)
            )
        ))
        
        # Add moving averages
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['MA7'], 
            name='7-day Moving Average',
            line=dict(color='#4B9FE3', width=2),
            hoverlabel=dict(
                bgcolor='#1E1E1E',
                bordercolor='#4B9FE3',
                font=dict(color='#FFFFFF', size=14)
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "7-day MA: $%{y:,.2f}<br>" +
                         "<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['MA30'], 
            name='30-day Moving Average',
            line=dict(color='#FF9900', width=2),
            hoverlabel=dict(
                bgcolor='#1E1E1E',
                bordercolor='#FF9900',
                font=dict(color='#FFFFFF', size=14)
            ),
            hovertemplate="<b>%{x}</b><br>" +
                         "30-day MA: $%{y:,.2f}<br>" +
                         "<extra></extra>"
        ))
        
        fig.update_layout(
            title={
                'text': 'Bitcoin Price History with Moving Averages',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            yaxis_title='Price (USD)',
            xaxis_title='Date',
            template='plotly_dark',
            hovermode='x unified',
            hoverdistance=100,
            spikedistance=1000,
            hoverlabel=dict(
                bgcolor='#1E1E1E',
                font=dict(color='#FFFFFF', size=14),
                bordercolor='#FF9900'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(30, 30, 30, 0.8)',
                font=dict(color='#FFFFFF'),
                bordercolor='#FF9900',
                borderwidth=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            xaxis=dict(
                showspikes=True,
                spikethickness=2,
                spikedash='dot',
                spikecolor='#FF9900',
                spikesnap='cursor'
            ),
            yaxis=dict(
                showspikes=True,
                spikethickness=2,
                spikedash='dot',
                spikecolor='#FF9900',
                spikesnap='cursor'
            )
        )
        
        return fig
    
    def plot_technical_indicators(self, df):
        # Create subplots for different indicators
        fig = go.Figure()
        
        # Calculate additional technical indicators
        df['RSI_Color'] = ['red' if x > 70 else 'green' if x < 30 else 'gray' for x in df['RSI']]
        
        # Create subplots
        fig = go.Figure()
        
        # RSI subplot
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['RSI'],
            name='RSI',
            line=dict(color='white', width=2),
            hovertemplate='RSI: %{y:.2f}<extra></extra>'
        ))
        
        # Add RSI reference lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
        fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
        
        # MACD
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['MACD'],
            name='MACD',
            line=dict(color='blue', width=2),
            hovertemplate='MACD: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title={
                'text': 'Technical Indicators Analysis',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=24)
            },
            showlegend=True,
            template='plotly_dark',
            hovermode='x unified',
            height=500,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255, 255, 255, 0.1)'
            ),
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Add helpful annotations
        fig.add_annotation(
            text="RSI > 70: Overbought - Consider Selling",
            xref="paper", yref="paper",
            x=0.01, y=0.95,
            showarrow=False,
            font=dict(color="red", size=12)
        )
        
        fig.add_annotation(
            text="RSI < 30: Oversold - Consider Buying",
            xref="paper", yref="paper",
            x=0.01, y=0.90,
            showarrow=False,
            font=dict(color="green", size=12)
        )
        
        return fig
    
    def run(self):
        # Main title with styling
        st.markdown("""
        <style>
        .big-title {
            font-size: 48px;
            font-weight: bold;
            color: #FF9900;
            text-align: center;
            margin-bottom: 30px;
        }
        .subtitle {
            font-size: 24px;
            color: #CCCCCC;
            text-align: center;
            margin-bottom: 50px;
        }
        .metric-card {
            background-color: #1E1E1E;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="big-title">🚀 Bitcoin Price Prediction Dashboard</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">Real-time Bitcoin price analysis and predictions using machine learning</p>', unsafe_allow_html=True)
        
        # Help section
        with st.expander("ℹ️ How to Use This Dashboard"):
            st.markdown("""
            ### Welcome to the Bitcoin Price Prediction Dashboard!
            
            This dashboard helps you:
            1. **Monitor Current Bitcoin Prices** 📊
            2. **View Price Predictions** 🎯
            3. **Analyze Technical Indicators** 📈
            4. **Set Price Alerts** ⚠️
            
            #### Getting Started:
            - Use the sidebar to select prediction timeframes
            - View current and predicted prices in the top cards
            - Analyze price trends in the interactive charts
            - Set custom price alerts below
            
            #### Understanding the Charts:
            - **Candlestick Chart**: Shows price movement (green=price up, red=price down)
            - **Moving Averages**: Help identify trends (blue=7-day, orange=30-day)
            - **Technical Indicators**: Help predict future price movements
            """)
        
        # Sidebar
        st.sidebar.header('Settings')
        prediction_days = st.sidebar.selectbox(
            'Prediction Horizon',
            self.config['prediction_horizons'],
            format_func=lambda x: f'{x} day{"s" if x > 1 else ""}'
        )
        
        selected_models = st.sidebar.multiselect(
            'Select Models',
            ['KNN', 'XGBoost', 'LightGBM'],
            default=['KNN'],
            help='KNN is the basic model, while XGBoost and LightGBM are advanced models'
        )
        
        chart_period = st.sidebar.slider(
            'Historical data to show (days)',
            min_value=30,
            max_value=365,
            value=180,
            step=30,
            help='Adjust how much historical data you want to see in the charts'
        )
        
        # Main content
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Current Bitcoin Status')
            try:
                current_price, predicted_price = self.predictor.predict_future_price(days_ahead=prediction_days)
                
                if current_price and predicted_price:
                    price_change = ((predicted_price - current_price) / current_price) * 100
                    
                    st.metric(
                        label="Current Price",
                        value=f"${current_price:,.2f}",
                        delta=f"{price_change:+.2f}% predicted"
                    )
                    
                    st.metric(
                        label=f"Predicted Price ({prediction_days} days)",
                        value=f"${predicted_price:,.2f}"
                    )
            except Exception as e:
                st.error(f"Error getting predictions: {str(e)}")
        
        # Model Performance Section
        st.subheader('🤖 Model Performance')
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            if Path('bitcoin_model.joblib').exists():
                st.success('✅ Model is trained and ready')
                if st.button('🔄 Retrain Model'):
                    with st.spinner('Retraining model...'):
                        btc_data = self.predictor.fetch_bitcoin_data()
                        df = self.predictor.create_features(btc_data)
                        X_train, X_test, y_train, y_test = self.predictor.prepare_data(df)
                        self.predictor.train_model(X_train, y_train)
                        y_test, y_pred = self.predictor.evaluate_model(X_test, y_test)
                        st.success('✨ Model retrained successfully!')
            else:
                st.warning('⚠️ Model needs to be trained')
                if st.button('🎯 Train Model'):
                    with st.spinner('Training model...'):
                        btc_data = self.predictor.fetch_bitcoin_data()
                        df = self.predictor.create_features(btc_data)
                        X_train, X_test, y_train, y_test = self.predictor.prepare_data(df)
                        self.predictor.train_model(X_train, y_train)
                        st.success('✨ Model trained successfully!')
        
        with model_col2:
            st.info('💡 Model Information')
            st.markdown('''
            - **Algorithm**: K-Nearest Neighbors (KNN)
            - **Features**: Price, Returns, Moving Averages, RSI, MACD
            - **Training Data**: Historical Bitcoin prices since 2015
            - **Update Frequency**: Real-time predictions
            ''')
        
        # Price History Chart
        st.subheader('Price History and Predictions')
        btc_data = self.predictor.fetch_bitcoin_data()
        df = self.predictor.create_features(btc_data)
        
        fig_price = self.plot_price_history(df)
        st.plotly_chart(fig_price, use_container_width=True)
        
        # Technical Indicators
        st.subheader('Technical Analysis')
        fig_tech = self.plot_technical_indicators(df)
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # Alerts Section
        st.subheader('Price Alerts')
        alert_price = st.number_input('Set Price Alert ($)', min_value=0.0, value=float(current_price) if current_price else 0.0)
        alert_change = st.number_input('Price Change Threshold (%)', min_value=0.0, value=5.0)
        
        if st.button('Set Alert'):
            st.success(f'Alert set for ${alert_price:,.2f} with {alert_change}% threshold')

if __name__ == '__main__':
    app = CryptoApp()
    app.run()
