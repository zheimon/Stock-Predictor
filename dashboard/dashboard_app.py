#!/usr/bin/env python3
"""
Stock Trading Dashboard with Auto-Start Functionality

A comprehensive Streamlit dashboard for monitoring real-time trading signals,
pattern predictions, and system status.

Features:
- Real-time signal monitoring for multiple tickers
- Auto-start functionality for live predictions
- Interactive charts and visualizations
- Pattern probability distributions
- System health monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
import subprocess
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

# Page configuration
st.set_page_config(
    page_title="Stock Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
    .buy-signal {
        color: #00ff00;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .sell-signal {
        color: #ff0000;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .hold-signal {
        color: #ffa500;
        font-weight: bold;
        font-size: 1.5rem;
    }
    .status-active {
        color: #00ff00;
    }
    .status-inactive {
        color: #ff0000;
    }
</style>
""", unsafe_allow_html=True)


class DashboardApp:
    """Main dashboard application class."""
    
    def __init__(self):
        self.data_dir = Path(Config.DATA_DIR)
        self.models_dir = Path(Config.MODELS_DIR)
        self.logs_dir = Path(Config.LOGS_DIR)
        self.available_tickers = self._get_available_tickers()
        
    def _get_available_tickers(self):
        """Get list of tickers with live signal files."""
        signal_files = list(self.data_dir.glob("live_signals_*.json"))
        tickers = [f.stem.replace("live_signals_", "") for f in signal_files]
        return sorted(tickers)
    
    def _load_live_signals(self, ticker):
        """Load live signals from JSON file."""
        signal_file = self.data_dir / f"live_signals_{ticker}.json"
        if signal_file.exists():
            try:
                with open(signal_file, 'r') as f:
                    signals = json.load(f)
                return signals if signals else []
            except Exception as e:
                st.warning(f"Error loading signals for {ticker}: {e}")
                return []
        return []
    
    def _load_live_status(self, ticker):
        """Load live prediction status from JSON file."""
        status_file = self.data_dir / f"live_status_{ticker}.json"
        if status_file.exists():
            try:
                with open(status_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                st.warning(f"Error loading status for {ticker}: {e}")
                return None
        return None
    
    def _check_prediction_running(self, ticker):
        """Check if live prediction is running for a ticker."""
        # Check both status file AND signals file for recent activity
        is_active = False
        
        # First check status file
        status = self._load_live_status(ticker)
        if status:
            last_update = datetime.fromisoformat(status.get('last_update', ''))
            time_diff = datetime.now() - last_update
            # Consider active if status updated within last 30 minutes
            if time_diff.total_seconds() < 1800:
                is_active = True
        
        # Also check signals file for recent signals (more reliable)
        if not is_active:
            signals = self._load_live_signals(ticker)
            if signals and len(signals) > 0:
                try:
                    # Check the most recent signal timestamp
                    latest_signal = signals[-1]
                    signal_timestamp = datetime.fromisoformat(latest_signal.get('timestamp', ''))
                    time_diff = datetime.now() - signal_timestamp
                    # Consider active if signal generated within last 10 minutes
                    if time_diff.total_seconds() < 600:
                        is_active = True
                except:
                    pass
        
        return is_active
    
    def _start_live_prediction(self, ticker, model_name):
        """Start live prediction for a ticker in the background."""
        try:
            script_path = Path(__file__).parent.parent / "scripts" / "live_prediction.py"
            cmd = [sys.executable, str(script_path), 
                   "--model_name", model_name, 
                   "--ticker", ticker]
            
            # Start in background (platform-specific)
            if sys.platform == 'win32':
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_CONSOLE)
            else:
                subprocess.Popen(cmd, start_new_session=True)
            
            return True
        except Exception as e:
            st.error(f"Failed to start live prediction: {e}")
            return False
    
    def render_header(self):
        """Render dashboard header."""
        st.markdown('<h1 class="main-header">📈 Stock Trading Dashboard</h1>', 
                   unsafe_allow_html=True)
        st.markdown("---")
    
    def render_sidebar(self):
        """Render sidebar with controls."""
        st.sidebar.header("⚙️ Dashboard Controls")
        
        # Model selection
        available_models = self._get_available_models()
        selected_model = st.sidebar.selectbox(
            "Select Model",
            available_models,
            index=0 if available_models else 0
        )
        
        # Show all available tickers automatically (no selection needed)
        if self.available_tickers:
            selected_tickers = self.available_tickers
        else:
            selected_tickers = []

        # Enter ticker for prediction
        st.sidebar.markdown("---")
        st.sidebar.subheader("➕ Add New Ticker")
        new_ticker = st.sidebar.text_input("Enter Ticker Symbol", "").upper()
        
        if st.sidebar.button("🚀 Start Live Prediction"):
            if new_ticker:
                with st.spinner(f"Starting live prediction for {new_ticker}..."):
                    if self._start_live_prediction(new_ticker, selected_model):
                        st.sidebar.success(f"Started prediction for {new_ticker}")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.sidebar.error(f"Failed to start prediction for {new_ticker}")
            else:
                st.sidebar.warning("Please enter a ticker symbol")
        
        # Refresh controls
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔄 Refresh Settings")
        auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
        refresh_interval = st.sidebar.slider("Refresh Interval (seconds)", 5, 60, 10)
        
        if st.sidebar.button("🔄 Refresh Now"):
            st.rerun()
        
        # System info
        st.sidebar.markdown("---")
        st.sidebar.subheader("ℹ️ System Info")
        st.sidebar.info(f"Active Tickers: {len(self.available_tickers)}")
        st.sidebar.info(f"Models Available: {len(available_models)}")
        
        return selected_tickers, auto_refresh, refresh_interval
    
    def _get_available_models(self):
        """Get list of available model files."""
        model_files = list(self.models_dir.glob("*.h5"))
        models = [f.stem for f in model_files]
        return sorted(models) if models else ["lstm_pattern_classifier"]
    
    def render_ticker_status(self, ticker):
        """Render status card for a ticker."""
        signals = self._load_live_signals(ticker)
        is_running = self._check_prediction_running(ticker)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_text = "🟢 Active" if is_running else "🔴 Inactive"
            st.metric("Status", status_text)
        
        # Calculate metrics from signals file (more reliable)
        with col2:
            prediction_count = len(signals) if signals else 0
            st.metric("Total Signals", prediction_count)
        
        with col3:
            # Count errors could be tracked differently, for now show 0
            # Could enhance to track API failures or other issues
            st.metric("Errors", 0)
        
        with col4:
            if signals and len(signals) > 0:
                try:
                    latest_signal = signals[-1]
                    signal_timestamp = datetime.fromisoformat(latest_signal.get('timestamp', ''))
                    time_ago = datetime.now() - signal_timestamp
                    minutes_ago = int(time_ago.total_seconds() / 60)
                    
                    if minutes_ago < 60:
                        st.metric("Last Signal", f"{minutes_ago}m ago")
                    elif minutes_ago < 1440:  # Less than 24 hours
                        hours_ago = int(minutes_ago / 60)
                        st.metric("Last Signal", f"{hours_ago}h ago")
                    else:
                        days_ago = int(minutes_ago / 1440)
                        st.metric("Last Signal", f"{days_ago}d ago")
                except:
                    st.metric("Last Signal", "N/A")
            else:
                st.metric("Last Signal", "N/A")
    
    def render_latest_signal(self, ticker):
        """Render the latest signal for a ticker."""
        signals = self._load_live_signals(ticker)
        
        if not signals:
            st.warning(f"No signals available for {ticker}")
            return
        
        # Get latest signal
        latest = signals[-1]
        
        # Signal card
        col1, col2, col3 = st.columns([2, 2, 3])
        
        with col1:
            signal_type = latest.get('signal', 'HOLD')
            confidence = latest.get('confidence', 0) * 100
            
            if signal_type == 'BUY':
                st.markdown(f'<p class="buy-signal">🟢 {signal_type}</p>', unsafe_allow_html=True)
            elif signal_type == 'SELL':
                st.markdown(f'<p class="sell-signal">🔴 {signal_type}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="hold-signal">🟡 {signal_type}</p>', unsafe_allow_html=True)
            
            st.metric("Confidence", f"{confidence:.2f}%")
        
        with col2:
            st.metric("Current Price", f"${latest.get('current_price', 0):.2f}")
            pattern = latest.get('pattern_predicted', 'Unknown')
            st.metric("Pattern", pattern)
        
        with col3:
            # Pattern probabilities pie chart
            probs = latest.get('pattern_probabilities', {})
            if probs:
                fig = go.Figure(data=[go.Pie(
                    labels=[k.replace('_', ' ').title() for k in probs.keys()],
                    values=list(probs.values()),
                    hole=.3
                )])
                fig.update_layout(
                    title="Pattern Probabilities",
                    height=250,
                    margin=dict(t=50, b=0, l=0, r=0)
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_history(self, ticker, limit=50):
        """Render signal history chart for a ticker."""
        signals = self._load_live_signals(ticker)
        
        if not signals:
            st.info(f"No signal history for {ticker}")
            return
        
        # Limit signals
        signals = signals[-limit:]
        
        # Convert to dataframe
        df = pd.DataFrame(signals)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['confidence_pct'] = df['confidence'] * 100
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add price trace
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['current_price'],
            name='Price',
            line=dict(color='blue', width=2),
            yaxis='y'
        ))
        
        # Add confidence trace
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['confidence_pct'],
            name='Confidence %',
            line=dict(color='green', width=2, dash='dash'),
            yaxis='y2'
        ))
        
        # Add signal markers
        buy_signals = df[df['signal'] == 'BUY']
        sell_signals = df[df['signal'] == 'SELL']
        
        if not buy_signals.empty:
            fig.add_trace(go.Scatter(
                x=buy_signals['timestamp'],
                y=buy_signals['current_price'],
                mode='markers',
                name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up'),
                yaxis='y'
            ))
        
        if not sell_signals.empty:
            fig.add_trace(go.Scatter(
                x=sell_signals['timestamp'],
                y=sell_signals['current_price'],
                mode='markers',
                name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down'),
                yaxis='y'
            ))
        
        # Update layout with dual y-axes
        fig.update_layout(
            title=f"{ticker} - Price and Confidence History",
            xaxis=dict(title="Time"),
            yaxis=dict(
                title=dict(text="Price ($)", font=dict(color="blue")),
                tickfont=dict(color="blue")
            ),
            yaxis2=dict(
                title=dict(text="Confidence (%)", font=dict(color="green")),
                tickfont=dict(color="green"),
                anchor="x",
                overlaying="y",
                side="right"
            ),
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_pattern_distribution(self, ticker):
        """Render pattern distribution for recent signals."""
        signals = self._load_live_signals(ticker)
        
        if not signals:
            return
        
        # Get recent signals (last 50)
        recent_signals = signals[-50:]
        
        # Count patterns
        pattern_counts = {}
        for signal in recent_signals:
            pattern = signal.get('pattern_predicted', 'Unknown')
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        if pattern_counts:
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(pattern_counts.keys()),
                    y=list(pattern_counts.values()),
                    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'][:len(pattern_counts)]
                )
            ])
            
            fig.update_layout(
                title=f"{ticker} - Pattern Distribution (Last 50 Signals)",
                xaxis_title="Pattern",
                yaxis_title="Count",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_signal_table(self, ticker, limit=10):
        """Render table of recent signals."""
        signals = self._load_live_signals(ticker)
        
        if not signals:
            return
        
        # Get recent signals
        recent = signals[-limit:]
        recent.reverse()  # Most recent first
        
        # Create dataframe
        df = pd.DataFrame(recent)
        
        # Select and format columns
        display_df = pd.DataFrame({
            'Time': pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S'),
            'Signal': df['signal'],
            'Pattern': df['pattern_predicted'],
            'Confidence': (df['confidence'] * 100).apply(lambda x: f"{x:.2f}%"),
            'Price': df['current_price'].apply(lambda x: f"${x:.2f}")
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    def render_overall_stats(self, selected_tickers):
        """Render overall statistics across all tickers."""
        st.subheader("📊 Overall Statistics")
        
        total_predictions = 0
        total_errors = 0
        total_buy = 0
        total_sell = 0
        
        for ticker in selected_tickers:
            status = self._load_live_status(ticker)
            signals = self._load_live_signals(ticker)
            
            if status:
                total_predictions += status.get('prediction_count', 0)
                total_errors += status.get('error_count', 0)
            
            if signals:
                for signal in signals:
                    if signal.get('signal') == 'BUY':
                        total_buy += 1
                    elif signal.get('signal') == 'SELL':
                        total_sell += 1
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            st.metric("Total Errors", total_errors)
        
        with col3:
            st.metric("Buy Signals", total_buy)
        
        with col4:
            st.metric("Sell Signals", total_sell)
    
    def run(self):
        """Main application loop."""
        self.render_header()
        
        # Sidebar controls
        selected_tickers, auto_refresh, refresh_interval = self.render_sidebar()
        
        if not selected_tickers:
            st.info("👈 Please select tickers from the sidebar to view their signals")
            st.markdown("""
            ### 🚀 Getting Started
            
            1. Enter a ticker symbol in the sidebar (e.g., AAPL, GOOGL, TSLA)
            2. Select a model from the dropdown
            3. Click "Start Live Prediction" to begin monitoring
            4. Once started, the ticker will appear in the Active Tickers list
            5. Select it to view real-time signals and charts
            
            ### 📈 Available Features
            
            - **Real-time Signals**: View live BUY/SELL signals with confidence scores
            - **Pattern Recognition**: See which chart patterns are detected
            - **Price History**: Track price movements and signal timing
            - **Pattern Distribution**: Analyze pattern frequency
            - **Auto-refresh**: Keep dashboard updated automatically
            """)
            return
        
        # Overall stats
        self.render_overall_stats(selected_tickers)
        st.markdown("---")
        
        # Render each ticker
        for ticker in selected_tickers:
            st.header(f"📊 {ticker}")
            
            # Status
            self.render_ticker_status(ticker)
            
            # Latest signal
            st.subheader("Latest Signal")
            self.render_latest_signal(ticker)
            
            # Charts in columns
            col1, col2 = st.columns(2)
            
            with col1:
                self.render_signal_history(ticker)
            
            with col2:
                self.render_pattern_distribution(ticker)
            
            # Signal table
            with st.expander("📋 Recent Signals"):
                self.render_signal_table(ticker, limit=20)
            
            st.markdown("---")
        
        # Auto refresh
        if auto_refresh:
            time.sleep(refresh_interval)
            st.rerun()


# Main entry point
if __name__ == "__main__":
    app = DashboardApp()
    app.run()
