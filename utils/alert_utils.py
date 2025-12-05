import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import logging
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class AlertSystem:
    """
    Handles sending alerts via email and Telegram for trading signals.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def send_email_alert(self, subject: str, message: str, to_email: Optional[str] = None) -> bool:
        """
        Send email alert.
        
        Args:
            subject: Email subject
            message: Email message body
            to_email: Recipient email (optional, uses config if not provided)
            
        Returns:
            True if email sent successfully
        """
        # Email alerts disabled to reduce terminal clutter
        return True
        
        try:
            if not all([Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD]):
                self.logger.warning("Email credentials not configured")
                return False
            
            to_email = to_email or Config.EMAIL_TO
            if not to_email:
                self.logger.warning("No recipient email address configured")
                return False
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = Config.EMAIL_USERNAME
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Connect to server and send email
            server = smtplib.SMTP(Config.EMAIL_SMTP_SERVER, Config.EMAIL_SMTP_PORT)
            server.starttls()
            server.login(Config.EMAIL_USERNAME, Config.EMAIL_PASSWORD)
            
            text = msg.as_string()
            server.sendmail(Config.EMAIL_USERNAME, to_email, text)
            server.quit()
            
            self.logger.info(f"Email alert sent to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email alert: {str(e)}")
            return False
    
    def send_telegram_alert(self, message: str, chat_id: Optional[str] = None) -> bool:
        """
        Send Telegram alert.
        
        Args:
            message: Message to send
            chat_id: Telegram chat ID (optional, uses config if not provided)
            
        Returns:
            True if message sent successfully
        """
        # Telegram alerts disabled to reduce terminal clutter
        return True
        
        try:
            if not Config.TELEGRAM_BOT_TOKEN:
                self.logger.warning("Telegram bot token not configured")
                return False
            
            chat_id = chat_id or Config.TELEGRAM_CHAT_ID
            if not chat_id:
                self.logger.warning("Telegram chat ID not configured")
                return False
            
            url = f"https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/sendMessage"
            
            payload = {
                'chat_id': chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            
            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Telegram alert sent to chat {chat_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {str(e)}")
            return False
    
    def format_signal_message(self, signal_data: Dict[str, Any], ticker: str) -> str:
        """
        Format trading signal data into alert message.
        
        Args:
            signal_data: Signal data from TradingSignalGenerator
            ticker: Stock ticker symbol
            
        Returns:
            Formatted message string
        """
        try:
            timestamp = signal_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            signal = signal_data['signal']
            confidence = signal_data['confidence']
            current_price = signal_data['current_price']
            
            message = f"""
🚨 <b>Trading Signal Alert</b> 🚨

<b>Ticker:</b> {ticker}
<b>Signal:</b> {signal}
<b>Confidence:</b> {confidence:.2%}
<b>Current Price:</b> ${current_price:.2f}
<b>Time:</b> {timestamp}
"""
            
            if 'predicted_price' in signal_data:
                predicted_price = signal_data['predicted_price']
                expected_change = signal_data['expected_change']
                message += f"""
<b>Predicted Price:</b> ${predicted_price:.2f}
<b>Expected Change:</b> {expected_change:.2%}
"""
            
            if 'probabilities' in signal_data:
                probs = signal_data['probabilities']
                message += f"""
<b>Probabilities:</b>
• Buy: {probs['buy']:.2%}
• Hold: {probs['hold']:.2%}
• Sell: {probs['sell']:.2%}
"""
            
            # Add signal emoji
            signal_emoji = {
                'BUY': '🟢',
                'SELL': '🔴',
                'HOLD': '🟡'
            }
            
            message = f"{signal_emoji.get(signal, '⚪')} {message}"
            
            return message.strip()
            
        except Exception as e:
            self.logger.error(f"Error formatting signal message: {str(e)}")
            return f"Trading signal: {signal_data.get('signal', 'UNKNOWN')} for {ticker}"
    
    def send_signal_alert(self, signal_data: Dict[str, Any], ticker: str, 
                         send_email: bool = True, send_telegram: bool = True) -> Dict[str, bool]:
        """
        Send trading signal alert via configured channels.
        
        Args:
            signal_data: Signal data from TradingSignalGenerator
            ticker: Stock ticker symbol
            send_email: Whether to send email alert
            send_telegram: Whether to send Telegram alert
            
        Returns:
            Dictionary with success status for each channel
        """
        try:
            message = self.format_signal_message(signal_data, ticker)
            subject = f"Trading Signal: {signal_data['signal']} for {ticker}"
            
            results = {}
            
            if send_email:
                results['email'] = self.send_email_alert(subject, message)
            
            if send_telegram:
                results['telegram'] = self.send_telegram_alert(message)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error sending signal alert: {str(e)}")
            return {'email': False, 'telegram': False}
    
    def send_system_alert(self, message: str, alert_type: str = "INFO") -> Dict[str, bool]:
        """
        Send system status or error alert.
        
        Args:
            message: Alert message
            alert_type: Type of alert (INFO, WARNING, ERROR)
            
        Returns:
            Dictionary with success status for each channel
        """
        try:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            alert_emoji = {
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌'
            }
            
            formatted_message = f"""
{alert_emoji.get(alert_type, 'ℹ️')} <b>System Alert</b>

<b>Type:</b> {alert_type}
<b>Time:</b> {timestamp}

<b>Message:</b>
{message}
"""
            
            subject = f"System Alert: {alert_type}"
            
            results = {}
            results['email'] = self.send_email_alert(subject, formatted_message)
            results['telegram'] = self.send_telegram_alert(formatted_message)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error sending system alert: {str(e)}")
            return {'email': False, 'telegram': False}


class AlertManager:
    """
    Manages alert frequency and prevents spam.
    """
    
    def __init__(self):
        self.last_alerts = {}  # ticker -> timestamp
        self.min_interval = 300  # 5 minutes minimum between alerts for same ticker
        self.alert_system = AlertSystem()
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, Config.LOG_LEVEL))
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def should_send_alert(self, ticker: str, signal: str) -> bool:
        """
        Check if alert should be sent based on timing and signal significance.
        
        Args:
            ticker: Stock ticker
            signal: Trading signal
            
        Returns:
            True if alert should be sent
        """
        try:
            now = datetime.now()
            
            # Check if enough time has passed since last alert for this ticker
            if ticker in self.last_alerts:
                time_since_last = (now - self.last_alerts[ticker]).total_seconds()
                if time_since_last < self.min_interval:
                    self.logger.debug(f"Skipping alert for {ticker}, too soon since last alert")
                    return False
            
            # Don't send alerts for HOLD signals unless specifically configured
            if signal == "HOLD":
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking alert timing: {str(e)}")
            return False
    
    def send_signal_alert(self, signal_data: Dict[str, Any], ticker: str) -> bool:
        """
        Send signal alert if conditions are met.
        
        Args:
            signal_data: Signal data
            ticker: Stock ticker
            
        Returns:
            True if alert was sent
        """
        try:
            signal = signal_data['signal']
            
            if not self.should_send_alert(ticker, signal):
                return False
            
            # Send alert
            results = self.alert_system.send_signal_alert(signal_data, ticker)
            
            # Update last alert time if at least one channel succeeded
            if any(results.values()):
                self.last_alerts[ticker] = datetime.now()
                self.logger.info(f"Alert sent for {ticker}: {signal}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in alert manager: {str(e)}")
            return False
