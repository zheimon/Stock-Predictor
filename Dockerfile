FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models logs

# Set environment variables
ENV PYTHONPATH=/app
ENV ENVIRONMENT=production

# Expose port for dashboard
EXPOSE 8501

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Start the live prediction service in background\n\
if [ "$ENABLE_LIVE_PREDICTION" = "true" ]; then\n\
    echo "Starting live prediction service..."\n\
    python scripts/live_prediction.py --model_name "$MODEL_NAME" --ticker "$TICKER" --interval "$INTERVAL" &\n\
fi\n\
\n\
# Start the dashboard\n\
echo "Starting dashboard..."\n\
streamlit run dashboard/dashboard_app.py --server.address 0.0.0.0 --server.port 8501\n\
' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/app/start.sh"]
