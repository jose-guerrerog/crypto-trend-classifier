FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy source files
COPY ./src /app/src
COPY ./src/models/crypto_classifier.pth /app/src/models/crypto_classifier.pth

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]