# Crypto Trend Classifier API

A simple REST API that predicts if a cryptocurrency will go **UP** or **DOWN** using 5 input features.

**Live Demo:** https://crypto-trend-classifier.onrender.com/docs

## Features

- Predicts crypto trends: "UP" or "DOWN"
- Built with FastAPI and PyTorch
- Dockerized and ready to deploy
- Interactive API documentation

## Project Structure

```
crypto-trend-classifier/
├── data/
│   └── btcusdt_1h.csv
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── api/
│   │   └── main.py
│   ├── models/
│   │   └── crypto_classifier.pth
│   ├── model.py
│   └── data_loader.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Send a POST request to `/predict` with 5 features:

```json
{
  "features": [0.005, 10940.0, 10980.0, 10850.0, 10910.0]
}
```

**The 5 features represent:**
1. **Price change percentage** (0.005) - Percentage change in price
2. **Open price** (10940.0) - Opening price of the time period
3. **High price** (10980.0) - Highest price during the time period
4. **Low price** (10850.0) - Lowest price during the time period
5. **Close price** (10910.0) - Closing price of the time period

Response:
```json
{
  "prediction": "UP"
}
```

## Run Locally

**With Docker:**
```bash
docker build -t crypto-classifier-api .
docker run -p 8000:8000 crypto-classifier-api
```

**Without Docker:**
```bash
cd src/api
uvicorn main:app --reload --port 8000
```

Open http://127.0.0.1:8000/docs in your browser.

## Deployment

The app is deployed on Render. To deploy your own:

1. Push to GitHub
2. Create a new Web Service on Render
3. Use Docker deployment
4. Set port to 8000

## Model

The PyTorch model is trained and saved as `crypto_classifier.pth`. It takes 5 numerical features and predicts the trend direction.