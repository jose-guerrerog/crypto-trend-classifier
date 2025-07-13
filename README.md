# 🪙 Crypto Trend Classifier API

A simple FastAPI-based REST API that predicts the short-term trend of a cryptocurrency using a PyTorch model.

## 🚀 Features

- Takes 5 numerical input features
- Returns a prediction: `"UP"` or `"DOWN"`
- Built with FastAPI and PyTorch
- Dockerized and ready for deployment
- Live on Render 🚀

📍 **Live Demo:**  
👉 [https://crypto-trend-classifier.onrender.com/docs](https://crypto-trend-classifier.onrender.com/docs)

---

## 📦 Requirements

- Python 3.10+
- `requirements.txt` includes:
  - `fastapi`
  - `uvicorn`
  - `torch`
  - `pydantic`

---

## 📁 Project Structure

```
crypto-trend-classifier/
├── app/
│   ├── main.py          # FastAPI app
│   └── model.pth        # Trained PyTorch model
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 🧪 Example Usage

### Request
```json
POST /predict
Content-Type: application/json

{
  "features": [0.005, 10940.0, 10980.0, 10850.0, 10910.0]
}
```

### Response
```json
{
  "prediction": "UP"
}
```

---

## 🐳 Run Locally with Docker

### Build the image
```bash
docker build -t crypto-classifier-api .
```

### Run the container
```bash
docker run -d -p 8000:8000 crypto-classifier-api
```

### Access locally
Open: http://127.0.0.1:8000/docs

---

## 🌍 Deployment Options

### ✅ Render (Live Now)
Your app is deployed at: 🔗 https://crypto-trend-classifier.onrender.com/docs

Steps:
1. Push code to GitHub
2. Create a Web Service on Render
3. Choose Docker deployment method
4. Set port to 8000

## 🧠 Model Training

The model was trained separately using PyTorch and saved with:

```python
torch.save(model.state_dict(), "model.pth")
```
