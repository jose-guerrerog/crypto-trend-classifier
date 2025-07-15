# 🪙 Crypto Trend Classifier API

A simple, Dockerized REST API built with **FastAPI** and **PyTorch** that predicts the short-term trend of a cryptocurrency (either `"UP"` or `"DOWN"`) based on 5 input features.

📍 **Live Demo:**  
👉 [https://crypto-trend-classifier.onrender.com/docs](https://crypto-trend-classifier.onrender.com/docs)

---

## 🚀 Features

- 🔎 Takes 5 numerical features as input
- 📈 Returns trend prediction: `"UP"` or `"DOWN"`
- 🧠 Powered by a PyTorch neural network
- ⚡ FastAPI backend with Swagger docs
- 🐳 Fully Dockerized
- 🌐 Deployed on Render

---

## 📁 Project Structure

```bash
crypto-trend-classifier/
├── data/                     # CSV or input feature data
│   └── btcusdt_1h.csv
├── notebooks/                # EDA and training exploration
│   └── eda.ipynb
├── src/
│   ├── api/
│   │   ├── main.py           # FastAPI app
│   ├── models/
│   │   └── crypto_classifier.pth  # Trained PyTorch model
│   ├── model.py              # CryptoClassifier model definition
│   └── data_loader.py        # (Optional) Feature/data utilities
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
📦 Requirements
Python 3.10+

Install dependencies with:

bash
Copy
Edit
pip install -r requirements.txt
Dependencies include:

fastapi

uvicorn

torch

pydantic

🧪 Example Usage
Request
http
Copy
Edit
POST /predict
Content-Type: application/json

{
  "features": [0.005, 10940.0, 10980.0, 10850.0, 10910.0]
}
Response
json
Copy
Edit
{
  "prediction": "UP"
}
Try it live at 👉 /docs Swagger UI.

🐳 Run Locally with Docker
Build the image
bash
Copy
Edit
docker build -t crypto-classifier-api .
Run the container
bash
Copy
Edit
docker run -d -p 8000:8000 crypto-classifier-api
Then open http://127.0.0.1:8000/docs in your browser.

🌍 Deployment Options
✅ Render (Live Now)
Your app is live at:
🔗 https://crypto-trend-classifier.onrender.com/docs

To deploy:

Push project to GitHub

Create a new Web Service on Render

Use Docker as the deployment method

Set the port to 8000

Done! 🚀

🧠 Model Training
The model was trained using PyTorch (in a notebook or script) and saved with:

python
Copy
Edit
torch.save(model.state_dict(), "crypto_classifier.pth")
This model is then loaded inside the FastAPI app to serve predictions.