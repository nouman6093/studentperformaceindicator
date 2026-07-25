# Student Performance Predictor

A machine learning web application built with **FastAPI**, **Scikit-Learn**, **Jinja2**, and **Docker**. This app predicts a student's math score based on various demographic factors and academic performance metrics.

--

##  Project Structure

```text
├── artifacts/
│   ├── model.pkl            # Trained ML model
│   └── preprocessor.pkl     # Fitted Scikit-Learn ColumnTransformer
├── templates/
│   └── index.html           # Jinja2 HTML template with custom CSS
├── train.py                 # Script to train model and export artifacts
├── main.py                  # FastAPI server and endpoints
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker container configuration
└── .dockerignore            # Files ignored by Docker build

```

---

##  Features

* **Machine Learning Pipeline:** Uses `ColumnTransformer` with `OneHotEncoder` and `StandardScaler` paired with `LinearRegression`.
* **FastAPI Backend:** Lightweight asynchronous Python web framework handling web forms (`python-multipart`).
* **Clean UI:** Responsive, modern CSS form rendered with Jinja2 templates.
* **Docker Ready:** Includes a lightweight `python:3.11-slim` container build configuration.

---

## Quick Start (Local Setup)

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-folder-name>

```

### 2. Create and Activate Virtual Environment

```bash
# On Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# On Windows
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Train the Model

Ensure `stud.csv` is in the root directory, then run:

```bash
python train.py

```

*This will generate the required `model.pkl` and `preprocessor.pkl` files inside the `artifacts/` folder.*

### 5. Start the Application

```bash
uvicorn main:app --reload

```

Open your browser and visit:

👉 **`[http://127.0.0.1:8000](http://127.0.0.1:8000)`**

---

## Docker Deployment

### 1. Build the Docker Image

```bash
docker build -t student-performance-app .

```

### 2. Run the Docker Container

```bash
docker run -p 8000:8000 student-performance-app

```

Access the application at **`http://localhost:8000`**.

---

## Deployment to AWS

To push and deploy this application on **AWS**:

1. **Push to Amazon ECR:**
```bash
aws ecr get-login-password --region <your-region> | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.<your-region>.amazonaws.com
docker tag student-performance-app:latest <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/student-performance-app:latest
docker push <your-account-id>.dkr.ecr.<your-region>.amazonaws.com/student-performance-app:latest

```


2. **Deploy Container:** Launch via **AWS App Runner** or **AWS ECS (Fargate)** pointing directly to the image hosted in Amazon ECR.