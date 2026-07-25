import pickle
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Setup HTML templates
templates = Jinja2Templates(directory="templates")

# Load trained preprocessor and model
with open("artifacts/preprocessor.pkl", "rb") as f:
    preprocessor = pickle.load(f)

with open("artifacts/model.pkl", "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
def predict(
    request: Request,
    gender: str = Form(...),
    race_ethnicity: str = Form(...),
    parental_level_of_education: str = Form(...),
    lunch: str = Form(...),
    test_preparation_course: str = Form(...),
    reading_score: float = Form(...),
    writing_score: float = Form(...)
):
    data = pd.DataFrame([{
        "gender": gender,
        "race_ethnicity": race_ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "reading_score": reading_score,
        "writing_score": writing_score
    }])

    # Preprocess & Predict
    scaled_data = preprocessor.transform(data)
    pred = model.predict(scaled_data)[0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": round(float(pred), 2)}
    )