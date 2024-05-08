import joblib
import uvicorn

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

with open("elastic_model.pkl", 'rb') as file:
    model = joblib.load(file)


class ModelRequestData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


class Result(BaseModel):
    result: float


@app.get("/health")
def health():
    return JSONResponse(content={"message": "It's alive!"}, status_code=200)


@app.post("/predict", response_model=Result)
def preprocess_data(data: ModelRequestData):
    input_data = data.dict()
    input_df = pd.DataFrame(input_data, index=[0])
    result = model.predict(input_df)[0]
    return Result(result=result)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
