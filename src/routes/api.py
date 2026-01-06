from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from src.services.data_preprocessor import preprocess_data, preprocess_for_analysis
from src.services.analyzer import analyze_data
from src.services.predictor import train_model, predict_scores
from src.utils.file_handler import process_input
from src.config.settings import API_KEY, THRESHOLD_WEAK_SUBJECT

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")

@router.post("/train", dependencies=[Depends(verify_api_key)])
async def train(file: UploadFile = File(None), json_input: str = Form(None)):
    try:
        data = await process_input(file, json_input)
        features_df, target_df, _, _ = preprocess_data(data, is_training=True)
        metrics = train_model(features_df, target_df)
        return JSONResponse(content={"message": "Model trained successfully", "metrics": metrics})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict(file: UploadFile = File(None), json_input: str = Form(None)):
    try:
        data = await process_input(file, json_input)
        features_df, _, _, subjects_dict = preprocess_data(data, is_training=False)
        results = predict_scores(features_df, subjects_dict)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/analyze", dependencies=[Depends(verify_api_key)])
async def analyze(file: UploadFile = File(None), json_input: str = Form(None), threshold: float = Form(THRESHOLD_WEAK_SUBJECT)):
    
    try:
        data = await process_input(file, json_input)
        features_df, exams_df, subjects_dict = preprocess_for_analysis(data)
        results = analyze_data(features_df, exams_df, subjects_dict)
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")