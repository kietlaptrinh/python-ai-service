from fastapi import FastAPI
from src.routes.api import router as api_router
from src.config.settings import PORT

app = FastAPI(
    title="AI Student Performance Prediction Service",
    description="A Machine Learning-based Python service for analyzing and predicting student exam performance.",
    version="1.0.0"
)

app.include_router(api_router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)