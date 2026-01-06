import json
import pandas as pd
from fastapi import UploadFile, HTTPException

async def process_input(file: UploadFile = None, json_input: str = None) -> dict:
  
    if file:
        if file.filename.endswith('.csv'):
            df = pd.read_csv(await file.read())
            data = df.to_dict(orient='records')
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(await file.read())
            data = df.to_dict(orient='records')
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Use CSV, Excel, or JSON.")
        raise HTTPException(status_code=400, detail="File input not fully supported for nested data. Use JSON.")
    elif json_input:
        data = json.loads(json_input)
    else:
        raise HTTPException(status_code=400, detail="No input provided. Provide file or JSON.")

    # Validate input structure
    if 'results' not in data or 'exams' not in data or 'subjects' not in data:
        raise HTTPException(status_code=400, detail="Invalid JSON structure. Missing 'results', 'exams', or 'subjects'.")

    return data