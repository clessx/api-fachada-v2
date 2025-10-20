# api_fachada.py
"""
Servicio FastAPI para clasificación de fachadas (válida / no válida)
Modelo: fachada_model_finetuned.keras
Autor: Cristian Yáñez — CorreosChile
"""

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
import numpy as np
from PIL import Image
import io
import uvicorn

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
MODEL_PATH = "models/fachada_model_finetuned.keras"
IMG_SIZE = (224, 224)
CLASSES = ["valida", "no_valida"]


# ============================================================
# CARGA DE MODELO
# ============================================================
print("🚀 Cargando modelo...")
model = load_model(MODEL_PATH)
print("✅ Modelo cargado correctamente.")

# ============================================================
# API FASTAPI
# ============================================================
app = FastAPI(
    title="POD-ML Fachada Service",
    description="API para clasificar imágenes de fachadas como válidas o no válidas",
    version="1.0"
)

@app.get("/")
async def root():
    return {"status": "ok", "message": "POD-ML Fachada Service online"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize(IMG_SIZE)
    x = np.expand_dims(np.array(img), axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    prob_valida = float(preds[0][0])
    label = CLASSES[int(round(prob_valida))]
    confidence = round(prob_valida if label == "valida" else 1 - prob_valida, 4)

    return {"prediction": label, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run("api_fachada:app", host="0.0.0.0", port=8000)
