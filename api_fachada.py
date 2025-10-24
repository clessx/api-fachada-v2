# api_fachada.py
"""
Servicio FastAPI para clasificación de fachadas (válida / no válida)
Modelo: fachada_model_finetuned.keras
Autor: Cristian Yáñez — CorreosChile
"""

from fastapi import FastAPI, File, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
from io import BytesIO
import uvicorn

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
MODEL_PATH = "models/fachada_model_finetuned.keras"
IMG_SIZE = (224, 224)
THRESHOLD = 0.35  # ✅ mismo criterio que el modelo de paquetes

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
    version="2.2"
)

# Permitir CORS (útil para pruebas desde otras apps)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "status": "ok",
        "message": "POD-ML Fachada Service online 🚀",
        "model": MODEL_PATH,
        "threshold": THRESHOLD
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe una imagen y devuelve la predicción: válida o no válida.
    """
    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        # === PREDICCIÓN ===
        pred = float(model.predict(img_array, verbose=0)[0][0])

        # === LÓGICA INVERTIDA ===
        # Si el modelo da valor alto → no_valida
        # Si el modelo da valor bajo → valida
        if pred >= THRESHOLD:
            clase = "no_valida"
            confidence = round(pred, 4)
        else:
            clase = "valida"
            confidence = round(1 - pred, 4)

        return {
            "prediction": clase,
            "confidence": confidence
        }

    except Exception as e:
        return {"error": f"❌ Error procesando imagen: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run("api_fachada:app", host="0.0.0.0", port=8000)
