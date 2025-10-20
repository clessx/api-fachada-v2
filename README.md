# POD-ML Fachada Service

Servicio API basado en FastAPI para clasificar imágenes de fachadas como **válidas** o **no válidas**.  
Modelo utilizado: `fachada_model_finetuned.keras`

---

## 🚀 Endpoints

| Método | Ruta        | Descripción |
|--------|-------------|--------------|
| GET    | `/`         | Estado del servicio |
| POST   | `/predict`  | Recibe una imagen JPG/PNG y devuelve predicción |

Ejemplo de respuesta:
```json
{
  "prediction": "valida",
  "confidence": 0.9823
}
