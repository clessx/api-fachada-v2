# ===============================
# 🐳 DOCKERFILE — POD-ML SERVICE
# ===============================

FROM python:3.10-slim

# Evitar prompts interactivos
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Crear carpeta del servicio
WORKDIR /app

# Copiar e instalar dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del proyecto
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando para iniciar el servicio
CMD ["uvicorn", "api_fachada:app", "--host", "0.0.0.0", "--port", "8000"]