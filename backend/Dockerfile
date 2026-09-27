FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /workspace/backend
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/app ./app
COPY experiments/optimization /workspace/experiments/optimization
COPY experiments/retrieval /workspace/experiments/retrieval
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
