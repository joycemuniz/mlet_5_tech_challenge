FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/   # 👈 isso resolve

EXPOSE 5000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "5000"]
