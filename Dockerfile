FROM python:3.10-slim

WORKDIR /app

COPY src/ src/
COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "5000"]