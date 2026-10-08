FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml .
RUN pip install -e . 
COPY notebooks/pyproject.toml .
RUN pip install -e . 

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
