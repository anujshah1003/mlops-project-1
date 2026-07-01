FROM python:3.10

WORKDIR /app

COPY . .

# RUN pip install pandas numpy scikit-learn mlflow

# CMD ["python", "src/train.py"]

RUN pip install fastapi uvicorn joblib numpy

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]

# docker build -t mlops-api .

# docker run -p 8000:8000 mlops-api
# http://localhost:8000/docs


