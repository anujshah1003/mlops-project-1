FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install pandas numpy scikit-learn mlflow

CMD ["python", "src/train.py"]