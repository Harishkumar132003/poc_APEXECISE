FROM python:3.12-slim

WORKDIR /app

COPY prophet/requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY prophet ./prophet

EXPOSE 5001

CMD ["python", "src/app.py"]
