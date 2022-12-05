FROM python:3.8.10-slim
EXPOSE 8501
WORKDIR /app
COPY . .
RUN pip3 install --no-cache-dir -r requirements.txt
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]