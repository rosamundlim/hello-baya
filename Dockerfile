FROM python:3.11.7

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "src/hellobaya.py", "--server.port=8501", "--server.address=0.0.0.0"]
