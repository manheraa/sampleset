FROM python:3.12.0-slim

WORKDIR /CHATBOT
COPY . .

COPY ./requirements.txt ./
RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]