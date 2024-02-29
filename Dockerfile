FROM python:3.11

EXPOSE 8501

WORKDIR /config/
COPY ./config/ ./

WORKDIR /app/
COPY ./app/ ./

RUN pip install -r requirements.txt

WORKDIR /

ENTRYPOINT [ "streamlit", "run", "app/main.py", "--browser.serverAddress=localhost", "--server.enableCORS=false", "--server.enableXsrfProtection=false" ]