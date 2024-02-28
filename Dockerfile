FROM python:3.11

EXPOSE 8501

WORKDIR /shared/
COPY ./shared/ ./

WORKDIR /app/
COPY ./app/ ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run", "main.py", "--browser.serverAddress=localhost", "--server.enableCORS=false", "--server.enableXsrfProtection=false" ]