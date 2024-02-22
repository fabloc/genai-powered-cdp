FROM python:3.11

EXPOSE 8080
WORKDIR /app

COPY ./app/ ./

RUN pip install -r requirements.txt

ENTRYPOINT [ "streamlit", "run", "__main__.py", "--server.port=8080", "--server.address=0.0.0.0" ]
