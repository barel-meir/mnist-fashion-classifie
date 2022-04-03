FROM python:3.8-slim-buster
RUN mkdir /app
WORKDIR /app
# COPY templates .
# COPY models .
# COPY flask_hello.py .
# COPY inference.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY . .
CMD [ "python3", "flask_hello.py"]

