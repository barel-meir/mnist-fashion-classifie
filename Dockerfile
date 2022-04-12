FROM ubuntu:20.04
RUN apt-get update
RUN apt-get update && apt-get install -y apt-utils
RUN apt-get install -y apt-transport-https
RUN apt update -y 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt install python3-pip -y libgl1 python3-opencv -y
RUN apt-get upgrade python3-pip
COPY requirements.txt ~/requirements.txt
WORKDIR ~
RUN python3 -m pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .
CMD ["/usr/bin/python3", "./flask_hello.py"]
