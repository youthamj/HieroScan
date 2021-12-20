FROM python:3.6-slim
COPY ./app.py /deploy/
COPY ./classify.py /deploy/
COPY ./app.py /deploy/
COPY ./classify.py /deploy/
COPY ./detect.py /deploy/
COPY ./preprocess.py /deploy/
COPY ./segmentation.py /deploy/
COPY ./translation.py /deploy/
COPY ./visualize.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./assets/ /deploy/assets/

WORKDIR /deploy/
RUN pip install -r requirements.txt
EXPOSE 8080
ENTRYPOINT ["python", "app.py"]