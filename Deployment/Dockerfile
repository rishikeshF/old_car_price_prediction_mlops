FROM python:3.9-slim

RUN pip install -U pip 
RUN pip install pipenv 

WORKDIR /app

COPY [ "Pipfile", "Pipfile.lock", "./" ] 

RUN pipenv install --system --deploy 

COPY [ "predict.py", "model/model.pkl", "./" ]
COPY [ "preprocess/preprocess.bin", "./" ]

EXPOSE 9696

ENTRYPOINT [ "waitress-serve", "--listen=*:9696", "predict:app" ]
