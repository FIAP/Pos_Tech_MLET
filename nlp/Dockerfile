FROM python:3.10.0

## System Dependencies
RUN apt-get update && apt-get upgrade -y && apt-get clean

# Remove unused files to make the image smaller
RUN apt-get autoremove -yqq --purge \
    && apt-get clean

WORKDIR /nlp

COPY . /nlp

ENV PYTHONPATH "${PYTHONPATH}:/nlp"

ENV PYTHONPATH "${PYTHONPATH}:/nlp/src"

RUN pip install -r requirements.txt

RUN python -m spacy download pt_core_news_sm

RUN python -m ipykernel install --user --name=nlp-env --display-name "nlp-env"

