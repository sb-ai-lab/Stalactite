FROM python:3.10.5-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /opt

COPY ./pyproject.toml /opt/pyproject.toml
COPY ./README.md /opt/README.md

RUN pip install poetry

RUN poetry config virtualenvs.create false && poetry lock --no-update
RUN poetry install --no-root # TODO double installation fix??

COPY ./stalactite /opt/stalactite

RUN poetry install

WORKDIR /opt/stalactite

# docker build -f ./docker/grpc-base.dockerfile -t grpc-base:latest .
