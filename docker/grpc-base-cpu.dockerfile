FROM python:3.10.5-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /opt

COPY ./pyproject.toml /opt/pyproject.toml
COPY ./poetry.lock /opt/poetry.lock
COPY ./README.md /opt/README.md

RUN pip install poetry

RUN poetry config virtualenvs.create false
# RUN poetry lock --no-update # TODO do we need to lock?
RUN poetry run pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

RUN poetry install --no-root

COPY ./stalactite /opt/stalactite

RUN poetry install --only-root

WORKDIR /opt/stalactite

# docker build -f ./docker/grpc-base.dockerfile -t grpc-base:latest .
