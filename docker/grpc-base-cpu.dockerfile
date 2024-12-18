FROM python:3.10.5-slim

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

WORKDIR /opt

COPY ./pyproject.toml /opt/pyproject.toml
COPY ./poetry.lock /opt/poetry.lock
COPY ./README.md /opt/README.md

RUN pip install poetry

RUN poetry config virtualenvs.create false
RUN poetry run pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
# RUN poetry run pip install phe==1.5.0  # Uncomment to run phe-based examples
RUN poetry install --no-root

COPY ./stalactite /opt/stalactite
COPY ./examples /opt/examples

RUN poetry install --only-root

WORKDIR /opt/stalactite
ENV GIT_PYTHON_REFRESH="quiet"
LABEL framework="stalactite"
COPY ./plugins /opt/plugins
