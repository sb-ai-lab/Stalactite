FROM python:3.10.5-slim

WORKDIR /opt

COPY pyproject.toml .

RUN pip install --upgrade pip poetry
# RUN pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu

RUN poetry config virtualenvs.create false && poetry lock --no-update && poetry install --no-root

# RUN pip install torch==2.0.1+cu117 --index-url https://download.pytorch.org/whl/cu117

COPY experiments/src/ /opt/experiments/src/

COPY experiments/run-experiment /opt/experiments/run-experiment
RUN chmod +x /opt/experiments/run-experiment

ENTRYPOINT ["bash", "/opt/experiments/run-experiment"]
