FROM apache/airflow:2.8.0-python3.10
COPY src/stalactite-0.1.0-py3-none-any.whl /dist/

RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" "/dist/stalactite-0.1.0-py3-none-any.whl"
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torchsummary
    #
