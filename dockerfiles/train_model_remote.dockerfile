# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY corrupt_mnist/ corrupt_mnist/

WORKDIR /
RUN pip install dvc --no-cache-dir
RUN pip install "dvc[gs]" --no-cache-dir
RUN dvc init --no-scm
RUN dvc remote add -d remote_storage gs://corrupt_mnist_data_bucket/
RUN dvc pull
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "corrupt_mnist/train_model.py"]