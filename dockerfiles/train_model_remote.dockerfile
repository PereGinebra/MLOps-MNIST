# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY corrupt_mnist/ corrupt_mnist/
COPY .dvc/config ./dvc/config

WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN dvc pull
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "corrupt_mnist/train_model.py"]