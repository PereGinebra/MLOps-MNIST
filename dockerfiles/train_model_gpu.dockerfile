# Base image
FROM nvcr.io/nvidia/pytorch:23.12-py3

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY config.yaml config.yaml
COPY corrupt_mnist/ corrupt_mnist/
COPY data/ data/

WORKDIR ../workspace/
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "corrupt_mnist/train_model.py"]