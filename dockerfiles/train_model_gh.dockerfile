# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY corrupt_mnist/ corrupt_mnist/
COPY .dvc/ .dvc/

WORKDIR /
RUN --mount=type=cache,target=~/.cache/pip pip install dvc
RUN --mount=type=cache,target=~/.cache/pip pip install 'dvc[gdrive]'
RUN dvc config core.no_scm True
RUN dvc pull
RUN --mount=type=cache,target=~/.cache/pip pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENTRYPOINT ["python", "-u", "corrupt_mnist/train_model.py"]