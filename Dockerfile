FROM python:3.12-slim-bookworm

# Copy the uv binary from the official uv image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Use the virtual environment that uv creates
ENV PATH="/app/.venv/bin:$PATH"

# Install the exact dependency versions from uv.lock
COPY .python-version pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

# Copy the service code and the trained model
COPY predict.py features.py model.bin ./

EXPOSE 9696

ENTRYPOINT ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]


