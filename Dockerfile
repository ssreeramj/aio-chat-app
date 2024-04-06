# temp stage
FROM python:3.12.2 as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir ./wheels -r requirements.txt


# final stage
FROM python:3.12.2-slim

WORKDIR /app

COPY --from=builder  /app/wheels ./wheels
COPY --from=builder /app/requirements.txt .

RUN pip install --no-cache ./wheels/*

# Copy modified langfuse script
COPY ./.venv/Lib/site-packages/langfuse/extract_model.py /usr/local/lib/python3.12/site-packages/langfuse/
COPY ./src .


EXPOSE 80

# Add health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 CMD curl --fail http://localhost:80 || exit 1

# do not change the arguments
ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]