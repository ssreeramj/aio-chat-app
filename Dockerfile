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

COPY ./src .
COPY .env .

# do not change the arguments
ENTRYPOINT ["chainlit", "run", "app.py", "--host=0.0.0.0", "--port=80", "--headless"]