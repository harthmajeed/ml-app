FROM python:3.10-slim
WORKDIR /app

# copy packaging metadata and requirements first for better layer caching
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# copy the rest of the project
COPY . /app
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]