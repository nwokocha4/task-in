FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==9.1.1 pytest-json-ctrf==0.5.2 numpy==1.26.4 scipy==1.12.0
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin sandbox
COPY . /tests/
RUN chmod +x /tests/test.sh && chmod -R go-rwx /tests
RUN mkdir -p /app
