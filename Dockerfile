FROM python:3.9-slim

# Pre-install common ML dependencies
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements-light.txt .
RUN pip install --no-cache-dir -r requirements-light.txt

COPY . .
EXPOSE 8000
CMD ["uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "$PORT"]