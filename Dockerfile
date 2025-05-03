# Base Image
FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl

# Working directory inside container
WORKDIR /app

# Install dependencies separately for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project after installing dependencies
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Command to run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
