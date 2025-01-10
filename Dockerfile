# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the maintainer label
LABEL maintainer="Ali Akram <akramsystems@gmail.com>"

# Copy and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy the application code
COPY ./src /src
WORKDIR /src

# Set the Python path
ENV PYTHONPATH=/src

# Run the start script
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
