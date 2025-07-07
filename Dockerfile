# base image 
FROM python:3.11-slim

# setting up  working directory
WORKDIR /app

# dependencies installation 
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Default command
CMD ["python", "train_mtl_demo.py"]
