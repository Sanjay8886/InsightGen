# Use a Python base image
FROM python:3.11-slim

# Set environment variables for non-buffered output
ENV PYTHONUNBUFFERED 1
# Set the application entry point (main.py file, with 'app' being the Flask instance name inside)
ENV FLASK_APP main.py

# Set the working directory inside the container
WORKDIR /app

# --- Production Optimization Start ---

# Install system dependencies required for complex Python libraries
# libsqlite3-dev is needed for sqlite-utils
# gcc/g++ are needed for building packages like pandas, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsqlite3-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# --- Production Optimization End ---

# Copy the requirements file and install dependencies first
# This leverages Docker's build cache
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Clean up build dependencies to minimize the final image size. 
# We remove the compiler (gcc) and related tools after pip install is complete.
RUN apt-get purge -y gcc libsqlite3-dev && apt-get autoremove -y

# Copy the rest of the application files
# This includes main.py and any other necessary files
COPY . .

# Expose the port where the application runs
EXPOSE 5001

# The CMD to run the application using Gunicorn for production stability and performance.
# -w 4: starts 4 worker processes
# main:app refers to the Flask instance named 'app' inside 'main.py'
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "-w", "4", "main:app"]
