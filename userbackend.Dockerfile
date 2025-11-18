FROM python:3.12-slim

# Set working directory
WORKDIR /app


# Copy requirements
COPY userbackend/requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY userbackend /app/userbackend

# Expose port
EXPOSE 5005

# Entry command
CMD ["python", "userbackend/userquery_service.py"]
