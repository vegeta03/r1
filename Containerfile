# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV GROQ_API_KEY=""
ENV GROQ_MODEL_ID=""

# Run cli.py when the container launches
CMD ["python", "cli.py"]

# podman build -t g1-cli .
# podman run -it --rm -e GROQ_API_KEY=your_api_key -e GROQ_MODEL_ID=llama-3.1-70b-versatile g1-cli
