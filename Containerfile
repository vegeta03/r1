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
ENV API_KEY=""
ENV PROVIDER="groq"
ENV BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_ID="llama-3.1-70b-versatile"
ENV CONTEXT_WINDOW=8000

# Run cli.py when the container launches
CMD ["python", "cli.py"]

# podman build -t r1-cli .
# podman run -it --rm -e API_KEY=your_api_key r1-cli
# podman run -it --rm -e PROVIDER=groq -e API_KEY=your_api_key -e BASE_URL=https://api.groq.com/openai/v1 -e MODEL_ID=llama-3.1-70b-versatile -e CONTEXT_WINDOW=8000 r1-cli