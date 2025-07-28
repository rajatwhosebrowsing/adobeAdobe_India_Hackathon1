# CORRECTED DOCKERFILE

# Stage 1: The Builder Stage (with Python)
# We use a standard Python image to install our dependencies first.
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Copy the requirements file and install libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: The Final Stage (with Ollama)
# Now we start from a clean Ollama image.
FROM ollama/ollama:latest

# Copy the installed Python libraries from the 'builder' stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy all your project files (main.py, Modelfile, .gguf file, etc.)
WORKDIR /app
COPY . .

# Create a custom Ollama model named "hackathon" from your file
RUN ollama serve & sleep 5 && \
    ollama create hackathon -f /app/Modelfile && \
    pkill ollama

# Pre-download the embedding model
RUN python3 -c "from langchain_huggingface import HuggingFaceEmbeddings; print('Downloading embedding model...'); HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'); print('Download complete.')"

# Set the command to run when the container starts
CMD ["python3", "final1.py"]
