# # Use a base image with Miniconda
# FROM continuumio/miniconda3

# Set working directory
WORKDIR /app

# Copy your README and any necessary files
COPY . .

# # Install git (if not already installed)
# RUN apt-get update && apt-get install -y git

# Default command
CMD ["/bin/bash"]

