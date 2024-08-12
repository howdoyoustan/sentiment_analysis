# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container 
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

COPY . .

# Expose port 8501 (the default Streamlit port)
EXPOSE 8501


# Run streamlit app when the container launches
CMD ["streamlit", "run", "app.py"]