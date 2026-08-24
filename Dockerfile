# 1. Start with a lightweight Linux environment running Python 3.11

FROM python:3.11-slim

# 2. Set the working directory inside the container

WORKDIR /app

# 3. Copy the clean requirements file into the container

COPY requirements.txt .

# 4. Install the Python libraries

RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy your entire project into the container

COPY . .

# 6. Expose the port Uvicorn will run on

EXPOSE 8000

# 7. The exact command to start the server

CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]