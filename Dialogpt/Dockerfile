FROM python:3.9
WORKDIR /dialogpt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","dialogpt.py"]
