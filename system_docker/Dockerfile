FROM python:3.9
WORKDIR /system
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python","system.py"]
