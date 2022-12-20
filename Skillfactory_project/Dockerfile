FROM python:3.9-slim-buster
# RUN apt-get update && apt-get install wget -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install libgomp1 -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /usr/local/app_ml
COPY . /usr/local/app_ml
RUN python3 -m pip install -r requirements.txt
EXPOSE 8015
CMD ["uvicorn", "src.skill_api.app:app", "--host", "0.0.0.0", "--port", "8015"]