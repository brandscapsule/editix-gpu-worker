FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pré-télécharge les modèles (Whisper + détourage vidéo) pour un démarrage à froid plus rapide
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('small', device='cpu', compute_type='int8')"
RUN python -c "import torch; torch.hub.load('PeterL1n/RobustVideoMatting', 'resnet50', trust_repo=True)"

COPY handler.py .

CMD ["python", "-u", "handler.py"]
