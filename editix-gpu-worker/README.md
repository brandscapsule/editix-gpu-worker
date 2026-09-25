# Editix Ai — Worker GPU (RunPod)

Programme qui tourne sur RunPod Serverless : dérush IA (transcription + silences,
répétitions, hésitations), normalisation vidéo et vérification santé.

## Ce que contient ce dossier

- `handler.py` — la logique du worker (opérations : `health`, `derush`, `normalize`)
- `requirements.txt` — dépendances Python
- `Dockerfile` — l'image Docker à déployer sur RunPod

## Étape 3 — Envoyer le programme sur GitHub (5 min)

1. Créez un dépôt **privé** nommé par exemple `editix-gpu-worker` sur github.com
   (New repository → Private → Create, sans README).
2. Décompressez `editix-gpu-worker.zip`, puis dans le dossier :
   « upload existing files » sur GitHub (bouton *uploading an existing file*)
   et déposez les 3 fichiers (`handler.py`, `requirements.txt`, `Dockerfile`).
3. Commit direct sur `main`.

## Étape 4 — Déployer sur RunPod (10 min)

1. Console RunPod → **Serverless** → **New Endpoint**.
2. Source : connectez votre compte GitHub et choisissez le dépôt
   `editix-gpu-worker` (RunPod construit l'image Docker tout seul).
   Sinon, construisez l'image vous-même :
   `docker build -t votreuser/editix-worker . && docker push votreuser/editix-worker`
   et utilisez cette image.
3. GPU : **L4** ou **A10** (24 Go). Activez un volume disque de 20 Go (cache Whisper).
4. Réglages : **Min workers = 0**, **Max workers = 1** (au début), **Idle timeout = 5 min**,
   **Execution timeout = 1800 s**.
5. Créez une clé API : **Settings → API Keys**, limitée à cet endpoint.
6. Récupérez :
   - l'**Endpoint ID** (visible sur la page de l'endpoint),
   - la **clé API** (`rpa_...`).

## Étape 5 — Connecter à Editix Ai

Collez l'Endpoint ID et la clé API RunPod quand Lovable vous le demande
(secrets `RUNPOD_ENDPOINT_ID` et `RUNPOD_API_KEY`). Le bouton de test de
l'app vérifiera que le GPU répond.

## Test manuel (optionnel)

```bash
curl -X POST "https://api.runpod.ai/v2/<ENDPOINT_ID>/run" \
  -H "Authorization: Bearer <CLÉ_API>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"operation": "health"}}'
```
