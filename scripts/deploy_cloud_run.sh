#!/usr/bin/env bash
# Deploy StudyBot to Google Cloud Run. Run from the project folder:
#
#   scripts/deploy_cloud_run.sh YOUR_PROJECT_ID [region]
#
# What it does, in order (it asks before the first step that can cost money):
#   1. enables the three Google services it needs
#   2. builds the image IN Google's cloud (no Docker needed on your Mac)
#   3. stores your API key from .env in Secret Manager (never in the image, never printed)
#   4. deploys one small instance with a request limit, and prints the URL
#
# Needs: the gcloud CLI, `gcloud auth login` done, billing enabled on the project, and
# data/processed/chunks.jsonl (make chunks) and .env with LLM_API_KEY=...
set -euo pipefail

PROJECT="${1:?usage: scripts/deploy_cloud_run.sh PROJECT_ID [region]}"
REGION="${2:-europe-west3}"
REPO="studybot"
SERVICE="studybot"
SECRET="studybot-llm-key"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/studybot:$(date +%Y%m%d-%H%M)"

command -v gcloud >/dev/null || { echo "gcloud is not installed: brew install --cask google-cloud-sdk"; exit 1; }
[ -f data/processed/chunks.jsonl ] || { echo "data/processed/chunks.jsonl is missing: run make chunks"; exit 1; }
[ -f Dockerfile ] || { echo "run this from the project folder"; exit 1; }
KEY="$(grep -E '^(LLM_API_KEY|ANTHROPIC_API_KEY)=' .env 2>/dev/null | head -1 | cut -d= -f2- | tr -d "\"' " || true)"
[ -n "$KEY" ] || { echo ".env has no LLM_API_KEY=..."; exit 1; }

echo "Project: ${PROJECT}   Region: ${REGION}"
echo "This will use Google Cloud services (Cloud Build, Artifact Registry, Secret Manager,"
echo "Cloud Run). Small free allowances exist, but billing must be enabled on the project."
read -r -p "Continue? [y/N] " answer
[ "$answer" = "y" ] || { echo "stopped"; exit 0; }

gcloud config set project "$PROJECT"
gcloud services enable run.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com cloudbuild.googleapis.com

gcloud artifacts repositories describe "$REPO" --location "$REGION" >/dev/null 2>&1 ||
  gcloud artifacts repositories create "$REPO" --repository-format=docker --location "$REGION"

echo "Building the image in Google's cloud (10-25 minutes, mostly downloading the models)..."
gcloud builds submit --tag "$IMAGE" --timeout 45m .

# The key goes from .env to Secret Manager through a pipe: it is never printed or stored in
# shell history.
if gcloud secrets describe "$SECRET" >/dev/null 2>&1; then
  printf %s "$KEY" | gcloud secrets versions add "$SECRET" --data-file=-
else
  printf %s "$KEY" | gcloud secrets create "$SECRET" --replication-policy=automatic --data-file=-
fi
NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
gcloud secrets add-iam-policy-binding "$SECRET" \
  --member="serviceAccount:${NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" >/dev/null

# --max-instances 1 and --concurrency 1 bound what a stranger can spend; STUDYBOT_RATE is the
# per-visitor limit per minute.
gcloud run deploy "$SERVICE" --image "$IMAGE" --region "$REGION" \
  --memory 4Gi --cpu 2 --cpu-boost --concurrency 1 --max-instances 1 --timeout 120 \
  --set-secrets "LLM_API_KEY=${SECRET}:latest" \
  --set-env-vars "STUDYBOT_RATE=10" \
  --allow-unauthenticated

echo
echo "Live URL:"
gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
echo "The first visit after idle takes about a minute (the models load)."
echo "To take it offline: gcloud run services delete $SERVICE --region $REGION"
