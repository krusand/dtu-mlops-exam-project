# Running Vertex AI Job

This markdown provides a description of how to run model-training with Vertex AI for our application. The job can be run straight away (Step 4.) provided the previous Steps 1-3. have been executed at least once (respectively, these steps are: uploading the docker image, creating a service account, and writing a custom job.yaml with the wandb API key). 

### 1) Build docker image and push to GCP artifact registry
```bash
docker build -f dockerfiles/train_vertex.dockerfile -t europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/trainvtx:latest --platform linux/amd64 --push .
```
If running the image *locally* on a Mac remove the ```--platform linux/amd64``` flag.

### 2) Enable Vertex AI, and create service account for training
```bash
gcloud services enable aiplatform.googleapis.com
```

```bash
gcloud iam service-accounts create vertexai-training-sa \
  --display-name="VertexAI Training Job Service Account for DTU MLOps"

gcloud projects add-iam-policy-binding decent-seeker-484209-j2 \
  --member="serviceAccount:vertexai-training-sa@decent-seeker-484209-j2.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

gcloud projects add-iam-policy-binding decent-seeker-484209-j2 \
  --member="serviceAccount:vertexai-training-sa@decent-seeker-484209-j2.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"
```

### 3) Configure the ```vertex_ai_job.yaml``` with ```WANDB_API_KEY```
Providing secrets to a Vertex AI job is non-trivial; we suggest the user creates a custom ```vertex_ai_job.yaml``` and insert their ```WANDB_API_KEY``` using the provided template: ```vertex_ai_job_template.yaml```.
```bash
worker_pool_specs:
  - machine_spec:
      machine_type: n1-standard-4
    replica_count: 1
    container_spec:
      image_uri: europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/trainvtx:latest
      env:
        - name: AIP_MODEL_DIR
          value: gs://dtu-mlops-exam-project-data/models/vertex_training/
        - name: WANDB_API_KEY
          value: YOUR_SECRET_HERE
      #args:
      #  - models=my_model
      #  - trainer.max_epochs=10
service_account: vertexai-training-sa@decent-seeker-484209-j2.iam.gserviceaccount.com
```

### 4) Run job with custom ```.yaml```
```bash
gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=vertexai-training-job \
  --config=vertex_ai_job.yaml 
```
The trained model is saved to the ```AIP_MODEL_DIR```

---

# Appendix

### A) Run local image locally
```bash
docker run  --rm \
  --env-file $(pwd)/.env \
  -v $(pwd)/decent-seeker-484209-j2-8707d48bfe74.json:/app/key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  --name trainvtxcontainer trainvtx:latest
```

### B) Run remote image locally
```bash
docker run  --rm \
  --env-file $(pwd)/.env \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/decent-seeker-484209-j2-8707d48bfe74.json:/app/key.json:ro \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/key.json \
  --name trainvtxcontainer europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/trainvtx:latest
```

### C) Train via Cloud Run
```bash
gcloud run jobs create my-training-job \
  --image=europe-west1-docker.pkg.dev/decent-seeker-484209-j2/myartifactregistry/trainvtx:latest \
  --region=europe-west1 \
  --service-account=my-training-job-sa@decent-seeker-484209-j2.iam.gserviceaccount.com \
  --set-env-vars="AIP_MODEL_DIR=gs://dtu-mlops-exam-project-data/models/vertex_training/,WANDB_API_KEY=YOUR_SECRET_HERE" \
  --memory=16Gi \
  --cpu=4 \
  --max-retries=0

gcloud run jobs execute my-training-job \
  --region=europe-west1
```