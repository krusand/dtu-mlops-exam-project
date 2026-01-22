#!/bin/bash
set -e   # exit if any command fails

#Check for GOOGLE_APPLICATION_CREDENTIALS is set
if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ]; then
    echo "GOOGLE_APPLICATION_CREDENTIALS not set. Assuming running on GCP with default credentials."
else
    echo "Using GOOGLE_APPLICATION_CREDENTIALS=$GOOGLE_APPLICATION_CREDENTIALS"
fi

# Step Inside App
cd /app

# Make sure the remote is set
echo "About to uv run dvc remote add..."
uv run dvc remote add -f gcsremote gs://dtu-mlops-exam-project-data || true

# Pull the dataset
echo "About to uv run dvc pull..."
uv run dvc pull data.dvc -r gcsremote --verbose

# Run training
echo "About to uv run train script..."
uv run src/exam_project/train.py "$@"