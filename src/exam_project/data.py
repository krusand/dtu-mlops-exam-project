import kagglehub
import os

if __name__ == "__main__":
    #Set KAGGLEHUB_CACHE environment variable
    os.environ["KAGGLEHUB_CACHE"] = "data/raw/"

    #Download latest version
    path = kagglehub.dataset_download("msambare/fer2013")

    print("Path to dataset files:", path)
