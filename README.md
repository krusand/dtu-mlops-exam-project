# exam_project

Contains code for exam project for DTU course MLOps

## Project structure

The directory structure of the project looks like this:
```txt
├── .github/                  # Github actions and dependabot
│   ├── dependabot.yaml
│   └── workflows/
│       └── tests.yaml
├── configs/                  # Configuration files
├── data/                     # Data directory
│   ├── processed
│   └── raw
├── dockerfiles/              # Dockerfiles
│   ├── api.Dockerfile
│   └── train.Dockerfile
├── docs/                     # Documentation
│   ├── mkdocs.yml
│   └── source/
│       └── index.md
├── models/                   # Trained models
├── notebooks/                # Jupyter notebooks
├── reports/                  # Reports
│   └── figures/
├── src/                      # Source code
│   ├── project_name/
│   │   ├── __init__.py
│   │   ├── api.py
│   │   ├── data.py
│   │   ├── evaluate.py
│   │   ├── models.py
│   │   ├── train.py
│   │   └── visualize.py
└── tests/                    # Tests
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_data.py
│   └── test_model.py
├── .gitignore
├── .pre-commit-config.yaml
├── LICENSE
├── pyproject.toml            # Python project file
├── README.md                 # Project README
├── requirements.txt          # Project requirements
├── requirements_dev.txt      # Development requirements
└── tasks.py                  # Project tasks
```


Created using [mlops_template](https://github.com/SkafteNicki/mlops_template),
a [cookiecutter template](https://github.com/cookiecutter/cookiecutter) for getting
started with Machine Learning Operations (MLOps).

## Project Description

**Overall goal of the project**
<br>
The goal of the project is to use deep learning and computer vision techniques to solve a classification task of predicting the facial expression shown in an image. Each image will be classified into one of seven emotion categories: fear, anger, happy, sad, neutral, disgust, and surprise.

**What framework are you going to use**
<br>
Since we chose an image-based classification problem, we plan to use PyTorch as the main deep learning framework for training and implementing our own models. In addition, we will use the Hugging Face ecosystem to access and fine-tune pretrained vision models (e.g., Vision Transformers). For systematic hyperparameter tuning, we plan to use Optuna with Bayesian optimization.

**How do you intend to include the framework into your project**
<br>
We plan on utilizing one of the main strengths of the Hugging Face ecosystem, which is that it provides easy access to a wide range of pretrained architectures that can be fine-tuned for specific tasks. As a starting point, we will implement and train simple custom baseline models in PyTorch and use them as reference points. From there, we will fine-tune pretrained models and evaluate whether they provide improvements in accuracy and generalization. Finally, Optuna will be integrated into the training pipeline to optimize key hyperparameters (e.g., learning rate, batch size, dropout, and weight decay).

**What data are you going to run on** 
<br>
We are using the Kaggle dataset [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013). The dataset contains grayscale facial images of size 48×48 pixels, organized into separate training and test splits. The training set includes 28,709 examples, and the test set includes 3,589 examples, distributed across the seven emotion classes. The dataset was chosen because it is well-known, relatively straightforward, and small enough to train models efficiently within a limited timeframe, while still being challenging enough to compare multiple architectures.

**What deep learning models do you expect to use**
<br>
We intend to begin with two custom baselines:
* a CNN designed specifically for small low-resolution images, and
* a simpler ANN baseline created by flattening the image input.
* These baselines will provide a foundation for comparison. After that, we plan to fine-tune several pretrained architectures, including ResNet, EfficientNet, MobileNet and a Vision Transformer (ViT), to test whether transfer learning improves performance over the baselines.