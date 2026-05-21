# House Price Prediction

A machine learning project to predict house prices using regression models.

## Project Structure

```
Houce-price-pridiction/
│
├── data/
│   ├── raw/              # Raw dataset files
│   ├── processed/        # Processed dataset files
│
├── notebooks/
│   └── EDA.ipynb         # Exploratory Data Analysis notebook
│
├── src/
│   ├── data_preprocessing.py  # Data cleaning and preprocessing
│   ├── model.py               # Model definitions
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│
├── models/
│   └── model.pkl              # Trained model
│
├── app/
│   └── app.py                 # Flask API
│
├── requirements.txt
└── README.md  
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preparation
Place your raw data in `data/raw/` folder.

### 2. Exploratory Data Analysis
Open `notebooks/EDA.ipynb` in Jupyter to explore the data.

### 3. Training
Run the training script:
```bash
cd src
python train.py
```

### 4. Evaluation
Evaluate model performance:
```bash
cd src
python evaluate.py
```

### 5. API Deployment
Start the Flask API:
```bash
cd app
python app.py
```

Then make predictions via API:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"feature1": 100, "feature2": 200, ...}'
```

## Project Files

- **data_preprocessing.py** - Data loading, cleaning, and feature scaling
- **model.py** - Model class with train/predict/evaluate methods
- **train.py** - Complete training pipeline
- **evaluate.py** - Model evaluation and metrics
- **app.py** - Flask REST API for predictions
- **EDA.ipynb** - Data exploration and visualization

## Models Supported

- Linear Regression
- Random Forest
- Gradient Boosting

## License

MIT License
