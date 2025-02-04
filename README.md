# ScoreSage

ScoreSage is an end-to-end machine learning project designed to predict student exam performance based on various factors such as gender, ethnicity, parental education level, lunch type, and test preparation course. The project encompasses data ingestion, data transformation, model training, and prediction pipelines, all integrated into a user-friendly web application.

## Table of Contents

- [Features](#features)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Data Ingestion**: Automatically ingests raw data and splits it into training and testing datasets.
- **Data Transformation**: Preprocesses the data using pipelines for numerical and categorical features.
- **Model Training**: Trains multiple models using GridSearchCV to find the best hyperparameters.
- **Prediction Pipeline**: Provides a pipeline for making predictions based on user input.
- **Web Application**: A Flask-based web app for interactive predictions.

## Directory Structure

```
ScoreSage/
├── logs/
├── data/
│   ├── processed/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── raw/
│   │   └── data.csv
│   └── archive.zip
├── tests/
│   ├── test_model_trainer.py
│   ├── test_predict_pipeline.py
│   ├── test_data_transformation.py
│   ├── test_data_ingestion.py
│   └── __init__.py
├── README.md
├── src/
│   ├── components/
│   │   ├── __pycache__/
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── __init__.py
│   │   └── data_ingestion.py
│   ├── utils/
│   │   ├── __pycache__/
│   │   ├── common.py
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── __pycache__/
│   ├── exception.py
│   ├── pipelines/
│   │   ├── __pycache__/
│   │   ├── predict_pipeline.py
│   │   └── __init__.py
│   ├── __init__.py
│   └── logger.py
├── requirements.txt
├── app/
│   ├── main.py
│   ├── templates/
│   │   ├── home.html
│   │   └── index.html
│   └── __init__.py
├── setup.py
├── models/
│   └── artifacts/
│       ├── preprocessor.pkl
│       └── model.pkl
```
## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/azeebneuron/ScoreSage.git
   cd ScoreSage
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   python app/main.py
   ```

4. **Access the web application**:
   Open your browser and navigate to `http://127.0.0.1:5000/`.

## Usage

1. **Data Ingestion**:
   - The data ingestion process automatically reads the raw data from `data/raw/data.csv` and splits it into training and testing datasets.

2. **Data Transformation**:
   - The data transformation pipeline preprocesses the data, handling missing values, encoding categorical variables, and scaling numerical features.

3. **Model Training**:
   - The model training process evaluates multiple models using GridSearchCV to find the best model based on R² score.

4. **Prediction**:
   - Use the web application to input student details and get predictions for their math scores.

## Testing

To run the tests, use the following command:

```bash
python -m unittest discover tests
```

This will run all the test cases in the `tests/` directory, ensuring that each component of the pipeline works as expected.


**ScoreSage** is designed to be a comprehensive tool for predicting student performance, making it easier for educators and administrators to identify students who may need additional support. Enjoy using ScoreSage!
