# Lab 3: Penguins Classification with XGBoost and FastAPI

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/aidi-2004-ai-enterprise/lab3_Khadija_Ramzi.git
   cd lab3_<first_name>_<last_name>
   
# Step-by-Step Instructions to Run the App

## Prerequisites
1. Ensure you have Python 3.10 or newer installed on your system.
2. Open Git Bash in the project directory and install required packages using [uv](https://github.com/astral-sh/uv):
   ```bash
   uv pip install -r requirements.txt
   ```
   If `requirements.txt` is not present, install the following manually:
   ```bash
   uv pip install pandas seaborn scikit-learn matplotlib numpy joblib
   ```

## Running the Application
1. Open Git Bash in the project root directory.
2. To run the main app using uv, execute:
   ```bash
   uv venv
   uv pip install -r requirements.txt
   uv pip run python app/main.py
   ```
   Or, to run training and testing scripts:
   ```bash
   uv pip run python train.py
   uv pip run python test.py
   ```

## Expected Results
* The application will process the penguins dataset, encode categorical variables, and display dataset statistics.
* You will see a histogram plot of the target variable (species) with a normal distribution curve and mean/std deviation lines.
* Model training and evaluation results will be printed in the terminal, including metrics such as classification report and confusion matrix.
* Log files will be generated in the `app/logs/` directory, and model artifacts will be saved in `app/data/`.


## Demonstration
* See demo.mp4 for:
    1. Valid predictions
    2. Input validation errors
    3. App running from Swagger UI


## Troubleshooting
* If you encounter missing package errors, ensure all required packages are installed.
* For any issues, check the log file at `app/logs/app.log` for details.