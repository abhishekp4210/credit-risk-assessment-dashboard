# Credit Risk Assessment Dashboard

A Streamlit dashboard that estimates loan default risk from applicant and loan details. It uses a trained machine learning pipeline and presents risk probabilities alongside model performance and SHAP/LIME explanations. The project also includes its dataset, trained model, and generated analysis reports.

## Run the dashboard

Python 3.10 or newer is recommended.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard uses the pre-trained model in `models/credit_model.joblib` and supporting charts and tables in `reports/`.

## Train the model again

The source dataset is `data/credit_risk_dataset.csv`. To rerun training and regenerate the model and reports:

```bash
python train.py
```

Training can take several minutes. `train.py` also accepts options such as `--data`, `--cost-fn`, `--cost-fp`, and `--cv`; run `python train.py --help` to see them.

## Project files

- `app.py` — Streamlit dashboard
- `data_pipeline.py` — data cleaning and preprocessing
- `train.py` — model training, calibration, threshold selection, and explainability reports
- `test_basic.py` — basic self-checks
- `data/` — credit risk dataset
- `models/` — trained model bundle used by the dashboard
- `reports/` — model metrics, comparisons, and visualizations

This project is for demonstration and learning; its predictions should not be used as the sole basis for real lending decisions.
