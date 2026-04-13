# Applied Machine Learning: Model Design, Training and Performance Evaluation

This folder contains implementation of **Inference Latency Prediction for Production Diffusion Model Serving** built on Alibaba's `cluster-trace-v2026-GenAI` dataset (Lin et al., 2025).

GitHub repository can be found at `https://github.com/sam-andaluri/machine-learning-project`.


## Directory Structure

```text
machine-learning-project/
|-- README.md
|-- latency_prediction.ipynb
|-- Machine_Learning_Analysis_Report.md
|-- Machine_Learning_Analysis_Report.pdf
|-- requirements.txt
|-- data/
|   |-- cluster-trace-v2026-GenAI/
|       |-- data_trace_processed.csv
|       |-- qps.csv
|       |-- pod_gpu_duty_cycle_anon.csv
|       |-- model_predict_data_anon.csv
|       |-- pipeline_inference_data_anon.csv
|       `-- queue_size_raw_anon.csv
|-- models/
|   |-- linear_regression.joblib
|   |-- random_forest.joblib
|   |-- xgboost.joblib
|   |-- xgboost_tuned.joblib
|   |-- scaler.joblib
|   |-- feature_columns.txt
|-- figures/
    |-- feature_correlation.png
    |-- predicted_vs_actual.png
    |-- residual_distribution.png
    |-- feature_importance.png
    |-- shap_summary.png
    |-- confusion_matrix.png
    |-- roc_curve.png
    |-- demand_forecast.png
    |-- error_over_time.png
```

## 1. Install `uv`

Install `uv` with the standalone installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or install it with Homebrew on macOS:

```bash
brew install uv
```

Confirm the installation:

```bash
uv --version
```

## 2. Prerequisites

Make sure the following tools are available on your system:

- Python 3.8 or higher
- Jupyter with `nbconvert`
- Pandoc for PDF generation

Quick checks:

```bash
python --version
python -m jupyter nbconvert --version
pandoc --version
```

## 3. Create and Activate a Virtual Environment

From the project folder:

```bash
cd machine-learning-project
uv venv .venv
source .venv/bin/activate
```

## 4. Install Dependencies

Install the pinned dependencies from `requirements.txt`:

```bash
uv pip install -r requirements.txt
```

## 5. Run the Notebook

Open and run `latency_prediction.ipynb` in Jupyter:

```bash
jupyter notebook latency_prediction.ipynb
```

Or using JupyterLab:

```bash
jupyter lab latency_prediction.ipynb
```

Run all cells from top to bottom (Cell > Run All).

## 6. Generate the Report PDF

If you update the report markdown, regenerate the PDF with:

```bash
pandoc Machine_Learning_Analysis_Report.md -o Machine_Learning_Analysis_Report.pdf
```

## 7. Generate Final requirements.txt

After running the notebook, generate the exact package versions:

```bash
pip freeze > requirements.txt
```

## References

Astral. (n.d.). *Installing uv*. uv documentation. Retrieved April 13, 2026, from https://docs.astral.sh/uv/getting-started/installation/

Astral. (n.d.). *Pip interface*. uv documentation. Retrieved April 13, 2026, from https://docs.astral.sh/uv/pip/

Astral. (n.d.). *Using environments*. uv documentation. Retrieved April 13, 2026, from https://docs.astral.sh/uv/pip/environments/

Alibaba Cloud. (n.d.). *Cluster-trace-v2026-GenAI* [Data set]. GitHub. Retrieved April 13, 2026, from https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI

Lin, Y., Wu, S., Luo, S., Xu, H., Shen, H., Ma, C., Shen, M., Chen, L., Xu, C., Qu, L., & Ye, K. (2025). Understanding Diffusion Model Serving in Production: A Top-Down Analysis of Workload, Scheduling, and Resource Efficiency. *Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC '25)*.

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
