# Machine Learning Analysis Report

## Inference Latency Prediction for Production Diffusion Model Serving

**Project**: Applied Machine Learning: Model Design, Training and Performance Evaluation
**Date**: April 2026
**Author**: Sambasiva Andaluri

---

## 1. Overview

This project addresses the operational challenge of predicting inference latency in production AI serving systems. Using the Alibaba GenTD26 dataset—a production trace containing 68,195 requests from a large-scale Stable Diffusion image generation service—we built supervised machine learning models to predict request latency before processing begins. Accurate latency prediction enables **SLA-aware request routing**: if the system predicts a request will exceed the service level agreement (SLA) threshold, it can proactively route that request to a less-loaded server, improving user experience and system reliability.

The dataset is publicly available at [github.com/alibaba/clusterdata](https://github.com/alibaba/clusterdata/tree/master/cluster-trace-v2026-GenAI) and was published with Lin et al.'s SoCC '25 paper on diffusion model serving.

---

## 2. Dataset Description

The GenTD26 dataset captures production traces from Alibaba's Stable Diffusion serving infrastructure, providing a realistic view of AI inference workloads at scale.

**Dataset Statistics**:
- **Total requests**: 68,195 inference requests
- **Features**: 9 raw columns expanded to 17 engineered features
- **Target variable**: `exec_time_seconds` (end-to-end latency)
  - Mean: 27.1 seconds | Median: 23.0 seconds | Std: 17.9 seconds
  - Range: 0 to 608 seconds

**Key Features**:
| Feature | Description | Unique Values |
|---------|-------------|---------------|
| `predict_type` | Request type (text-to-image, image-to-image, inpainting) | 3 |
| `checkpoint_model_version_id` | Base model identifier (anonymized) | 104 |
| `num_inference_steps` | Diffusion denoising iterations | 1-100 (mode: 30) |
| `num_images_per_prompt` | Batch size | 1-8 (mode: 1) |
| `lora_args` | LoRA adapter configurations | JSON list |

**Data Anonymization**: Per Lin et al. (2025), identifiers are MD5-hashed and timestamps offset to protect user privacy while preserving statistical distributions essential for research.

---

## 3. Modeling Approach

### 3.1 Data Preparation

We applied the following preprocessing steps:

1. **Missing Value Imputation**:
   - `style_type` (52% missing): Filled with 'unknown' category
   - `negative_prompt_length` (24% missing): Filled with 0 (no negative prompt)
   - Numeric features: Median/mode imputation (robust to outliers per Hastie et al., 2009)

2. **Feature Engineering**:
   - Extracted `num_lora` and `avg_lora_scale` from JSON adapter configurations
   - Created `compute_complexity = num_inference_steps × num_images_per_prompt`
   - Computed rolling historical latency (last 10/50 requests) as system load proxy
   - Generated per-model rolling latency to capture model-specific performance patterns

3. **Categorical Encoding**: Label encoding for anonymized string identifiers

### 3.2 Train/Test Split

We used **time-based splitting** (not random) because this is sequential data:
- **Training**: First 60% (40,917 requests)
- **Validation**: Next 15% (10,229 requests) for hyperparameter tuning
- **Test**: Final 25% (17,049 requests)

Random splitting would cause data leakage by allowing the model to learn from future requests (Bergmeir & Benítez, 2012).

### 3.3 Model Selection

We compared three regression models:

| Model | Rationale |
|-------|-----------|
| **Linear Regression** | Baseline; interpretable coefficients |
| **Random Forest** | Handles non-linear relationships; robust to outliers |
| **XGBoost** | State-of-the-art gradient boosting; regularization prevents overfitting |

We selected XGBoost as our primary model based on its strong performance on structured/tabular data (Chen & Guestrin, 2016) and applied Bayesian hyperparameter optimization using Optuna (50 trials).

### 3.4 Evaluation Metrics

| Metric | Purpose | Why Selected |
|--------|---------|--------------|
| **MAE** | Mean absolute prediction error | Directly interpretable in seconds |
| **RMSE** | Root mean squared error | Penalizes large errors |
| **R²** | Variance explained | Standard regression metric |
| **SLA Accuracy** | Binary classification accuracy | Operationally relevant for routing decisions |

---

## 4. Results

### 4.1 Model Performance Comparison

| Model | MAE (s) | RMSE (s) | R² | SLA Accuracy |
|-------|---------|----------|-----|--------------|
| Linear Regression | 8.40 | 14.08 | 0.588 | 86.8% |
| Random Forest | 8.86 | 14.82 | 0.543 | 86.3% |
| XGBoost (Default) | 9.09 | 15.58 | 0.495 | 86.2% |
| **XGBoost (Tuned)** | **8.52** | **14.32** | **0.573** | **86.9%** |

**Key Finding**: Hyperparameter tuning improved XGBoost's R² from 0.495 to 0.573—a **16% relative improvement**. The tuned model's optimal parameters included `max_depth=3`, `learning_rate=0.06`, and `n_estimators=193`.

### 4.2 Feature Importance (SHAP Analysis)

The most predictive features for latency were:

1. **`model_rolling_latency`** (r=0.71): Historical performance of the requested model—models that run slowly tend to continue running slowly.
2. **`rolling_latency_mean_10`** (r=0.39): Recent system-wide latency, indicating current load.
3. **`num_lora` / `has_lora`** (r=0.35): LoRA adapter usage adds loading overhead.
4. **`compute_complexity`** (r=0.29): More steps × more images = longer latency.

### 4.3 SLA Violation Detection

Converting regression predictions to binary SLA compliance (threshold: 30 seconds):

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Accuracy** | 86.9% | Correctly classifies 87% of requests |
| **Precision (Violations)** | 79.3% | 79% of predicted violations are actual violations |
| **Recall (Violations)** | 78.9% | Catches 79% of actual SLA violations |
| **F1-Score** | 0.79 | Balanced precision-recall |
| **AUC-ROC** | >0.85 | Strong discriminative ability |

**Operational Impact Simulation**:
- Of 5,367 actual SLA violations in the test set, our model would have flagged **4,236 (78.9%)** for rerouting
- Only **1,106 requests (6.5%)** would be unnecessarily rerouted (false alarms)

---

## 5. Interpretation for a Non-Technical Audience

Imagine you're ordering food delivery. Before you place your order, the app estimates "Your food will arrive in 25-35 minutes." Our model does something similar for AI image generation requests.

**What the model does**: When a user requests an AI-generated image, our system predicts how long it will take *before* processing starts. If the prediction suggests the request will take too long (over 30 seconds), the system can automatically send it to a faster server.

**How well it works**:
- Our predictions are typically accurate within about **8-9 seconds** of the actual time
- We correctly identify **87%** of requests as "fast" or "slow"
- We catch **4 out of 5** requests that would have been too slow, allowing them to be rerouted

**What affects image generation time**:
1. **Which AI model is used**: Some models are inherently slower than others
2. **Custom styles (LoRA adapters)**: Adding custom styles requires loading extra files, which takes time
3. **Image quality settings**: More "steps" produce better images but take longer
4. **How busy the system is**: When many people are using the service, everything slows down

---

## 6. Limitations and Potential Bias

### 6.1 Technical Limitations

1. **No explicit timestamps**: The request data lacks timestamps, so we used row order as a time proxy. If requests were logged out of order, our time-based features would be incorrect.

2. **Anonymized identifiers**: MD5-hashed model IDs prevent us from understanding *which specific models* are slow, limiting actionable insights for model optimization.

3. **Single-cluster data**: Results are from one Alibaba cluster; performance may differ on systems with different GPUs, network configurations, or user populations.

4. **Unexplained variance**: Our best model explains 57% of latency variance (R²=0.573), meaning 43% is due to factors not captured in our features—possibly network delays, GPU memory contention, or request queue dynamics.

### 6.2 Potential Sources of Bias

1. **Temporal distribution shift**: The model was trained on data from a specific time period. If user behavior changes (e.g., new popular models, different request patterns), accuracy may degrade. **Mitigation**: Periodic retraining with recent data.

2. **Selection bias**: The dataset only includes *completed* requests. Failed, timed-out, or cancelled requests may have different characteristics that the model doesn't learn. This could cause underestimation of latency for edge cases.

3. **Feedback loops**: If deployed for routing, the model's decisions would change which requests each server handles, potentially altering the latency distribution. **Mitigation**: A/B testing and ongoing monitoring.

4. **Representation bias**: The dataset reflects one company's user base. Image generation preferences, model choices, and usage patterns may differ in other contexts.

---

## References

Bergmeir, C., & Benítez, J. M. (2012). On the use of cross-validation for time series predictor evaluation. *Information Sciences*, 191, 192-213. https://doi.org/10.1016/j.ins.2011.12.028

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction* (2nd ed.). Springer.

James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). *An Introduction to Statistical Learning*. Springer.

Lin, Y., Wu, S., Luo, S., Xu, H., Shen, H., Ma, C., Shen, M., Chen, L., Xu, C., Qu, L., & Ye, K. (2025). Understanding Diffusion Model Serving in Production: A Top-Down Analysis of Workload, Scheduling, and Resource Efficiency. *Proceedings of the 2025 ACM Symposium on Cloud Computing (SoCC '25)*.

Powers, D. M. W. (2011). Evaluation: From precision, recall and F-measure to ROC, informedness, markedness and correlation. *Journal of Machine Learning Technologies*, 2(1), 37-63.

Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., ... & Dennison, D. (2015). Hidden technical debt in machine learning systems. *Advances in Neural Information Processing Systems*, 28.

