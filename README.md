# Smart Job–Candidate Matching System (MLOps Level 2 Project)

![Candidate-Role Matching](assets/hero_poster.png)

**Term Project Theme**: Developing a Resilient, High-Cardinality Prediction Service  
**Maturity Target**: MLOps Level 2 (CI/CD Pipeline Automation)  
**Status**: <span style="color:green; font-weight:bold">✅ RELEASED (100% Compliant)</span>

---

## 🚀 Executive Summary: Operational Efficiency & Risk Mitigation

> *"Build an ML system like a company would: Automate it. Control it. Monitor it."*

This system addresses the critical business need for scalable, automated recruitment matching. Unlike legacy manual models, this platform is designed for **Operational Resilience**:

![Operational Resilience](assets/ocean_resilience.jpg)

*   **Fail Safely**: If model confidence drops, the system triggers an **Algorithmic Fallback** instead of serving a wrong prediction.
*   **Self-Healing**: Kubernetes **Rolling Updates** ensure zero downtime deployment.
*   **Auditability**: Full **Model Governance** via MLflow ensures every decision is traceable.

**SDLC Flow**:  
`Planning` → `Requirements` → `Design` → `Development` → `Testing` → `Deployment` → `Maintenance`

---

## 👥 Team Structure & Role Ownership

| Member Name | ID | Primary Responsibilities |
| :--- | :--- | :--- |
| **Anas Brkji** | 220901178 | **DevOps Engineer & Business Analyst** (CI/CD Pipeline, Infrastructure) |
| **Misem Mohamed** | 220901646 | **Project Manager & Data Scientist** (Design Patterns, Governance) |
| **Ahmed A.S Abubreik** | 220901525 | **Test Engineer & Business Analyst** (Unit Testing, Requirements) |
| **Ahmed N.F AlHayek** | 229911872 | **Data Engineer** (Data Pipeline, Feature Hashing) |
| **Mohammed Ali** | 229912086 | **MLOps SRE Specialist** (Monitoring, Resilience, Kubernetes) |
| **Eman Mohammed** | 229910904 | **ML Engineer** (Optimization, Hyperparameter Tuning) |
| **Ele Ben Messaoud** | 220911597 | **ML Engineer** (Model Development, Training Logic) |

---

## 1. System Overview & Architecture

This system automates the lifecycle of machine learning models from training to deployment using a **Microservices Architecture**.

> [!IMPORTANT]
> **Tool Justification: Prefect**  
> We selected **Prefect** (over Airflow/Kubeflow) as mandated by Requirement II.2. Its **dynamic workflow** capabilities allow us to "fail fast" during the CI/CD loop and handle data-driven changes more intelligently than time-based schedulers.

### 🏗️ High-Level Architecture
```mermaid
graph LR
    A[Data Source] -->|Ingest| B(Prefect Workflow);
    B -->|Train & Tune| C{Validation};
    C -->|Pass| D[MLflow Registry];
    C -->|Fail| E[Alert Team];
    D -->|Promote| F[Staging Env];
    F -->|Deploy| G[FastAPI Service];
    G -->|Monitor| H[Prometheus];
```

---

## 2. Key Components

### 🔄 The Automated Pipeline (`workflow.py`)
The pipeline is the "brain" of the MLOps system. It orchestrates sequential tasks:
1.  **Static Analysis**: Runs `pylint` and `safety` checks. Fails immediately if code quality is low.
2.  **Unit Testing**: Executes `pytest` suites to validate feature engineering logic.
3.  **Training**: Trains an Ensemble Model (`VotingClassifier`) on the dataset.
4.  **Registration**: Saves the model artifact to the MLflow Registry, tagged with the commit SHA.

### 🛡️ The Inference Service (`inference_service.py`)
A robust REST API designed for high availability.
*   **Endpoint**: `POST /predict`
*   **Payload**: JSON (Skills, Qualification, Experience)
*   **Resilience**:
    *   **Drift Detection**: Calculates a rolling average of confidence scores.
    *   **Fallback**: Automatically flags uncertain predictions (`Review_Required`) preventing bad hiring advice.

### ☸️ Infrastructure (`Kubernetes`)
The system is cloud-native ready.
*   **Deployment**: Defined in `deployment.yaml` with **RollingUpdate** strategy.
*   **Redundancy**: 2 Pod Replicas ensure strictly zero downtime even if one pod crashes.

---

## 3. Mandatory Compliance Verification

This document confirms the completion of the Job Role Prediction project, meeting all mandatory MLOps Level 2 compliance requirements.

### ✅ A. Automated CI/CD Execution
The pipeline is fully automated via GitLab CI/CD (`.gitlab-ci.yml`) and Prefect (`workflow.py`).
-   **Static Analysis**: `flake8`, `pylint` (score > 7.0), and `safety` scans are enforced.
-   **Testing**: Fast unit tests for hashing and component tests for serving are executed on every build.
-   **Fail-Fast**: Any standard violation or test failure immediately halts the pipeline.

### ✅ B. Model Governance & Tracking (MLflow)
-   **Experiment Tracking**: Best hyperparameters logged from `RandomizedSearchCV`.
-   **Registry**: Models automatically transitioned to the 'Staging' stage after validation.

### ✅ C. Serving & Monitoring (FastAPI + Prometheus)
-   **Stateless API**: Exposed via FastAPI on port 8000.
-   **Monitoring**: Real-time metrics available at `/metrics`.
-   **Algorithmic Fallback**: Low-confidence predictions trigger human-in-the-loop fallback.

---

## 4. Technical Design Patterns

We have integrated advanced design patterns to address the "High-Cardinality" and "Resilience" constraints.

### � Data & Problem Representation
> *Addressing the complexity of unstructured skill data.*

*   **Pattern: HASHED FEATURE**
    *   **Context**: The `skills` column contains free-text with high cardinality.
    *   **Solution**: Implemented `HashingVectorizer` (1000 features) in `train_pipeline.py`.
    *   **Impact**: Fixed memory usage regardless of vocabulary growth.

*   **Pattern: REBALANCING**
    *   **Context**: Role distribution is highly skewed.
    *   **Solution**: Applied `RandomOverSampler` within the pipeline.
    *   **Impact**: Improved F1-Score on rare job roles.

### 🛡️ Resilient Serving Architecture
> *Ensuring reliability in a production environment.*

*   **Pattern: STATELESS SERVING FUNCTION**
    *   **Implementation**: `inference_service.py` handles requests without maintaining local state.
    *   **Benefit**: Enables seamless horizontal autoscaling.

*   **Pattern: ALGORITHMIC FALLBACK**
    *   **Validation**: `if confidence < FALLBACK_THRESHOLD: return "Generalist_Candidate_Review_Required"`
    *   **Benefit**: Prevents low-confidence predictions from reaching the end-user.

*   **Pattern: CONTINUOUS EVALUATION (CME)**
    *   **Implementation**: Prometheus metrics (`model_confidence_avg`) exposed at `/metrics`.
    *   **Benefit**: Real-time detection of Concept Drift.

---

## 5. How to Reproduce (Operations)

### 1. Installation
```bash
git clone <repo_url>
pip install -r requirements.txt
```

### 2. Run End-to-End Demo
```bash
python run_system_e2e.py
```
*This single command builds the pipeline, trains the model, launches the service, and sends test traffic to verify fallback and monitoring.*

### 3. Monitoring Access
*   **MLflow UI**: `http://localhost:5000`
*   **Prometheus Metrics**: `http://localhost:8000/metrics`

---

## 🤝 Contributing (Git Workflow)
1.  `git status`
2.  `git add .`
3.  `git commit -m "Update"`
4.  `git push`
