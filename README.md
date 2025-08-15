# MLflow Complete Reference Card

## Overview

MLflow is an open-source platform for managing the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

## Core Components

### 1. MLflow Tracking

Track experiments, parameters, metrics, and artifacts.

#### Basic Tracking Setup

```python
import mlflow
import mlflow.sklearn
from mlflow import log_metric, log_param, log_artifacts

# Set tracking URI (optional)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")
```

#### Run Context Management

```python
# Method 1: Context manager (recommended)
with mlflow.start_run():
    mlflow.log_param("alpha", 0.1)
    mlflow.log_metric("rmse", 0.786)
    
# Method 2: Manual start/end
run = mlflow.start_run()
mlflow.log_param("alpha", 0.1)
mlflow.end_run()

# Method 3: Active run
mlflow.start_run()
# ... logging code ...
mlflow.end_run()
```

#### Logging Functions

```python
# Parameters (hyperparameters, configs)
mlflow.log_param("learning_rate", 0.01)
mlflow.log_params({"batch_size": 32, "epochs": 100})

# Metrics (model performance)
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metrics({"precision": 0.92, "recall": 0.88})

# Step-wise metrics (for tracking over epochs)
for epoch in range(100):
    mlflow.log_metric("loss", loss_value, step=epoch)

# Artifacts (files, models, plots)
mlflow.log_artifact("model.pkl")
mlflow.log_artifacts("output_dir")

# Text
mlflow.log_text("Some important note", "notes.txt")

# Dictionary as JSON
mlflow.log_dict({"key": "value"}, "config.json")
```

#### Tags and Notes

```python
# Set tags
mlflow.set_tag("model_type", "random_forest")
mlflow.set_tags({"version": "1.0", "team": "data_science"})

# Add notes
mlflow.set_tag("mlflow.note.content", "This is a baseline model")
```

### 2. MLflow Models

Standardized model packaging and deployment.

#### Model Flavors

```python
# Scikit-learn
import mlflow.sklearn
mlflow.sklearn.log_model(model, "model")
loaded_model = mlflow.sklearn.load_model("runs:/{}/model".format(run_id))

# PyTorch
import mlflow.pytorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow/Keras
import mlflow.tensorflow
import mlflow.keras
mlflow.keras.log_model(model, "model")

# XGBoost
import mlflow.xgboost
mlflow.xgboost.log_model(model, "model")

# LightGBM
import mlflow.lightgbm
mlflow.lightgbm.log_model(model, "model")

# Statsmodels
import mlflow.statsmodels
mlflow.statsmodels.log_model(model, "model")

# Spark ML
import mlflow.spark
mlflow.spark.log_model(model, "model")

# Custom Python function
import mlflow.pyfunc
mlflow.pyfunc.log_model("model", python_model=custom_model)
```

#### Custom Model Example

```python
import mlflow.pyfunc

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        return self.model.predict(model_input)

# Log custom model
with mlflow.start_run():
    wrapped_model = ModelWrapper(trained_model)
    mlflow.pyfunc.log_model(
        "custom_model", 
        python_model=wrapped_model,
        registered_model_name="MyCustomModel"
    )
```

#### Model Signature

```python
from mlflow.models.signature import infer_signature
from mlflow.types.schema import Schema, ColSpec

# Infer signature automatically
signature = infer_signature(X_train, y_pred)

# Manual signature definition
input_schema = Schema([
    ColSpec("double", "feature1"),
    ColSpec("double", "feature2"),
    ColSpec("string", "category")
])
output_schema = Schema([ColSpec("double")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log with signature
mlflow.sklearn.log_model(model, "model", signature=signature)
```

### 3. MLflow Projects

Reproducible ML code packaging.

#### MLproject File

```yaml
name: My Project
conda_env: conda.yaml

entry_points:
  main:
    parameters:
      alpha: {type: float, default: 0.1}
      l1_ratio: {type: float, default: 0.1}
    command: "python train.py {alpha} {l1_ratio}"
    
  validate:
    parameters:
      model_uri: string
    command: "python validate.py {model_uri}"
```

#### Running Projects

```python
# Run local project
mlflow.run(".", parameters={"alpha": 0.5})

# Run from GitHub
mlflow.run(
    "https://github.com/user/repo.git",
    parameters={"alpha": 0.5}
)

# Run specific entry point
mlflow.run(".", entry_point="validate", parameters={"model_uri": "runs:/abc/model"})
```

### 4. MLflow Model Registry

Central model store for lifecycle management.

#### Registering Models

```python
# Register during logging
mlflow.sklearn.log_model(
    model, 
    "model",
    registered_model_name="ProductionModel"
)

# Register existing run
model_uri = "runs:/{}/model".format(run_id)
mlflow.register_model(model_uri, "ProductionModel")

# Register with version description
result = mlflow.register_model(
    model_uri,
    "ProductionModel",
    description="Model trained on latest dataset"
)
```

#### Model Versions and Stages

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# List registered models
models = client.list_registered_models()

# Get model versions
versions = client.get_registered_model("ProductionModel").latest_versions

# Transition model stage
client.transition_model_version_stage(
    name="ProductionModel",
    version=1,
    stage="Production",
    archive_existing_versions=True
)

# Update model version
client.update_model_version(
    name="ProductionModel",
    version=1,
    description="Updated description"
)

# Delete model version
client.delete_model_version("ProductionModel", version=1)
```

#### Loading Models from Registry

```python
# Load latest version
model = mlflow.pyfunc.load_model("models:/ProductionModel/latest")

# Load specific version
model = mlflow.pyfunc.load_model("models:/ProductionModel/1")

# Load by stage
model = mlflow.pyfunc.load_model("models:/ProductionModel/Production")
```

## Advanced Features

### Autologging

```python
# Enable autologging for specific libraries
mlflow.sklearn.autolog()
mlflow.keras.autolog()
mlflow.pytorch.autolog()
mlflow.xgboost.autolog()

# Enable for all supported libraries
mlflow.autolog()

# Disable autologging
mlflow.sklearn.autolog(disable=True)

# Configure autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=False  # Don't log models automatically
)
```

### Parent and Child Runs

```python
# Parent run
with mlflow.start_run() as parent_run:
    mlflow.log_param("experiment_type", "hyperparameter_tuning")
    
    # Child runs for different configurations
    for alpha in [0.1, 0.5, 1.0]:
        with mlflow.start_run(nested=True) as child_run:
            mlflow.log_param("alpha", alpha)
            # Train and log model...
```

### Search and Query Runs

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs
runs = mlflow.search_runs(
    experiment_ids=["1"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Get run data
run = client.get_run(run_id)
print(f"Status: {run.info.status}")
print(f"Start time: {run.info.start_time}")
print(f"Parameters: {run.data.params}")
print(f"Metrics: {run.data.metrics}")

# Get metric history
history = client.get_metric_history(run_id, "loss")
```

### Experiment Management

```python
# Create experiment
experiment_id = mlflow.create_experiment(
    "New Experiment",
    artifact_location="s3://bucket/artifacts",
    tags={"team": "ml", "project": "recommendation"}
)

# Get experiment
experiment = mlflow.get_experiment(experiment_id)

# List experiments
experiments = mlflow.list_experiments()

# Set experiment by name
mlflow.set_experiment("My Experiment")

# Delete experiment
mlflow.delete_experiment(experiment_id)
```

## Deployment Options

### Local Model Serving

```bash
# Serve model locally
mlflow models serve -m models:/ProductionModel/1 -p 5001

# Serve with conda environment
mlflow models serve -m models:/ProductionModel/1 --env-manager conda

# Build Docker image
mlflow models build-docker -m models:/ProductionModel/1 -n my-model

# Generate Dockerfile
mlflow models generate-dockerfile -m models:/ProductionModel/1 -d ./dockerfile_dir
```

### Cloud Deployments

```python
# Deploy to AWS SageMaker
mlflow.deployments.deploy(
    "sagemaker",
    model_uri="models:/ProductionModel/1",
    config={
        "instance_type": "ml.m5.large",
        "instance_count": 1,
        "region_name": "us-west-2"
    }
)

# Deploy to Azure ML
mlflow.deployments.deploy(
    "azureml",
    model_uri="models:/ProductionModel/1",
    config={
        "compute_type": "ACI",
        "location": "eastus2"
    }
)
```

### Batch Inference

```python
# Apply model to Spark DataFrame
predictions = mlflow.pyfunc.spark_udf(
    spark, 
    model_uri="models:/ProductionModel/1"
)

df_with_predictions = df.withColumn("predictions", predictions(*feature_cols))
```

## Configuration and Setup

### Tracking Server Setup

```bash
# Start local tracking server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# With remote artifact store
mlflow server \
    --backend-store-uri postgresql://user:pass@host:port/db \
    --default-artifact-root s3://bucket/artifacts \
    --host 0.0.0.0
```

### Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export MLFLOW_EXPERIMENT_NAME=default
```

### Configuration File

```python
# mlflow_config.py
import mlflow
import os

def setup_mlflow():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "default"))
    
    # Configure autologging
    mlflow.autolog()
```

## Best Practices and Patterns

### Experiment Organization

```python
# Use descriptive experiment names
mlflow.set_experiment(f"recommendation_model_{datetime.now().strftime('%Y%m%d')}")

# Use consistent tagging strategy
tags = {
    "model_type": "collaborative_filtering",
    "data_version": "v2.1",
    "feature_set": "user_item_interactions",
    "environment": "production"
}
mlflow.set_tags(tags)

# Log important context
mlflow.log_param("git_commit", get_git_commit())
mlflow.log_param("dataset_size", len(train_data))
mlflow.log_param("feature_count", X_train.shape[1])
```

### Model Validation Pattern

```python
def train_and_validate_model(params):
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(params)
        
        # Train model
        model = train_model(params)
        
        # Validate
        train_score = evaluate_model(model, X_train, y_train)
        val_score = evaluate_model(model, X_val, y_val)
        
        # Log metrics
        mlflow.log_metrics({
            "train_accuracy": train_score,
            "val_accuracy": val_score,
            "overfitting": train_score - val_score
        })
        
        # Log model if performance is good
        if val_score > 0.8:
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="BestModel" if val_score > 0.9 else None
            )
        
        return model, val_score
```

### Pipeline Integration

```python
class MLflowPipeline:
    def __init__(self, experiment_name):
        mlflow.set_experiment(experiment_name)
        
    def run_experiment(self, config):
        with mlflow.start_run() as run:
            # Data preparation
            X_train, X_val, y_train, y_val = self.prepare_data(config)
            mlflow.log_params(config["data"])
            
            # Feature engineering
            features = self.engineer_features(X_train, config["features"])
            mlflow.log_param("feature_count", len(features))
            
            # Model training
            model = self.train_model(X_train, y_train, config["model"])
            mlflow.log_params(config["model"])
            
            # Evaluation
            metrics = self.evaluate_model(model, X_val, y_val)
            mlflow.log_metrics(metrics)
            
            # Model logging
            mlflow.sklearn.log_model(model, "model")
            
            return run.info.run_id
```

### Error Handling and Cleanup

```python
def safe_mlflow_run(training_function, **kwargs):
    run_id = None
    try:
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            result = training_function(**kwargs)
            mlflow.log_metrics({"success": 1})
            return result
            
    except Exception as e:
        if run_id:
            # Log error information
            mlflow.log_param("error_message", str(e))
            mlflow.log_metrics({"success": 0})
        
        # End run gracefully
        if mlflow.active_run():
            mlflow.end_run(status="FAILED")
        raise
```

## Common Use Cases and Examples

### Hyperparameter Tuning with Optuna

```python
import optuna
from optuna.integration.mlflow import MLflowCallback

def objective(trial):
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3)
    }
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(params)
        
        # Train and evaluate
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        score = model.score(X_val, y_val)
        mlflow.log_metric('accuracy', score)
        
        return score

# Run optimization
study = optuna.create_study(direction='maximize')
mlflc = MLflowCallback(tracking_uri="http://localhost:5000", metric_name="accuracy")

with mlflow.start_run():
    study.optimize(objective, n_trials=100, callbacks=[mlflc])
    
    # Log best parameters
    mlflow.log_params(study.best_params)
    mlflow.log_metric('best_accuracy', study.best_value)
```

### A/B Testing Model Comparison

```python
def compare_models():
    models = {
        "random_forest": RandomForestClassifier(),
        "gradient_boosting": GradientBoostingClassifier(),
        "svm": SVC()
    }
    
    results = {}
    
    with mlflow.start_run(run_name="model_comparison"):
        for name, model in models.items():
            with mlflow.start_run(run_name=name, nested=True):
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                # Log results
                mlflow.log_params(model.get_params())
                mlflow.log_metrics({
                    "train_accuracy": train_score,
                    "test_accuracy": test_score
                })
                
                # Log model
                mlflow.sklearn.log_model(model, "model")
                
                results[name] = test_score
        
        # Log comparison results
        best_model = max(results.items(), key=lambda x: x[1])
        mlflow.log_param("best_model", best_model[0])
        mlflow.log_metric("best_accuracy", best_model[1])
    
    return results
```

### Cross-Validation Tracking

```python
from sklearn.model_selection import cross_val_score

def cross_validate_with_mlflow(model, X, y, cv=5):
    with mlflow.start_run():
        # Log model parameters
        mlflow.log_params(model.get_params())
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # Log individual fold scores
        for fold, score in enumerate(scores):
            mlflow.log_metric(f"fold_{fold}_accuracy", score)
        
        # Log summary statistics
        mlflow.log_metrics({
            "mean_cv_accuracy": scores.mean(),
            "std_cv_accuracy": scores.std(),
            "min_cv_accuracy": scores.min(),
            "max_cv_accuracy": scores.max()
        })
        
        # Log the model
        model.fit(X, y)  # Fit on full dataset
        mlflow.sklearn.log_model(model, "model")
        
        return scores
```

## CLI Commands Reference

```bash
# Tracking Server
mlflow server --help
mlflow ui  # Start UI on localhost:5000

# Runs
mlflow runs list --experiment-id 1
mlflow runs describe --run-id <run_id>
mlflow runs restore --run-id <run_id>

# Experiments
mlflow experiments list
mlflow experiments create --experiment-name "New Experiment"
mlflow experiments delete --experiment-id 1

# Models
mlflow models serve --help
mlflow models predict --help
mlflow models build-docker --help

# Projects
mlflow run --help
mlflow run . -P alpha=0.1

# Registry
mlflow models list-registered-models
mlflow models get-model-version --name "ModelName" --version 1
```

## Troubleshooting Common Issues

### Connection Issues

```python
# Test connection
try:
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.get_experiment_by_name("Default")
    print("Connection successful")
except Exception as e:
    print(f"Connection failed: {e}")
```

### Storage Issues

```python
# Check artifact location
run = mlflow.get_run(run_id)
print(f"Artifact URI: {run.info.artifact_uri}")

# Download artifacts
mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="model",
    dst_path="./downloads"
)
```

### Performance Optimization

```python
# Batch logging for better performance
with mlflow.start_run():
    # Instead of multiple log_metric calls
    metrics = {}
    for epoch in range(100):
        metrics[f"epoch_{epoch}_loss"] = calculate_loss(epoch)
    
    mlflow.log_metrics(metrics)  # Single batch call
```

