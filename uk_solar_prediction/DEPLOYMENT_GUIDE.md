# UK Solar Energy Prediction - AWS SageMaker Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the UK Solar Energy Prediction model to AWS SageMaker. The system uses physics-informed neural networks to predict solar energy generation for UK solar farms with high accuracy and physical consistency.

## Prerequisites

### AWS Account Setup
1. **AWS Account**: Active AWS account with appropriate permissions
2. **IAM Role**: SageMaker execution role with required permissions
3. **S3 Bucket**: For storing model artifacts and data
4. **AWS CLI**: Configured with your credentials

### Required Permissions
Your IAM role should include:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `CloudWatchLogsFullAccess`

### Local Environment
- Python 3.8+
- AWS CLI configured
- Docker (for custom container deployment)
- This project repository

## Quick Deployment

### 1. Prepare Your Environment

```bash
# Clone the repository
git clone <your-repo-url>
cd uk_solar_prediction

# Set up environment
python scripts/setup_environment.py --dev

# Install AWS dependencies
pip install boto3 sagemaker
```

### 2. Configure AWS Credentials

```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 3. Prepare Your Data

```bash
# Upload your solar farm data to S3
aws s3 cp data/raw/your_solar_data.csv s3://your-bucket/data/

# Update solar farm configuration
cp configs/sample_solar_farm.yaml configs/your_farm.yaml
# Edit configs/your_farm.yaml with your farm details
```

### 4. Train the Model

```bash
# Local training (recommended for testing)
python src/models/trainer.py --config configs/training_config.yaml

# Or SageMaker training
python sagemaker/train.py \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole \
    --data-path s3://your-bucket/data/ \
    --output-path s3://your-bucket/models/
```

### 5. Deploy to SageMaker

```bash
python sagemaker/deploy.py \
    --role arn:aws:iam::YOUR_ACCOUNT:role/SageMakerRole \
    --model-data-url s3://your-bucket/models/model.tar.gz \
    --endpoint-name uk-solar-prediction \
    --instance-type ml.m5.large
```

## Detailed Deployment Steps

### Step 1: Data Preparation

#### 1.1 Data Format Requirements

Your solar farm data should be in CSV format with the following columns:

```csv
timestamp,temperature,humidity,pressure,wind_speed,wind_direction,cloud_cover,ghi,dni,dhi,energy_output
2024-01-01T00:00:00Z,5.2,85.3,1015.2,3.1,180.5,0.8,0,0,0,0
2024-01-01T01:00:00Z,4.8,87.1,1014.8,2.9,175.2,0.9,0,0,0,0
...
```

#### 1.2 Data Validation

```python
from src.data.data_pipeline import UKSolarDataProcessor, load_solar_farm_config

# Load your configuration
config = load_solar_farm_config('configs/your_farm.yaml')

# Create processor
processor = UKSolarDataProcessor(config)

# Validate your data
import pandas as pd
data = pd.read_csv('data/raw/your_solar_data.csv')
validated_data = processor.validate_data(data)

print(f"Original data: {len(data)} rows")
print(f"Validated data: {len(validated_data)} rows")
```

#### 1.3 Upload to S3

```bash
# Create S3 bucket (if needed)
aws s3 mb s3://your-solar-prediction-bucket

# Upload data
aws s3 cp data/raw/your_solar_data.csv s3://your-solar-prediction-bucket/data/train/
aws s3 cp configs/your_farm.yaml s3://your-solar-prediction-bucket/config/
```

### Step 2: Model Training

#### 2.1 Local Training (Recommended First)

```python
# Train locally to validate setup
from src.models.trainer import UKSolarTrainer, TrainingConfig
from src.data.data_pipeline import load_solar_farm_config

# Load configurations
farm_config = load_solar_farm_config('configs/your_farm.yaml')
training_config = TrainingConfig.from_yaml('configs/training_config.yaml')

# Create trainer
trainer = UKSolarTrainer(farm_config, training_config)

# Load and process data
data = trainer.load_data('data/raw/your_solar_data.csv')
processed_data = trainer.process_data(data)

# Train model
model, metrics = trainer.train(processed_data)

# Save model
trainer.save_model(model, 'models/trained/local_model.pt')
```

#### 2.2 SageMaker Training

```bash
# Submit SageMaker training job
python sagemaker/train.py \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --data-path s3://your-solar-prediction-bucket/data/ \
    --config-path s3://your-solar-prediction-bucket/config/your_farm.yaml \
    --output-path s3://your-solar-prediction-bucket/models/ \
    --instance-type ml.m5.xlarge \
    --instance-count 1 \
    --max-runtime 3600 \
    --job-name uk-solar-training-$(date +%Y%m%d-%H%M%S)
```

#### 2.3 Monitor Training

```python
import boto3

sagemaker = boto3.client('sagemaker')

# Check training job status
response = sagemaker.describe_training_job(
    TrainingJobName='your-training-job-name'
)

print(f"Status: {response['TrainingJobStatus']}")
print(f"Progress: {response.get('SecondaryStatus', 'N/A')}")
```

### Step 3: Model Deployment

#### 3.1 Basic Deployment

```python
from sagemaker.deploy import SageMakerDeployer

# Initialize deployer
deployer = SageMakerDeployer(
    role='arn:aws:iam::123456789012:role/SageMakerRole',
    region='us-east-1'
)

# Create model
model = deployer.create_model(
    model_name='uk-solar-model-v1',
    model_data_url='s3://your-bucket/models/model.tar.gz',
    source_dir='sagemaker',
    entry_point='inference.py'
)

# Deploy endpoint
predictor = deployer.deploy_endpoint(
    model=model,
    endpoint_name='uk-solar-endpoint',
    instance_type='ml.m5.large',
    initial_instance_count=1
)
```

#### 3.2 Advanced Deployment with Auto-scaling

```python
# Deploy with auto-scaling
predictor = deployer.deploy_endpoint(
    model=model,
    endpoint_name='uk-solar-endpoint-autoscale',
    instance_type='ml.m5.large',
    initial_instance_count=1
)

# Setup auto-scaling
deployer.create_auto_scaling_policy(
    endpoint_name='uk-solar-endpoint-autoscale',
    min_capacity=1,
    max_capacity=10,
    target_value=70.0  # Target CPU utilization
)
```

#### 3.3 Multi-Model Endpoint

```python
# Deploy multiple model versions
models = []

for version in ['v1', 'v2', 'v3']:
    model = deployer.create_model(
        model_name=f'uk-solar-model-{version}',
        model_data_url=f's3://your-bucket/models/model-{version}.tar.gz',
        source_dir='sagemaker'
    )
    models.append(model)

# Deploy multi-model endpoint
from sagemaker.multidatamodel import MultiDataModel

multi_model = MultiDataModel(
    name='uk-solar-multi-model',
    model_data_prefix='s3://your-bucket/models/',
    models=models
)

predictor = multi_model.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='uk-solar-multi-endpoint'
)
```

### Step 4: Testing and Validation

#### 4.1 Basic Endpoint Testing

```python
# Test the deployed endpoint
test_data = {
    'timestamp': '2024-01-01T12:00:00Z',
    'temperature': 15.0,
    'humidity': 70.0,
    'pressure': 1013.25,
    'wind_speed': 5.0,
    'wind_direction': 180.0,
    'cloud_cover': 0.3,
    'ghi': 500.0,
    'dni': 600.0,
    'dhi': 200.0
}

# Make prediction
result = predictor.predict_solar_output(test_data)
print(f"Predicted energy output: {result['energy_output_mw']} MW")
print(f"Uncertainty: {result.get('uncertainty', 'N/A')}")
```

#### 4.2 Batch Testing

```python
import pandas as pd

# Load test data
test_df = pd.read_csv('data/test/test_data.csv')

# Batch prediction
batch_results = []
for _, row in test_df.iterrows():
    test_input = row.to_dict()
    result = predictor.predict_solar_output(test_input)
    batch_results.append(result)

# Analyze results
results_df = pd.DataFrame(batch_results)
print(f"Mean predicted output: {results_df['energy_output_mw'].mean():.2f} MW")
print(f"Prediction range: {results_df['energy_output_mw'].min():.2f} - {results_df['energy_output_mw'].max():.2f} MW")
```

#### 4.3 Performance Testing

```python
import time
import numpy as np

# Performance test
num_requests = 100
latencies = []

for i in range(num_requests):
    start_time = time.time()
    result = predictor.predict_solar_output(test_data)
    latency = time.time() - start_time
    latencies.append(latency)

print(f"Average latency: {np.mean(latencies)*1000:.2f} ms")
print(f"95th percentile: {np.percentile(latencies, 95)*1000:.2f} ms")
print(f"Max latency: {np.max(latencies)*1000:.2f} ms")
```

### Step 5: Monitoring and Maintenance

#### 5.1 Setup CloudWatch Monitoring

```python
# Enable model monitoring
deployer.setup_monitoring(
    endpoint_name='uk-solar-endpoint',
    monitoring_schedule_name='uk-solar-monitoring'
)
```

#### 5.2 Custom Metrics

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Custom metric for prediction accuracy
def log_prediction_accuracy(actual, predicted):
    accuracy = 1 - abs(actual - predicted) / actual
    
    cloudwatch.put_metric_data(
        Namespace='UKSolar/Predictions',
        MetricData=[
            {
                'MetricName': 'PredictionAccuracy',
                'Value': accuracy,
                'Unit': 'Percent'
            }
        ]
    )
```

#### 5.3 Automated Model Updates

```python
# Schedule model retraining
import boto3

events = boto3.client('events')
lambda_client = boto3.client('lambda')

# Create CloudWatch Events rule for weekly retraining
events.put_rule(
    Name='uk-solar-weekly-retrain',
    ScheduleExpression='rate(7 days)',
    Description='Weekly model retraining for UK Solar prediction'
)

# Connect to Lambda function for retraining
events.put_targets(
    Rule='uk-solar-weekly-retrain',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-east-1:123456789012:function:uk-solar-retrain'
        }
    ]
)
```

## Production Deployment Checklist

### Pre-deployment
- [ ] Data validation completed
- [ ] Model training successful
- [ ] Local testing passed
- [ ] Security review completed
- [ ] Performance benchmarks met
- [ ] Cost estimation approved

### Deployment
- [ ] IAM roles configured
- [ ] S3 buckets created and secured
- [ ] Model artifacts uploaded
- [ ] Endpoint deployed successfully
- [ ] Auto-scaling configured
- [ ] Monitoring enabled

### Post-deployment
- [ ] Endpoint health checks passing
- [ ] Performance metrics within targets
- [ ] Monitoring dashboards created
- [ ] Alerting configured
- [ ] Documentation updated
- [ ] Team training completed

## Troubleshooting

### Common Issues

#### 1. Training Job Failures

```bash
# Check training job logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/TrainingJobs

# Get specific log stream
aws logs get-log-events \
    --log-group-name /aws/sagemaker/TrainingJobs \
    --log-stream-name your-training-job-name/algo-1-1234567890
```

#### 2. Endpoint Deployment Issues

```python
# Check endpoint status
import boto3

sagemaker = boto3.client('sagemaker')
response = sagemaker.describe_endpoint(EndpointName='uk-solar-endpoint')
print(f"Status: {response['EndpointStatus']}")

if response['EndpointStatus'] == 'Failed':
    print(f"Failure reason: {response.get('FailureReason', 'Unknown')}")
```

#### 3. Inference Errors

```python
# Test with minimal data
minimal_test = {
    'timestamp': '2024-01-01T12:00:00Z',
    'temperature': 15.0,
    'humidity': 70.0,
    'pressure': 1013.25,
    'wind_speed': 5.0,
    'cloud_cover': 0.3,
    'ghi': 500.0
}

try:
    result = predictor.predict_solar_output(minimal_test)
    print("Success:", result)
except Exception as e:
    print("Error:", str(e))
```

### Performance Optimization

#### 1. Instance Type Selection

| Instance Type | vCPUs | Memory | Use Case |
|---------------|-------|--------|----------|
| ml.t3.medium | 2 | 4 GB | Development/Testing |
| ml.m5.large | 2 | 8 GB | Low-volume production |
| ml.m5.xlarge | 4 | 16 GB | Medium-volume production |
| ml.c5.2xlarge | 8 | 16 GB | High-throughput |
| ml.p3.2xlarge | 8 | 61 GB | GPU acceleration |

#### 2. Batch Transform for Large Datasets

```python
from sagemaker.transformer import Transformer

# Create transformer
transformer = Transformer(
    model_name='uk-solar-model-v1',
    instance_count=1,
    instance_type='ml.m5.xlarge',
    output_path='s3://your-bucket/batch-predictions/'
)

# Run batch transform
transformer.transform(
    data='s3://your-bucket/batch-input/',
    content_type='text/csv',
    split_type='Line'
)
```

### Cost Optimization

#### 1. Spot Instances for Training

```python
# Use spot instances for training
training_job = sagemaker.create_training_job(
    TrainingJobName='uk-solar-spot-training',
    RoleArn='arn:aws:iam::123456789012:role/SageMakerRole',
    InputDataConfig=[...],
    OutputDataConfig={...},
    ResourceConfig={
        'InstanceType': 'ml.m5.xlarge',
        'InstanceCount': 1,
        'VolumeSizeInGB': 30
    },
    EnableManagedSpotTraining=True,
    StoppingCondition={
        'MaxRuntimeInSeconds': 3600,
        'MaxWaitTimeInSeconds': 7200
    }
)
```

#### 2. Serverless Inference

```python
from sagemaker.serverless import ServerlessInferenceConfig

# Deploy serverless endpoint
serverless_config = ServerlessInferenceConfig(
    memory_size_in_mb=4096,
    max_concurrency=10
)

predictor = model.deploy(
    serverless_inference_config=serverless_config,
    endpoint_name='uk-solar-serverless'
)
```

## Security Best Practices

### 1. IAM Policies

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "sagemaker:CreateModel",
                "sagemaker:CreateEndpoint",
                "sagemaker:CreateEndpointConfig",
                "sagemaker:InvokeEndpoint"
            ],
            "Resource": "arn:aws:sagemaker:*:*:*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "s3:GetObject",
                "s3:PutObject"
            ],
            "Resource": "arn:aws:s3:::your-solar-prediction-bucket/*"
        }
    ]
}
```

### 2. VPC Configuration

```python
# Deploy in VPC for enhanced security
vpc_config = {
    'SecurityGroupIds': ['sg-12345678'],
    'Subnets': ['subnet-12345678', 'subnet-87654321']
}

model = deployer.create_model(
    model_name='uk-solar-secure-model',
    model_data_url='s3://your-bucket/model.tar.gz',
    source_dir='sagemaker',
    vpc_config=vpc_config
)
```

### 3. Encryption

```python
# Enable encryption at rest and in transit
model = deployer.create_model(
    model_name='uk-solar-encrypted-model',
    model_data_url='s3://your-bucket/model.tar.gz',
    source_dir='sagemaker',
    enable_network_isolation=True,
    encrypt_inter_container_traffic=True
)
```

## Support and Resources

### Documentation
- [AWS SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [SageMaker Python SDK](https://sagemaker.readthedocs.io/)
- [Project README](README.md)

### Monitoring
- CloudWatch Dashboards
- SageMaker Model Monitor
- Custom metrics and alerts

### Support Channels
- AWS Support (for AWS-related issues)
- Project GitHub Issues
- Internal team documentation

---

**Note**: This deployment guide assumes familiarity with AWS services and machine learning concepts. For production deployments, always follow your organization's security and compliance requirements.

