"""
AWS SageMaker Deployment Script for UK Solar Energy Prediction

This script handles automated deployment of the trained physics-informed model
to SageMaker endpoints with proper configuration and monitoring setup.

Author: Manus AI
Date: 2025-08-16
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import argparse
from datetime import datetime


class UKSolarPredictor(Predictor):
    """
    Custom predictor class for UK Solar Energy predictions.
    """
    
    def __init__(self, endpoint_name: str, sagemaker_session: Optional[sagemaker.Session] = None):
        """
        Initialize UK Solar predictor.
        
        Args:
            endpoint_name: Name of the SageMaker endpoint
            sagemaker_session: SageMaker session
        """
        super().__init__(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=JSONSerializer(),
            deserializer=JSONDeserializer()
        )
    
    def predict_solar_output(self, 
                           weather_data: Dict[str, Any],
                           return_uncertainty: bool = True) -> Dict[str, Any]:
        """
        Predict solar energy output from weather data.
        
        Args:
            weather_data: Dictionary containing weather measurements
            return_uncertainty: Whether to return prediction uncertainty
            
        Returns:
            Prediction results with energy output and metadata
        """
        # Validate required fields
        required_fields = ['timestamp', 'temperature', 'humidity', 'pressure', 
                          'wind_speed', 'cloud_cover', 'ghi']
        
        missing_fields = [field for field in required_fields if field not in weather_data]
        if missing_fields:
            raise ValueError(f"Missing required fields: {missing_fields}")
        
        # Make prediction
        result = self.predict(weather_data)
        
        # Format response
        if isinstance(result, dict) and 'predictions' in result:
            formatted_result = {
                'energy_output_mw': result['predictions'][-1] if result['predictions'] else 0.0,
                'timestamp': result['timestamps'][-1] if result['timestamps'] else weather_data['timestamp'],
                'model_info': result.get('model_info', {}),
                'input_data': weather_data
            }
            
            if return_uncertainty and 'uncertainties' in result:
                formatted_result['uncertainty'] = result['uncertainties'][-1] if result['uncertainties'] else 0.0
            
            return formatted_result
        
        return result


class SageMakerDeployer:
    """
    Handles deployment of UK Solar models to SageMaker.
    """
    
    def __init__(self, 
                 role: str,
                 region: str = 'us-east-1',
                 session: Optional[sagemaker.Session] = None):
        """
        Initialize SageMaker deployer.
        
        Args:
            role: IAM role ARN for SageMaker
            region: AWS region
            session: SageMaker session
        """
        self.role = role
        self.region = region
        self.session = session or sagemaker.Session()
        self.logger = self._setup_logging()
        
        # Initialize boto3 clients
        self.sagemaker_client = boto3.client('sagemaker', region_name=region)
        self.s3_client = boto3.client('s3', region_name=region)
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('SageMakerDeployer')
        logger.setLevel(logging.INFO)
        
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def create_model(self,
                    model_name: str,
                    model_data_url: str,
                    source_dir: str,
                    entry_point: str = 'inference.py',
                    framework_version: str = '1.12.0',
                    py_version: str = 'py38',
                    instance_type: str = 'ml.m5.large') -> PyTorchModel:
        """
        Create SageMaker PyTorch model.
        
        Args:
            model_name: Name for the model
            model_data_url: S3 URL to model artifacts
            source_dir: Directory containing inference code
            entry_point: Entry point script name
            framework_version: PyTorch framework version
            py_version: Python version
            instance_type: Instance type for deployment
            
        Returns:
            Configured PyTorch model
        """
        self.logger.info(f"Creating model: {model_name}")
        
        model = PyTorchModel(
            name=model_name,
            model_data=model_data_url,
            role=self.role,
            source_dir=source_dir,
            entry_point=entry_point,
            framework_version=framework_version,
            py_version=py_version,
            sagemaker_session=self.session,
            predictor_cls=UKSolarPredictor
        )
        
        self.logger.info(f"Model created successfully: {model_name}")
        return model
    
    def deploy_endpoint(self,
                       model: PyTorchModel,
                       endpoint_name: str,
                       instance_type: str = 'ml.m5.large',
                       initial_instance_count: int = 1,
                       wait: bool = True,
                       update_endpoint: bool = False) -> UKSolarPredictor:
        """
        Deploy model to SageMaker endpoint.
        
        Args:
            model: PyTorch model to deploy
            endpoint_name: Name for the endpoint
            instance_type: EC2 instance type
            initial_instance_count: Number of instances
            wait: Whether to wait for deployment completion
            update_endpoint: Whether to update existing endpoint
            
        Returns:
            Predictor for the deployed endpoint
        """
        self.logger.info(f"Deploying endpoint: {endpoint_name}")
        
        try:
            # Check if endpoint already exists
            existing_endpoints = self.sagemaker_client.list_endpoints(
                NameContains=endpoint_name
            )['Endpoints']
            
            endpoint_exists = any(ep['EndpointName'] == endpoint_name 
                                for ep in existing_endpoints)
            
            if endpoint_exists and not update_endpoint:
                self.logger.info(f"Endpoint {endpoint_name} already exists")
                return UKSolarPredictor(endpoint_name, self.session)
            
            # Deploy the model
            predictor = model.deploy(
                endpoint_name=endpoint_name,
                initial_instance_count=initial_instance_count,
                instance_type=instance_type,
                wait=wait,
                update_endpoint=update_endpoint
            )
            
            self.logger.info(f"Endpoint deployed successfully: {endpoint_name}")
            return predictor
            
        except Exception as e:
            self.logger.error(f"Deployment failed: {e}")
            raise
    
    def create_auto_scaling_policy(self,
                                  endpoint_name: str,
                                  variant_name: str = 'AllTraffic',
                                  min_capacity: int = 1,
                                  max_capacity: int = 10,
                                  target_value: float = 70.0,
                                  scale_in_cooldown: int = 300,
                                  scale_out_cooldown: int = 300):
        """
        Create auto-scaling policy for the endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            variant_name: Name of the production variant
            min_capacity: Minimum number of instances
            max_capacity: Maximum number of instances
            target_value: Target CPU utilization percentage
            scale_in_cooldown: Scale-in cooldown period in seconds
            scale_out_cooldown: Scale-out cooldown period in seconds
        """
        self.logger.info(f"Creating auto-scaling policy for {endpoint_name}")
        
        try:
            # Create Application Auto Scaling client
            autoscaling_client = boto3.client('application-autoscaling', 
                                            region_name=self.region)
            
            # Register scalable target
            resource_id = f"endpoint/{endpoint_name}/variant/{variant_name}"
            
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=min_capacity,
                MaxCapacity=max_capacity
            )
            
            # Create scaling policy
            policy_name = f"{endpoint_name}-scaling-policy"
            
            autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': target_value,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleInCooldown': scale_in_cooldown,
                    'ScaleOutCooldown': scale_out_cooldown
                }
            )
            
            self.logger.info(f"Auto-scaling policy created: {policy_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to create auto-scaling policy: {e}")
            raise
    
    def setup_monitoring(self,
                        endpoint_name: str,
                        monitoring_schedule_name: Optional[str] = None,
                        instance_type: str = 'ml.m5.xlarge',
                        volume_size_gb: int = 20):
        """
        Setup data quality monitoring for the endpoint.
        
        Args:
            endpoint_name: Name of the endpoint to monitor
            monitoring_schedule_name: Name for the monitoring schedule
            instance_type: Instance type for monitoring jobs
            volume_size_gb: Volume size for monitoring jobs
        """
        if monitoring_schedule_name is None:
            monitoring_schedule_name = f"{endpoint_name}-monitoring"
        
        self.logger.info(f"Setting up monitoring: {monitoring_schedule_name}")
        
        try:
            from sagemaker.model_monitor import DefaultModelMonitor
            from sagemaker.model_monitor.dataset_format import DatasetFormat
            
            # Create model monitor
            monitor = DefaultModelMonitor(
                role=self.role,
                instance_count=1,
                instance_type=instance_type,
                volume_size_in_gb=volume_size_gb,
                max_runtime_in_seconds=3600,
                sagemaker_session=self.session
            )
            
            # Create monitoring schedule
            monitor.create_monitoring_schedule(
                monitor_schedule_name=monitoring_schedule_name,
                endpoint_input=endpoint_name,
                schedule_cron_expression='cron(0 * * * ? *)',  # Hourly
                statistics=None,  # Will be created from baseline
                constraints=None,  # Will be created from baseline
                enable_cloudwatch_metrics=True
            )
            
            self.logger.info(f"Monitoring schedule created: {monitoring_schedule_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to setup monitoring: {e}")
            raise
    
    def test_endpoint(self,
                     predictor: UKSolarPredictor,
                     test_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Test the deployed endpoint with sample data.
        
        Args:
            predictor: Deployed predictor
            test_data: Test data (uses default if None)
            
        Returns:
            Test results
        """
        self.logger.info("Testing endpoint")
        
        if test_data is None:
            # Default test data
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
        
        try:
            # Test prediction
            start_time = time.time()
            result = predictor.predict_solar_output(test_data)
            inference_time = time.time() - start_time
            
            test_results = {
                'status': 'success',
                'inference_time_ms': inference_time * 1000,
                'prediction_result': result,
                'test_data': test_data
            }
            
            self.logger.info(f"Endpoint test successful. Inference time: {inference_time:.3f}s")
            return test_results
            
        except Exception as e:
            self.logger.error(f"Endpoint test failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_data': test_data
            }
    
    def delete_endpoint(self, endpoint_name: str):
        """
        Delete SageMaker endpoint and associated resources.
        
        Args:
            endpoint_name: Name of endpoint to delete
        """
        self.logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Delete endpoint configuration
            try:
                self.sagemaker_client.delete_endpoint_config(
                    EndpointConfigName=endpoint_name
                )
            except Exception as e:
                self.logger.warning(f"Could not delete endpoint config: {e}")
            
            self.logger.info(f"Endpoint deleted: {endpoint_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete endpoint: {e}")
            raise


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Deploy UK Solar Energy Prediction Model')
    
    parser.add_argument('--role', type=str, required=True,
                       help='IAM role ARN for SageMaker')
    parser.add_argument('--model-data-url', type=str, required=True,
                       help='S3 URL to model artifacts')
    parser.add_argument('--source-dir', type=str, default='.',
                       help='Directory containing inference code')
    parser.add_argument('--model-name', type=str, default=None,
                       help='Name for the model')
    parser.add_argument('--endpoint-name', type=str, default=None,
                       help='Name for the endpoint')
    parser.add_argument('--instance-type', type=str, default='ml.m5.large',
                       help='Instance type for deployment')
    parser.add_argument('--instance-count', type=int, default=1,
                       help='Number of instances')
    parser.add_argument('--region', type=str, default='us-east-1',
                       help='AWS region')
    parser.add_argument('--framework-version', type=str, default='1.12.0',
                       help='PyTorch framework version')
    parser.add_argument('--enable-autoscaling', action='store_true',
                       help='Enable auto-scaling')
    parser.add_argument('--enable-monitoring', action='store_true',
                       help='Enable model monitoring')
    parser.add_argument('--test-endpoint', action='store_true',
                       help='Test endpoint after deployment')
    parser.add_argument('--update-endpoint', action='store_true',
                       help='Update existing endpoint')
    
    return parser.parse_args()


def main():
    """Main deployment function."""
    args = parse_args()
    
    # Generate names if not provided
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    model_name = args.model_name or f'uk-solar-model-{timestamp}'
    endpoint_name = args.endpoint_name or f'uk-solar-endpoint-{timestamp}'
    
    # Initialize deployer
    deployer = SageMakerDeployer(
        role=args.role,
        region=args.region
    )
    
    try:
        # Create model
        model = deployer.create_model(
            model_name=model_name,
            model_data_url=args.model_data_url,
            source_dir=args.source_dir,
            framework_version=args.framework_version,
            instance_type=args.instance_type
        )
        
        # Deploy endpoint
        predictor = deployer.deploy_endpoint(
            model=model,
            endpoint_name=endpoint_name,
            instance_type=args.instance_type,
            initial_instance_count=args.instance_count,
            update_endpoint=args.update_endpoint
        )
        
        # Setup auto-scaling if requested
        if args.enable_autoscaling:
            deployer.create_auto_scaling_policy(endpoint_name)
        
        # Setup monitoring if requested
        if args.enable_monitoring:
            deployer.setup_monitoring(endpoint_name)
        
        # Test endpoint if requested
        if args.test_endpoint:
            test_results = deployer.test_endpoint(predictor)
            print(f"Test results: {json.dumps(test_results, indent=2)}")
        
        print(f"Deployment completed successfully!")
        print(f"Model name: {model_name}")
        print(f"Endpoint name: {endpoint_name}")
        print(f"Endpoint URL: https://runtime.sagemaker.{args.region}.amazonaws.com/endpoints/{endpoint_name}/invocations")
        
        # Save deployment info
        deployment_info = {
            'model_name': model_name,
            'endpoint_name': endpoint_name,
            'instance_type': args.instance_type,
            'instance_count': args.instance_count,
            'region': args.region,
            'deployment_time': datetime.now().isoformat(),
            'model_data_url': args.model_data_url,
            'autoscaling_enabled': args.enable_autoscaling,
            'monitoring_enabled': args.enable_monitoring
        }
        
        with open('deployment_info.json', 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        print("Deployment information saved to deployment_info.json")
        
    except Exception as e:
        print(f"Deployment failed: {e}")
        raise


if __name__ == '__main__':
    main()

