"""
AWS SageMaker Inference Script for UK Solar Energy Prediction

This script handles model loading and inference in the SageMaker deployment environment
with proper input validation, preprocessing, and output formatting.

Author: Manus AI
Date: 2025-08-16
"""

import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import io
from datetime import datetime, timezone

# Add src directory to path for imports
sys.path.append('/opt/ml/code/src')

from data.data_pipeline import UKSolarDataProcessor, SolarFarmConfig
from features.feature_engineering import create_comprehensive_features
from models.physics_informed_model import PhysicsInformedSolarModel


# Global variables for model and configuration
model = None
feature_columns = None
solar_farm_config = None
data_processor = None
device = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def model_fn(model_dir: str):
    """
    Load model for inference.
    
    Args:
        model_dir: Directory containing model artifacts
        
    Returns:
        Loaded model and associated artifacts
    """
    global model, feature_columns, solar_farm_config, data_processor, device
    
    logger.info(f"Loading model from {model_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load model configuration
        config_path = Path(model_dir) / 'model_config.json'
        with open(config_path, 'r') as f:
            model_config = json.load(f)
        
        logger.info(f"Model configuration: {model_config}")
        
        # Load feature columns
        features_path = Path(model_dir) / 'feature_columns.json'
        with open(features_path, 'r') as f:
            feature_columns = json.load(f)
        
        logger.info(f"Loaded {len(feature_columns)} feature columns")
        
        # Load solar farm configuration
        farm_config_path = Path(model_dir) / 'solar_farm_config.json'
        with open(farm_config_path, 'r') as f:
            farm_config_dict = json.load(f)
        
        # Create SolarFarmConfig object
        solar_farm_config = SolarFarmConfig(
            name=farm_config_dict['name'],
            latitude=farm_config_dict['latitude'],
            longitude=farm_config_dict['longitude'],
            elevation=farm_config_dict['elevation'],
            capacity_mw=farm_config_dict['capacity_mw'],
            panel_tilt=farm_config_dict['panel_tilt'],
            panel_azimuth=farm_config_dict['panel_azimuth'],
            panel_technology=farm_config_dict['panel_technology'],
            commissioning_date=datetime(2020, 1, 1),  # Default date
            region=farm_config_dict['region']
        )
        
        logger.info(f"Solar farm configuration: {solar_farm_config}")
        
        # Initialize data processor
        data_processor = UKSolarDataProcessor(solar_farm_config)
        
        # Create model instance
        model = PhysicsInformedSolarModel(
            input_size=model_config['input_size'],
            hidden_size=model_config['hidden_size'],
            num_layers=model_config['num_layers'],
            output_size=model_config['output_size'],
            dropout=model_config['dropout']
        )
        
        # Load model weights
        model_path = Path(model_dir) / 'model.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        
        logger.info("Model loaded successfully")
        
        return {
            'model': model,
            'feature_columns': feature_columns,
            'solar_farm_config': solar_farm_config,
            'data_processor': data_processor,
            'model_config': model_config
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


def input_fn(request_body: Union[str, bytes], content_type: str = 'application/json'):
    """
    Parse input data for inference.
    
    Args:
        request_body: Raw request body
        content_type: Content type of the request
        
    Returns:
        Parsed input data
    """
    logger.info(f"Processing input with content type: {content_type}")
    
    try:
        if content_type == 'application/json':
            # Parse JSON input
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            input_data = json.loads(request_body)
            
            # Convert to DataFrame if it's a list of records
            if isinstance(input_data, list):
                df = pd.DataFrame(input_data)
            elif isinstance(input_data, dict):
                if 'instances' in input_data:
                    # SageMaker batch transform format
                    df = pd.DataFrame(input_data['instances'])
                else:
                    # Single instance
                    df = pd.DataFrame([input_data])
            else:
                raise ValueError(f"Unsupported input format: {type(input_data)}")
        
        elif content_type == 'text/csv':
            # Parse CSV input
            if isinstance(request_body, bytes):
                request_body = request_body.decode('utf-8')
            
            df = pd.read_csv(io.StringIO(request_body))
        
        else:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        logger.info(f"Parsed input data: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error parsing input: {e}")
        raise


def predict_fn(input_data: pd.DataFrame, model_artifacts: Dict[str, Any]):
    """
    Generate predictions from input data.
    
    Args:
        input_data: Preprocessed input DataFrame
        model_artifacts: Model and configuration artifacts
        
    Returns:
        Prediction results
    """
    global model, feature_columns, solar_farm_config, data_processor
    
    logger.info("Starting prediction")
    
    try:
        # Extract artifacts
        model = model_artifacts['model']
        feature_columns = model_artifacts['feature_columns']
        solar_farm_config = model_artifacts['solar_farm_config']
        data_processor = model_artifacts['data_processor']
        model_config = model_artifacts['model_config']
        
        # Validate input data
        if 'timestamp' not in input_data.columns:
            raise ValueError("Input data must contain 'timestamp' column")
        
        # Ensure timestamp is datetime
        input_data['timestamp'] = pd.to_datetime(input_data['timestamp'])
        
        # Process input data through the pipeline
        logger.info("Processing input data")
        processed_data = data_processor.process_full_pipeline(
            input_data, include_lags=True, include_rolling=True
        )
        
        # Create comprehensive features
        processed_data = create_comprehensive_features(
            processed_data, 
            solar_farm_config.latitude, 
            solar_farm_config.longitude
        )
        
        # Select required features
        missing_features = [col for col in feature_columns if col not in processed_data.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features[:10]}...")  # Show first 10
            # Fill missing features with zeros
            for col in missing_features:
                processed_data[col] = 0.0
        
        # Extract feature data
        feature_data = processed_data[feature_columns].values
        
        # Handle sequence creation for time series model
        sequence_length = model_config.get('sequence_length', 24)
        
        if len(feature_data) < sequence_length:
            # Pad with zeros if insufficient data
            padding = np.zeros((sequence_length - len(feature_data), len(feature_columns)))
            feature_data = np.vstack([padding, feature_data])
        
        # Create sequences for prediction
        predictions = []
        uncertainties = []
        
        # Process in batches if multiple sequences can be created
        for i in range(len(feature_data) - sequence_length + 1):
            sequence = feature_data[i:i + sequence_length]
            
            # Convert to tensor
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)  # Add batch dimension
            
            # Get solar geometry features if available
            solar_geometry_cols = [col for col in processed_data.columns 
                                 if any(x in col.lower() for x in ['zenith', 'azimuth', 'elevation', 'air_mass'])]
            
            solar_geometry_tensor = None
            if solar_geometry_cols and len(solar_geometry_cols) >= 4:
                solar_geometry_data = processed_data[solar_geometry_cols[:4]].iloc[i + sequence_length - 1].values
                solar_geometry_tensor = torch.FloatTensor(solar_geometry_data).unsqueeze(0).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(sequence_tensor, solar_geometry_tensor)
                prediction = outputs['prediction'].cpu().numpy()[0, 0]  # Extract scalar prediction
                
                # Calculate uncertainty if available
                if 'prediction_uncertainty' in outputs:
                    uncertainty = outputs['prediction_uncertainty'].cpu().numpy()[0, 0]
                else:
                    uncertainty = 0.0
                
                predictions.append(prediction)
                uncertainties.append(uncertainty)
        
        # If no sequences could be created, make a single prediction with the last available data
        if not predictions:
            # Use the last sequence_length points (pad if necessary)
            if len(feature_data) >= sequence_length:
                sequence = feature_data[-sequence_length:]
            else:
                padding = np.zeros((sequence_length - len(feature_data), len(feature_columns)))
                sequence = np.vstack([padding, feature_data])
            
            sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(sequence_tensor)
                prediction = outputs['prediction'].cpu().numpy()[0, 0]
                uncertainty = 0.0
                
                predictions.append(prediction)
                uncertainties.append(uncertainty)
        
        # Prepare results
        results = {
            'predictions': predictions,
            'uncertainties': uncertainties,
            'timestamps': processed_data['timestamp'].iloc[-len(predictions):].dt.isoformat().tolist(),
            'model_info': {
                'model_type': 'PhysicsInformedSolarModel',
                'solar_farm': solar_farm_config.name,
                'capacity_mw': solar_farm_config.capacity_mw,
                'location': {
                    'latitude': solar_farm_config.latitude,
                    'longitude': solar_farm_config.longitude
                }
            }
        }
        
        logger.info(f"Generated {len(predictions)} predictions")
        return results
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise


def output_fn(prediction_results: Dict[str, Any], accept: str = 'application/json'):
    """
    Format prediction results for output.
    
    Args:
        prediction_results: Results from predict_fn
        accept: Requested output format
        
    Returns:
        Formatted output
    """
    logger.info(f"Formatting output with accept type: {accept}")
    
    try:
        if accept == 'application/json':
            return json.dumps(prediction_results, indent=2)
        
        elif accept == 'text/csv':
            # Convert to CSV format
            df = pd.DataFrame({
                'timestamp': prediction_results['timestamps'],
                'prediction': prediction_results['predictions'],
                'uncertainty': prediction_results['uncertainties']
            })
            return df.to_csv(index=False)
        
        else:
            # Default to JSON
            return json.dumps(prediction_results, indent=2)
            
    except Exception as e:
        logger.error(f"Error formatting output: {e}")
        raise


def handler(event: Dict[str, Any], context: Any):
    """
    Lambda handler for serverless inference (if deployed as Lambda).
    
    Args:
        event: Lambda event
        context: Lambda context
        
    Returns:
        Prediction response
    """
    try:
        # Extract request body
        body = event.get('body', '')
        content_type = event.get('headers', {}).get('Content-Type', 'application/json')
        
        # Process input
        input_data = input_fn(body, content_type)
        
        # Load model if not already loaded
        if model is None:
            model_artifacts = model_fn('/opt/ml/model')
        else:
            model_artifacts = {
                'model': model,
                'feature_columns': feature_columns,
                'solar_farm_config': solar_farm_config,
                'data_processor': data_processor
            }
        
        # Make prediction
        results = predict_fn(input_data, model_artifacts)
        
        # Format output
        accept = event.get('headers', {}).get('Accept', 'application/json')
        output = output_fn(results, accept)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': accept,
                'Access-Control-Allow-Origin': '*'
            },
            'body': output
        }
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }


# Health check endpoint
def ping():
    """Health check for SageMaker endpoint."""
    try:
        if model is not None:
            return {'status': 'healthy', 'model_loaded': True}
        else:
            return {'status': 'healthy', 'model_loaded': False}
    except Exception as e:
        return {'status': 'unhealthy', 'error': str(e)}


if __name__ == '__main__':
    # Test the inference script locally
    import tempfile
    
    # Create test data
    test_data = {
        'timestamp': ['2024-01-01T12:00:00Z'],
        'temperature': [15.0],
        'humidity': [70.0],
        'pressure': [1013.25],
        'wind_speed': [5.0],
        'wind_direction': [180.0],
        'cloud_cover': [0.3],
        'ghi': [500.0],
        'dni': [600.0],
        'dhi': [200.0]
    }
    
    # Test input parsing
    json_input = json.dumps(test_data)
    parsed_data = input_fn(json_input, 'application/json')
    print(f"Parsed data shape: {parsed_data.shape}")
    print(f"Columns: {list(parsed_data.columns)}")
    
    print("Inference script test completed successfully")

