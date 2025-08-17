"""
AWS SageMaker Training Script for UK Solar Energy Prediction

This script handles model training in the SageMaker environment with proper
data loading, model initialization, and result saving for production deployment.

Author: Manus AI
Date: 2025-08-16
"""

import os
import sys
import json
import argparse
import logging
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
import tarfile
import pickle
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append('/opt/ml/code/src')

from data.data_pipeline import UKSolarDataProcessor, SolarFarmConfig
from features.feature_engineering import create_comprehensive_features
from models.trainer import UKSolarTrainer, TrainingConfig
from models.physics_informed_model import PhysicsInformedSolarModel


def setup_logging():
    """Setup logging for SageMaker environment."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('/opt/ml/output/training.log')
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='UK Solar Energy Prediction Training')
    
    # SageMaker environment paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION', '/opt/ml/input/data/validation'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST', '/opt/ml/input/data/test'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # Model hyperparameters
    parser.add_argument('--hidden-size', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--sequence-length', type=int, default=24)
    parser.add_argument('--prediction-horizon', type=int, default=1)
    
    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--gradient-clip', type=float, default=1.0)
    
    # Physics constraint weights
    parser.add_argument('--energy-conservation-weight', type=float, default=0.1)
    parser.add_argument('--thermodynamic-weight', type=float, default=0.05)
    parser.add_argument('--uk-atmospheric-weight', type=float, default=0.08)
    parser.add_argument('--temporal-consistency-weight', type=float, default=0.05)
    
    # Solar farm configuration
    parser.add_argument('--latitude', type=float, default=51.5074)
    parser.add_argument('--longitude', type=float, default=-0.1278)
    parser.add_argument('--elevation', type=float, default=50.0)
    parser.add_argument('--capacity-mw', type=float, default=50.0)
    parser.add_argument('--panel-tilt', type=float, default=35.0)
    parser.add_argument('--panel-azimuth', type=float, default=180.0)
    
    # Data processing options
    parser.add_argument('--include-lags', type=bool, default=True)
    parser.add_argument('--include-rolling', type=bool, default=True)
    parser.add_argument('--feature-selection-k', type=int, default=100)
    
    # Distributed training
    parser.add_argument('--backend', type=str, default='nccl')
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ.get('SM_HOSTS', '["localhost"]')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'localhost'))
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', '0')))
    
    return parser.parse_args()


def load_data(data_path: str, logger: logging.Logger) -> pd.DataFrame:
    """
    Load data from SageMaker input channels.
    
    Args:
        data_path: Path to data directory
        logger: Logger instance
        
    Returns:
        Loaded dataframe
    """
    logger.info(f"Loading data from {data_path}")
    
    # Check for different file formats
    data_files = list(Path(data_path).glob('*'))
    logger.info(f"Found files: {[f.name for f in data_files]}")
    
    if not data_files:
        raise ValueError(f"No data files found in {data_path}")
    
    # Load the first data file (assuming single file per channel)
    data_file = data_files[0]
    
    if data_file.suffix == '.csv':
        df = pd.read_csv(data_file)
    elif data_file.suffix == '.parquet':
        df = pd.read_parquet(data_file)
    elif data_file.suffix == '.json':
        df = pd.read_json(data_file)
    else:
        # Try to load as CSV by default
        df = pd.read_csv(data_file)
    
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def create_solar_farm_config(args) -> SolarFarmConfig:
    """Create solar farm configuration from arguments."""
    from datetime import datetime
    
    return SolarFarmConfig(
        name="UK Solar Farm",
        latitude=args.latitude,
        longitude=args.longitude,
        elevation=args.elevation,
        capacity_mw=args.capacity_mw,
        panel_tilt=args.panel_tilt,
        panel_azimuth=args.panel_azimuth,
        panel_technology="c-Si",
        commissioning_date=datetime(2020, 1, 1),
        region="uk"
    )


def process_data(df: pd.DataFrame, config: SolarFarmConfig, args, logger: logging.Logger) -> pd.DataFrame:
    """
    Process raw data through the complete pipeline.
    
    Args:
        df: Raw dataframe
        config: Solar farm configuration
        args: Command line arguments
        logger: Logger instance
        
    Returns:
        Processed dataframe ready for training
    """
    logger.info("Starting data processing pipeline")
    
    # Initialize data processor
    processor = UKSolarDataProcessor(config)
    
    # Process through full pipeline
    processed_df = processor.process_full_pipeline(
        df, 
        include_lags=args.include_lags,
        include_rolling=args.include_rolling
    )
    
    logger.info(f"Data processing complete. Shape: {processed_df.shape}")
    
    # Create comprehensive features
    logger.info("Creating comprehensive features")
    processed_df = create_comprehensive_features(
        processed_df, 
        config.latitude, 
        config.longitude
    )
    
    logger.info(f"Feature engineering complete. Final shape: {processed_df.shape}")
    
    return processed_df


def identify_feature_columns(df: pd.DataFrame, target_column: str = 'energy_output') -> List[str]:
    """
    Identify feature columns automatically.
    
    Args:
        df: Processed dataframe
        target_column: Target variable column name
        
    Returns:
        List of feature column names
    """
    # Exclude non-feature columns
    exclude_columns = {
        'timestamp', target_column, 'year', 'month', 'day', 'hour', 'minute'
    }
    
    # Get numerical columns only
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    
    # Filter out excluded columns
    feature_columns = [col for col in numerical_columns if col not in exclude_columns]
    
    return feature_columns


def setup_distributed_training(args, logger: logging.Logger):
    """Setup distributed training if multiple GPUs/nodes available."""
    if args.num_gpus > 1 or len(args.hosts) > 1:
        logger.info("Setting up distributed training")
        
        # Initialize process group
        world_size = len(args.hosts) * args.num_gpus
        rank = args.hosts.index(args.current_host) * args.num_gpus
        
        if args.num_gpus > 0:
            torch.cuda.set_device(rank % args.num_gpus)
        
        dist.init_process_group(
            backend=args.backend,
            world_size=world_size,
            rank=rank
        )
        
        logger.info(f"Distributed training setup complete. World size: {world_size}, Rank: {rank}")
        
        return True
    
    return False


def save_model_artifacts(model: PhysicsInformedSolarModel, 
                        feature_columns: List[str],
                        config: SolarFarmConfig,
                        training_config: TrainingConfig,
                        model_dir: str,
                        logger: logging.Logger):
    """
    Save model and associated artifacts for deployment.
    
    Args:
        model: Trained model
        feature_columns: List of feature column names
        config: Solar farm configuration
        training_config: Training configuration
        model_dir: Directory to save model artifacts
        logger: Logger instance
    """
    logger.info(f"Saving model artifacts to {model_dir}")
    
    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    
    # Save model state dict
    model_path = Path(model_dir) / 'model.pt'
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved model state dict to {model_path}")
    
    # Save model configuration
    model_config = {
        'input_size': training_config.input_size,
        'hidden_size': training_config.hidden_size,
        'num_layers': training_config.num_layers,
        'output_size': training_config.output_size,
        'dropout': training_config.dropout,
        'sequence_length': training_config.sequence_length,
        'prediction_horizon': training_config.prediction_horizon
    }
    
    config_path = Path(model_dir) / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model configuration to {config_path}")
    
    # Save feature columns
    features_path = Path(model_dir) / 'feature_columns.json'
    with open(features_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    logger.info(f"Saved feature columns to {features_path}")
    
    # Save solar farm configuration
    farm_config_dict = {
        'name': config.name,
        'latitude': config.latitude,
        'longitude': config.longitude,
        'elevation': config.elevation,
        'capacity_mw': config.capacity_mw,
        'panel_tilt': config.panel_tilt,
        'panel_azimuth': config.panel_azimuth,
        'panel_technology': config.panel_technology,
        'region': config.region
    }
    
    farm_config_path = Path(model_dir) / 'solar_farm_config.json'
    with open(farm_config_path, 'w') as f:
        json.dump(farm_config_dict, f, indent=2)
    logger.info(f"Saved solar farm configuration to {farm_config_path}")
    
    # Save complete model for inference (including architecture)
    complete_model_path = Path(model_dir) / 'complete_model.pt'
    torch.save(model, complete_model_path)
    logger.info(f"Saved complete model to {complete_model_path}")
    
    # Create model.tar.gz for SageMaker deployment
    tar_path = Path(model_dir) / 'model.tar.gz'
    with tarfile.open(tar_path, 'w:gz') as tar:
        for file_path in Path(model_dir).glob('*.pt'):
            tar.add(file_path, arcname=file_path.name)
        for file_path in Path(model_dir).glob('*.json'):
            tar.add(file_path, arcname=file_path.name)
    
    logger.info(f"Created deployment archive: {tar_path}")


def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    logger = setup_logging()
    logger.info("Starting UK Solar Energy Prediction training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Setup distributed training if needed
    is_distributed = setup_distributed_training(args, logger)
    
    # Set device
    if args.num_gpus > 0:
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    try:
        # Load training data
        train_df = load_data(args.train, logger)
        
        # Load validation data if available
        val_df = None
        if os.path.exists(args.validation):
            val_df = load_data(args.validation, logger)
        
        # Create solar farm configuration
        solar_config = create_solar_farm_config(args)
        logger.info(f"Solar farm config: {solar_config}")
        
        # Process training data
        processed_train_df = process_data(train_df, solar_config, args, logger)
        
        # Process validation data if available
        if val_df is not None:
            processed_val_df = process_data(val_df, solar_config, args, logger)
            # Combine for training (trainer will split internally)
            combined_df = pd.concat([processed_train_df, processed_val_df], ignore_index=True)
        else:
            combined_df = processed_train_df
        
        # Identify feature columns
        feature_columns = identify_feature_columns(combined_df)
        logger.info(f"Identified {len(feature_columns)} feature columns")
        
        # Identify solar geometry columns
        solar_geometry_columns = [col for col in combined_df.columns 
                                if any(x in col.lower() for x in ['zenith', 'azimuth', 'elevation', 'air_mass'])]
        logger.info(f"Solar geometry columns: {solar_geometry_columns}")
        
        # Create training configuration
        training_config = TrainingConfig(
            input_size=len(feature_columns),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            output_size=1,
            dropout=args.dropout,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            patience=args.patience,
            sequence_length=args.sequence_length,
            prediction_horizon=args.prediction_horizon,
            weight_decay=args.weight_decay,
            gradient_clip=args.gradient_clip,
            energy_conservation_weight=args.energy_conservation_weight,
            thermodynamic_weight=args.thermodynamic_weight,
            uk_atmospheric_weight=args.uk_atmospheric_weight,
            temporal_consistency_weight=args.temporal_consistency_weight,
            checkpoint_dir='/opt/ml/checkpoints',
            log_dir='/opt/ml/output'
        )
        
        logger.info(f"Training configuration: {training_config}")
        
        # Create trainer
        trainer = UKSolarTrainer(training_config)
        
        # Train model
        logger.info("Starting model training")
        results = trainer.train(
            data=combined_df,
            feature_columns=feature_columns,
            target_column='energy_output',
            solar_geometry_columns=solar_geometry_columns if solar_geometry_columns else None
        )
        
        logger.info("Training completed successfully")
        logger.info(f"Training results: {results}")
        
        # Load best model for saving
        best_model_path = Path(training_config.checkpoint_dir) / 'best_model.pt'
        if best_model_path.exists():
            # Create model instance
            model = trainer.create_model()
            
            # Load best weights
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            logger.info("Loaded best model for deployment")
        else:
            logger.warning("Best model checkpoint not found, using current model")
            model = trainer.create_model()
        
        # Save model artifacts
        save_model_artifacts(
            model=model,
            feature_columns=feature_columns,
            config=solar_config,
            training_config=training_config,
            model_dir=args.model_dir,
            logger=logger
        )
        
        # Save training results
        results_path = Path(args.output_data_dir) / 'training_results.json'
        Path(args.output_data_dir).mkdir(parents=True, exist_ok=True)
        
        # Convert numpy types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                json_results[key] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                   for k, v in value.items()}
            elif isinstance(value, (np.floating, np.integer)):
                json_results[key] = float(value)
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Saved training results to {results_path}")
        
        # Plot and save training history
        try:
            plot_path = Path(args.output_data_dir) / 'training_history.png'
            trainer.plot_training_history(str(plot_path))
            logger.info(f"Saved training history plot to {plot_path}")
        except Exception as e:
            logger.warning(f"Could not save training plot: {e}")
        
        logger.info("Training job completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup distributed training
        if is_distributed:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()

