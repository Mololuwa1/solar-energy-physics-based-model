"""
Training Module for Physics-Informed UK Solar Models

Comprehensive training framework with physics constraint optimization,
adaptive learning, and UK-specific validation metrics.

Author: Mololuwa Obafemi-Moses
Date: 2025-08-16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import logging
from pathlib import Path
import json
import time
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from physics_informed_model import PhysicsInformedSolarModel, UKSolarEnsemble, PhysicsConstraints


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model parameters
    input_size: int
    hidden_size: int = 128
    num_layers: int = 3
    output_size: int = 1
    dropout: float = 0.1
    
    # Training parameters
    batch_size: int = 64
    learning_rate: float = 0.001
    num_epochs: int = 100
    patience: int = 15
    min_delta: float = 1e-4
    
    # Physics constraints
    energy_conservation_weight: float = 0.1
    thermodynamic_weight: float = 0.05
    radiative_transfer_weight: float = 0.1
    temporal_consistency_weight: float = 0.05
    uk_atmospheric_weight: float = 0.08
    
    # Data parameters
    sequence_length: int = 24
    prediction_horizon: int = 1
    validation_split: float = 0.2
    test_split: float = 0.1
    
    # Optimization
    optimizer: str = 'adam'
    scheduler: str = 'reduce_on_plateau'
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    
    # Logging and checkpointing
    log_interval: int = 10
    save_interval: int = 50
    checkpoint_dir: str = 'checkpoints'
    log_dir: str = 'logs'


class SolarDataset(Dataset):
    """
    PyTorch Dataset for UK solar farm data.
    """
    
    def __init__(self, 
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 target_column: str,
                 sequence_length: int = 24,
                 prediction_horizon: int = 1,
                 solar_geometry_columns: Optional[List[str]] = None):
        """
        Initialize solar dataset.
        
        Args:
            data: Preprocessed solar farm data
            feature_columns: List of feature column names
            target_column: Target variable column name
            sequence_length: Length of input sequences
            prediction_horizon: Number of steps ahead to predict
            solar_geometry_columns: Solar geometry feature columns
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.solar_geometry_columns = solar_geometry_columns or []
        
        # Prepare sequences
        self.sequences = self._create_sequences()
        
    def _create_sequences(self) -> List[Dict[str, np.ndarray]]:
        """Create sequences for time series modeling."""
        sequences = []
        
        # Sort by timestamp
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Extract features and targets
        features = self.data[self.feature_columns].values
        targets = self.data[self.target_column].values
        
        if self.solar_geometry_columns:
            solar_geometry = self.data[self.solar_geometry_columns].values
        else:
            solar_geometry = None
        
        # Create sequences
        for i in range(len(self.data) - self.sequence_length - self.prediction_horizon + 1):
            # Input sequence
            x = features[i:i + self.sequence_length]
            
            # Target (prediction_horizon steps ahead)
            y = targets[i + self.sequence_length + self.prediction_horizon - 1]
            
            # Solar geometry for physics constraints (use last timestep)
            if solar_geometry is not None:
                geom = solar_geometry[i + self.sequence_length - 1]
            else:
                geom = None
            
            sequences.append({
                'features': x.astype(np.float32),
                'target': np.array([y], dtype=np.float32),
                'solar_geometry': geom.astype(np.float32) if geom is not None else None,
                'timestamp': self.data.iloc[i + self.sequence_length - 1]['timestamp']
            })
        
        return sequences
    
    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item by index."""
        sequence = self.sequences[idx]
        
        item = {
            'features': torch.from_numpy(sequence['features']),
            'target': torch.from_numpy(sequence['target']),
            'timestamp': sequence['timestamp']
        }
        
        if sequence['solar_geometry'] is not None:
            item['solar_geometry'] = torch.from_numpy(sequence['solar_geometry'])
        
        return item


class UKSolarTrainer:
    """
    Comprehensive trainer for UK solar prediction models.
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.logger = self._setup_logging()
        
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [], 'val_loss': [], 'physics_losses': []
        }
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger('UKSolarTrainer')
        logger.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        file_handler = logging.FileHandler(
            Path(self.config.log_dir) / 'training.log'
        )
        file_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
        
        return logger
    
    def create_model(self) -> PhysicsInformedSolarModel:
        """Create and initialize model."""
        physics_constraints = PhysicsConstraints(
            energy_conservation_weight=self.config.energy_conservation_weight,
            thermodynamic_weight=self.config.thermodynamic_weight,
            radiative_transfer_weight=self.config.radiative_transfer_weight,
            temporal_consistency_weight=self.config.temporal_consistency_weight,
            uk_atmospheric_weight=self.config.uk_atmospheric_weight
        )
        
        model = PhysicsInformedSolarModel(
            input_size=self.config.input_size,
            hidden_size=self.config.hidden_size,
            num_layers=self.config.num_layers,
            output_size=self.config.output_size,
            dropout=self.config.dropout,
            physics_constraints=physics_constraints
        )
        
        return model.to(self.device)
    
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == 'adam':
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer.lower() == 'sgd':
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        return optimizer
    
    def create_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == 'reduce_on_plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        elif self.config.scheduler.lower() == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.num_epochs
            )
        elif self.config.scheduler.lower() == 'step':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, step_size=30, gamma=0.1
            )
        else:
            scheduler = None
        
        return scheduler
    
    def prepare_data(self, data: pd.DataFrame,
                    feature_columns: List[str],
                    target_column: str,
                    solar_geometry_columns: Optional[List[str]] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Prepare data loaders for training, validation, and testing.
        
        Args:
            data: Preprocessed data
            feature_columns: Feature column names
            target_column: Target column name
            solar_geometry_columns: Solar geometry column names
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        # Create dataset
        dataset = SolarDataset(
            data=data,
            feature_columns=feature_columns,
            target_column=target_column,
            sequence_length=self.config.sequence_length,
            prediction_horizon=self.config.prediction_horizon,
            solar_geometry_columns=solar_geometry_columns
        )
        
        # Split dataset
        total_size = len(dataset)
        test_size = int(total_size * self.config.test_split)
        val_size = int(total_size * self.config.validation_split)
        train_size = total_size - test_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size,
            shuffle=False, num_workers=4, pin_memory=True
        )
        
        self.logger.info(f"Data split - Train: {train_size}, Val: {val_size}, Test: {test_size}")
        
        return train_loader, val_loader, test_loader
    
    def train_epoch(self, model: nn.Module, 
                   train_loader: DataLoader,
                   optimizer: optim.Optimizer) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            model: Model to train
            train_loader: Training data loader
            optimizer: Optimizer
            
        Returns:
            Dictionary of training metrics
        """
        model.train()
        
        total_loss = 0.0
        total_data_loss = 0.0
        total_physics_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            features = batch['features'].to(self.device)
            targets = batch['target'].to(self.device)
            solar_geometry = batch.get('solar_geometry')
            if solar_geometry is not None:
                solar_geometry = solar_geometry.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features, solar_geometry)
            
            # Calculate loss
            total_loss_batch, loss_components = model.compute_total_loss(
                outputs, targets, features
            )
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
            
            optimizer.step()
            
            # Accumulate losses
            total_loss += total_loss_batch.item()
            total_data_loss += loss_components['data_loss'].item()
            total_physics_loss += (total_loss_batch.item() - loss_components['data_loss'].item())
            num_batches += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                self.logger.info(
                    f'Epoch {self.current_epoch}, Batch {batch_idx}/{len(train_loader)}, '
                    f'Loss: {total_loss_batch.item():.6f}'
                )
        
        return {
            'total_loss': total_loss / num_batches,
            'data_loss': total_data_loss / num_batches,
            'physics_loss': total_physics_loss / num_batches
        }
    
    def validate_epoch(self, model: nn.Module, 
                      val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            model: Model to validate
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        model.eval()
        
        total_loss = 0.0
        total_data_loss = 0.0
        predictions = []
        targets_list = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                features = batch['features'].to(self.device)
                targets = batch['target'].to(self.device)
                solar_geometry = batch.get('solar_geometry')
                if solar_geometry is not None:
                    solar_geometry = solar_geometry.to(self.device)
                
                # Forward pass
                outputs = model(features, solar_geometry)
                
                # Calculate loss
                total_loss_batch, loss_components = model.compute_total_loss(
                    outputs, targets, features
                )
                
                total_loss += total_loss_batch.item()
                total_data_loss += loss_components['data_loss'].item()
                
                # Collect predictions for metrics
                predictions.append(outputs['prediction'].cpu().numpy())
                targets_list.append(targets.cpu().numpy())
        
        # Calculate additional metrics
        predictions = np.concatenate(predictions, axis=0)
        targets_array = np.concatenate(targets_list, axis=0)
        
        mse = mean_squared_error(targets_array, predictions)
        mae = mean_absolute_error(targets_array, predictions)
        r2 = r2_score(targets_array, predictions)
        
        return {
            'total_loss': total_loss / len(val_loader),
            'data_loss': total_data_loss / len(val_loader),
            'mse': mse,
            'mae': mae,
            'r2': r2,
            'rmse': np.sqrt(mse)
        }
    
    def save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                       scheduler: Optional[Any], metrics: Dict[str, float],
                       is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'config': asdict(self.config),
            'metrics': metrics,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = Path(self.config.checkpoint_dir) / f'checkpoint_epoch_{self.current_epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = Path(self.config.checkpoint_dir) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best model with validation loss: {metrics['total_loss']:.6f}")
    
    def load_checkpoint(self, checkpoint_path: str, 
                       model: nn.Module, 
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[Any] = None) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'val_loss': [], 'physics_losses': []
        })
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
        return checkpoint
    
    def train(self, data: pd.DataFrame,
             feature_columns: List[str],
             target_column: str,
             solar_geometry_columns: Optional[List[str]] = None,
             resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            data: Training data
            feature_columns: Feature column names
            target_column: Target column name
            solar_geometry_columns: Solar geometry column names
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Training results dictionary
        """
        self.logger.info("Starting training...")
        
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(
            data, feature_columns, target_column, solar_geometry_columns
        )
        
        # Create model
        model = self.create_model()
        optimizer = self.create_optimizer(model)
        scheduler = self.create_scheduler(optimizer)
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from, model, optimizer, scheduler)
        
        self.logger.info(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
        
        # Training loop
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(model, train_loader, optimizer)
            
            # Validate epoch
            val_metrics = self.validate_epoch(model, val_loader)
            
            # Update learning rate
            if scheduler:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['total_loss'])
                else:
                    scheduler.step()
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.6f}, "
                f"Val Loss: {val_metrics['total_loss']:.6f}, "
                f"Val RMSE: {val_metrics['rmse']:.6f}, "
                f"Val R²: {val_metrics['r2']:.4f}"
            )
            
            # Update training history
            self.training_history['train_loss'].append(train_metrics['total_loss'])
            self.training_history['val_loss'].append(val_metrics['total_loss'])
            self.training_history['physics_losses'].append(train_metrics['physics_loss'])
            
            # Check for best model
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if epoch % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(model, optimizer, scheduler, val_metrics, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Final evaluation on test set
        test_metrics = self.validate_epoch(model, test_loader)
        
        training_time = time.time() - start_time
        
        self.logger.info(f"Training completed in {training_time:.2f} seconds")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.6f}")
        self.logger.info(f"Test metrics - RMSE: {test_metrics['rmse']:.6f}, R²: {test_metrics['r2']:.4f}")
        
        # Save final results
        results = {
            'best_val_loss': self.best_val_loss,
            'test_metrics': test_metrics,
            'training_history': self.training_history,
            'training_time': training_time,
            'final_epoch': self.current_epoch
        }
        
        # Save results to file
        results_path = Path(self.config.log_dir) / 'training_results.json'
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for key, value in results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: float(v) if isinstance(v, np.floating) else v 
                                       for k, v in value.items()}
                else:
                    json_results[key] = float(value) if isinstance(value, np.floating) else value
            json.dump(json_results, f, indent=2)
        
        return results
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training and validation loss
        axes[0, 0].plot(self.training_history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.training_history['val_loss'], label='Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Physics loss
        axes[0, 1].plot(self.training_history['physics_losses'], label='Physics Loss', color='red')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Physics Loss')
        axes[0, 1].set_title('Physics Constraint Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Loss ratio
        if len(self.training_history['train_loss']) > 0:
            loss_ratio = np.array(self.training_history['physics_losses']) / np.array(self.training_history['train_loss'])
            axes[1, 0].plot(loss_ratio, label='Physics/Total Loss Ratio', color='green')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].set_title('Physics to Total Loss Ratio')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Learning rate (if available)
        axes[1, 1].text(0.5, 0.5, 'Training Summary\n\n' + 
                        f'Best Val Loss: {self.best_val_loss:.6f}\n' +
                        f'Final Epoch: {self.current_epoch}\n' +
                        f'Total Epochs: {len(self.training_history["train_loss"])}',
                        transform=axes[1, 1].transAxes, ha='center', va='center',
                        fontsize=12, bbox=dict(boxstyle='round', facecolor='lightblue'))
        axes[1, 1].set_title('Training Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_training_config(input_size: int, **kwargs) -> TrainingConfig:
    """
    Create training configuration with sensible defaults.
    
    Args:
        input_size: Number of input features
        **kwargs: Additional configuration parameters
        
    Returns:
        Training configuration
    """
    config_dict = {
        'input_size': input_size,
        'hidden_size': 128,
        'num_layers': 3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'num_epochs': 100,
        'patience': 15
    }
    
    config_dict.update(kwargs)
    
    return TrainingConfig(**config_dict)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample configuration
    config = create_training_config(
        input_size=50,
        hidden_size=128,
        num_epochs=10,  # Reduced for testing
        batch_size=32
    )
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H'),
        **{f'feature_{i}': np.random.randn(n_samples) for i in range(50)},
        'energy_output': np.random.uniform(0, 100, n_samples)
    })
    
    # Feature and target columns
    feature_columns = [f'feature_{i}' for i in range(50)]
    target_column = 'energy_output'
    
    # Create trainer
    trainer = UKSolarTrainer(config)
    
    # Train model
    results = trainer.train(
        data=sample_data,
        feature_columns=feature_columns,
        target_column=target_column
    )
    
    print("Training completed!")
    print(f"Best validation loss: {results['best_val_loss']:.6f}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.6f}")
    
    # Plot training history
    trainer.plot_training_history()

