"""
Physics-Informed Neural Network for UK Solar Energy Prediction

This module implements a sophisticated physics-informed neural network (PINN)
that incorporates atmospheric physics, solar radiation models, and UK-specific
constraints for accurate solar energy forecasting.

Author: Mololuwa Obafemi-Moses
Date: 2025-08-16
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
import math


@dataclass
class PhysicsConstraints:
    """Configuration for physics constraints in the model."""
    energy_conservation_weight: float = 0.1
    thermodynamic_weight: float = 0.05
    radiative_transfer_weight: float = 0.1
    temporal_consistency_weight: float = 0.05
    uk_atmospheric_weight: float = 0.08


class MinimalGatedUnit(nn.Module):
    """
    Minimal Gated Unit (MGU) - simplified RNN cell optimized for solar forecasting.
    
    More efficient than LSTM while maintaining temporal modeling capabilities
    essential for solar energy prediction.
    """
    
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        """
        Initialize MGU cell.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            bias: Whether to use bias terms
        """
        super(MinimalGatedUnit, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Forget gate
        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        # Candidate values
        self.candidate = nn.Linear(input_size + hidden_size, hidden_size, bias=bias)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with appropriate values."""
        std = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-std, std)
    
    def forward(self, input_tensor: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through MGU cell.
        
        Args:
            input_tensor: Input tensor [batch_size, input_size]
            hidden: Previous hidden state [batch_size, hidden_size]
            
        Returns:
            New hidden state [batch_size, hidden_size]
        """
        if hidden is None:
            hidden = torch.zeros(input_tensor.size(0), self.hidden_size,
                               dtype=input_tensor.dtype, device=input_tensor.device)
        
        # Concatenate input and hidden state
        combined = torch.cat([input_tensor, hidden], dim=1)
        
        # Forget gate
        forget = torch.sigmoid(self.forget_gate(combined))
        
        # Candidate values
        candidate = torch.tanh(self.candidate(combined))
        
        # New hidden state
        new_hidden = forget * hidden + (1 - forget) * candidate
        
        return new_hidden


class MGULayer(nn.Module):
    """
    Multi-step MGU layer for sequence processing.
    """
    
    def __init__(self, input_size: int, hidden_size: int, 
                 num_layers: int = 1, dropout: float = 0.0, 
                 batch_first: bool = True):
        """
        Initialize MGU layer.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden state
            num_layers: Number of MGU layers
            dropout: Dropout probability
            batch_first: Whether batch dimension is first
        """
        super(MGULayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        
        # Create MGU cells
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(MinimalGatedUnit(cell_input_size, hidden_size))
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
    
    def forward(self, input_seq: torch.Tensor, 
                hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MGU layer.
        
        Args:
            input_seq: Input sequence [batch_size, seq_len, input_size]
            hidden: Initial hidden state
            
        Returns:
            Tuple of (output_sequence, final_hidden_state)
        """
        if not self.batch_first:
            input_seq = input_seq.transpose(0, 1)
        
        batch_size, seq_len, _ = input_seq.size()
        
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size,
                               dtype=input_seq.dtype, device=input_seq.device)
        
        outputs = []
        
        for t in range(seq_len):
            x = input_seq[:, t, :]
            
            new_hidden = []
            for layer in range(self.num_layers):
                x = self.cells[layer](x, hidden[layer])
                if self.dropout and layer < self.num_layers - 1:
                    x = self.dropout(x)
                new_hidden.append(x)
            
            hidden = torch.stack(new_hidden)
            outputs.append(x)
        
        output_seq = torch.stack(outputs, dim=1)
        
        if not self.batch_first:
            output_seq = output_seq.transpose(0, 1)
        
        return output_seq, hidden


class UKAtmosphericPhysicsModule(nn.Module):
    """
    Neural network module that enforces UK atmospheric physics constraints.
    """
    
    def __init__(self, feature_size: int):
        """
        Initialize atmospheric physics module.
        
        Args:
            feature_size: Size of input features
        """
        super(UKAtmosphericPhysicsModule, self).__init__()
        
        self.feature_size = feature_size
        
        # Atmospheric transparency factor network
        self.atf_network = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # ATF should be between 0 and 1
        )
        
        # Clear sky enhancement network
        self.clear_sky_network = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # UK cloud transmission network
        self.cloud_network = nn.Sequential(
            nn.Linear(feature_size, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3)  # Direct, diffuse, global transmission
        )
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass through atmospheric physics module.
        
        Args:
            features: Input features
            
        Returns:
            Dictionary of atmospheric physics outputs
        """
        # Atmospheric transparency factor
        atf = self.atf_network(features)
        
        # Clear sky enhancement
        clear_sky_enhancement = self.clear_sky_network(features)
        
        # Cloud transmission factors
        cloud_transmission = self.cloud_network(features)
        cloud_transmission = torch.sigmoid(cloud_transmission)  # Ensure 0-1 range
        
        return {
            'atmospheric_transparency': atf,
            'clear_sky_enhancement': clear_sky_enhancement,
            'cloud_transmission': cloud_transmission
        }


class SolarGeometryModule(nn.Module):
    """
    Module for solar geometry calculations and constraints.
    """
    
    def __init__(self):
        """Initialize solar geometry module."""
        super(SolarGeometryModule, self).__init__()
        
    def forward(self, solar_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate solar geometry constraints.
        
        Args:
            solar_features: Solar geometry features [batch_size, solar_features]
                          Expected: [zenith, azimuth, elevation, air_mass, ...]
            
        Returns:
            Dictionary of solar geometry outputs
        """
        # Extract solar geometry components
        zenith = solar_features[:, 0]
        elevation = solar_features[:, 2]
        air_mass = solar_features[:, 3]
        
        # Solar elevation constraint (should be consistent with zenith)
        elevation_constraint = torch.abs(elevation - (90 - zenith))
        
        # Air mass constraint (should increase with zenith angle)
        expected_air_mass = 1.0 / (torch.cos(torch.deg2rad(zenith)) + 1e-6)
        air_mass_constraint = torch.abs(air_mass - expected_air_mass) / (expected_air_mass + 1e-6)
        
        # Daylight constraint (solar radiation should be zero when sun is below horizon)
        daylight_mask = elevation > 0
        
        return {
            'elevation_constraint': elevation_constraint,
            'air_mass_constraint': air_mass_constraint,
            'daylight_mask': daylight_mask
        }


class PhysicsInformedSolarModel(nn.Module):
    """
    Physics-Informed Neural Network for UK Solar Energy Prediction.
    
    Combines data-driven learning with physics constraints for improved
    accuracy and physical consistency.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 output_size: int = 1,
                 dropout: float = 0.1,
                 physics_constraints: Optional[PhysicsConstraints] = None):
        """
        Initialize physics-informed solar model.
        
        Args:
            input_size: Number of input features
            hidden_size: Hidden layer size
            num_layers: Number of recurrent layers
            output_size: Number of output predictions
            dropout: Dropout probability
            physics_constraints: Physics constraint configuration
        """
        super(PhysicsInformedSolarModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        if physics_constraints is None:
            physics_constraints = PhysicsConstraints()
        self.physics_constraints = physics_constraints
        
        # Input processing layers
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # MGU layers for temporal modeling
        self.mgu_layers = MGULayer(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Physics modules
        self.atmospheric_physics = UKAtmosphericPhysicsModule(hidden_size)
        self.solar_geometry = SolarGeometryModule()
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, output_size)
        )
        
        # Physics-informed output adjustment
        self.physics_adjustment = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size // 4),  # +4 for physics outputs
            nn.ReLU(),
            nn.Linear(hidden_size // 4, output_size),
            nn.Sigmoid()  # Adjustment factor between 0 and 1
        )
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, 
                solar_geometry_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through physics-informed model.
        
        Args:
            x: Input features [batch_size, seq_len, input_size]
            solar_geometry_features: Solar geometry features for physics constraints
            
        Returns:
            Dictionary containing predictions and physics outputs
        """
        batch_size, seq_len, _ = x.size()
        
        # Input projection
        x_proj = self.input_projection(x.view(-1, self.input_size))
        x_proj = x_proj.view(batch_size, seq_len, self.hidden_size)
        
        # MGU processing
        mgu_output, final_hidden = self.mgu_layers(x_proj)
        
        # Use final hidden state for prediction
        final_features = final_hidden[-1]  # Last layer's final hidden state
        
        # Base prediction
        base_prediction = self.output_layers(final_features)
        
        # Physics modules
        atmospheric_outputs = self.atmospheric_physics(final_features)
        
        # Solar geometry constraints (if provided)
        if solar_geometry_features is not None:
            geometry_outputs = self.solar_geometry(solar_geometry_features)
        else:
            # Create dummy outputs if no geometry features provided
            geometry_outputs = {
                'elevation_constraint': torch.zeros(batch_size, device=x.device),
                'air_mass_constraint': torch.zeros(batch_size, device=x.device),
                'daylight_mask': torch.ones(batch_size, device=x.device, dtype=torch.bool)
            }
        
        # Physics-informed adjustment
        physics_features = torch.cat([
            final_features,
            atmospheric_outputs['atmospheric_transparency'],
            atmospheric_outputs['clear_sky_enhancement'],
            geometry_outputs['elevation_constraint'].unsqueeze(1),
            geometry_outputs['air_mass_constraint'].unsqueeze(1)
        ], dim=1)
        
        physics_adjustment = self.physics_adjustment(physics_features)
        
        # Final prediction with physics adjustment
        final_prediction = base_prediction * physics_adjustment
        
        # Apply daylight constraint (zero output when sun is below horizon)
        if solar_geometry_features is not None:
            daylight_mask = geometry_outputs['daylight_mask'].unsqueeze(1)
            final_prediction = final_prediction * daylight_mask.float()
        
        return {
            'prediction': final_prediction,
            'base_prediction': base_prediction,
            'physics_adjustment': physics_adjustment,
            'atmospheric_outputs': atmospheric_outputs,
            'geometry_outputs': geometry_outputs
        }
    
    def calculate_physics_loss(self, outputs: Dict[str, torch.Tensor],
                              targets: torch.Tensor,
                              features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate physics-based loss components.
        
        Args:
            outputs: Model outputs dictionary
            targets: Target values
            features: Input features for physics calculations
            
        Returns:
            Dictionary of physics loss components
        """
        losses = {}
        
        # Energy conservation loss
        # Solar energy output should not exceed theoretical maximum
        if 'ghi_clear' in features or features.size(-1) > 10:  # Assume clear sky GHI is available
            # Simplified energy conservation check
            max_theoretical = torch.ones_like(targets) * 1000  # Simplified max value
            energy_violation = F.relu(outputs['prediction'] - max_theoretical)
            losses['energy_conservation'] = energy_violation.mean()
        else:
            losses['energy_conservation'] = torch.tensor(0.0, device=targets.device)
        
        # Thermodynamic consistency loss
        # Predictions should be physically reasonable
        negative_energy = F.relu(-outputs['prediction'])
        losses['thermodynamic'] = negative_energy.mean()
        
        # Atmospheric transparency consistency
        atf = outputs['atmospheric_outputs']['atmospheric_transparency']
        # ATF should be between 0 and 1.2 (allowing for cloud enhancement)
        atf_violation = F.relu(atf - 1.2) + F.relu(-atf)
        losses['atmospheric'] = atf_violation.mean()
        
        # Temporal consistency loss
        if outputs['prediction'].size(0) > 1:
            temporal_diff = torch.diff(outputs['prediction'], dim=0)
            # Penalize very large changes (> 50% of capacity per time step)
            max_change = 0.5 * torch.ones_like(temporal_diff)
            temporal_violation = F.relu(torch.abs(temporal_diff) - max_change)
            losses['temporal_consistency'] = temporal_violation.mean()
        else:
            losses['temporal_consistency'] = torch.tensor(0.0, device=targets.device)
        
        # Solar geometry consistency
        elevation_loss = outputs['geometry_outputs']['elevation_constraint'].mean()
        air_mass_loss = outputs['geometry_outputs']['air_mass_constraint'].mean()
        losses['solar_geometry'] = elevation_loss + air_mass_loss
        
        return losses
    
    def compute_total_loss(self, outputs: Dict[str, torch.Tensor],
                          targets: torch.Tensor,
                          features: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss including data and physics components.
        
        Args:
            outputs: Model outputs
            targets: Target values
            features: Input features
            
        Returns:
            Tuple of (total_loss, loss_components)
        """
        # Data loss (MSE)
        data_loss = F.mse_loss(outputs['prediction'], targets)
        
        # Physics losses
        physics_losses = self.calculate_physics_loss(outputs, targets, features)
        
        # Weighted total loss
        total_loss = data_loss
        
        # Add physics loss components with weights
        total_loss += (self.physics_constraints.energy_conservation_weight * 
                      physics_losses['energy_conservation'])
        total_loss += (self.physics_constraints.thermodynamic_weight * 
                      physics_losses['thermodynamic'])
        total_loss += (self.physics_constraints.uk_atmospheric_weight * 
                      physics_losses['atmospheric'])
        total_loss += (self.physics_constraints.temporal_consistency_weight * 
                      physics_losses['temporal_consistency'])
        
        # Combine all loss components
        loss_components = {
            'data_loss': data_loss,
            'total_loss': total_loss,
            **physics_losses
        }
        
        return total_loss, loss_components


class UKSolarEnsemble(nn.Module):
    """
    Ensemble of physics-informed models for improved robustness.
    """
    
    def __init__(self, 
                 input_size: int,
                 num_models: int = 3,
                 hidden_size: int = 128,
                 num_layers: int = 3,
                 dropout: float = 0.1):
        """
        Initialize ensemble model.
        
        Args:
            input_size: Number of input features
            num_models: Number of models in ensemble
            hidden_size: Hidden layer size for each model
            num_layers: Number of layers for each model
            dropout: Dropout probability
        """
        super(UKSolarEnsemble, self).__init__()
        
        self.num_models = num_models
        
        # Create ensemble of models with different configurations
        self.models = nn.ModuleList()
        for i in range(num_models):
            # Vary model configurations slightly
            model_hidden = hidden_size + (i - 1) * 16  # Vary hidden size
            model_layers = max(2, num_layers + (i - 1))  # Vary number of layers
            
            model = PhysicsInformedSolarModel(
                input_size=input_size,
                hidden_size=model_hidden,
                num_layers=model_layers,
                dropout=dropout
            )
            self.models.append(model)
        
        # Ensemble weighting network
        self.ensemble_weights = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, num_models),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x: torch.Tensor, 
                solar_geometry_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features
            solar_geometry_features: Solar geometry features
            
        Returns:
            Ensemble predictions and individual model outputs
        """
        # Get predictions from all models
        model_outputs = []
        for model in self.models:
            output = model(x, solar_geometry_features)
            model_outputs.append(output['prediction'])
        
        # Stack predictions
        all_predictions = torch.stack(model_outputs, dim=-1)  # [batch, output_size, num_models]
        
        # Calculate ensemble weights based on input
        input_for_weights = x[:, -1, :] if len(x.shape) == 3 else x  # Use last timestep
        weights = self.ensemble_weights(input_for_weights)  # [batch, num_models]
        
        # Weighted ensemble prediction
        ensemble_prediction = torch.sum(
            all_predictions * weights.unsqueeze(1), dim=-1
        )
        
        # Calculate prediction uncertainty (ensemble spread)
        prediction_std = torch.std(all_predictions, dim=-1)
        
        return {
            'prediction': ensemble_prediction,
            'individual_predictions': all_predictions,
            'ensemble_weights': weights,
            'prediction_uncertainty': prediction_std
        }


def create_uk_solar_model(config: Dict[str, Any]) -> PhysicsInformedSolarModel:
    """
    Create a configured UK solar prediction model.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured physics-informed solar model
    """
    # Physics constraints configuration
    physics_constraints = PhysicsConstraints(
        energy_conservation_weight=config.get('energy_conservation_weight', 0.1),
        thermodynamic_weight=config.get('thermodynamic_weight', 0.05),
        radiative_transfer_weight=config.get('radiative_transfer_weight', 0.1),
        temporal_consistency_weight=config.get('temporal_consistency_weight', 0.05),
        uk_atmospheric_weight=config.get('uk_atmospheric_weight', 0.08)
    )
    
    # Create model
    model = PhysicsInformedSolarModel(
        input_size=config['input_size'],
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 3),
        output_size=config.get('output_size', 1),
        dropout=config.get('dropout', 0.1),
        physics_constraints=physics_constraints
    )
    
    return model


if __name__ == "__main__":
    # Example usage and testing
    
    # Model configuration
    config = {
        'input_size': 50,  # Number of input features
        'hidden_size': 128,
        'num_layers': 3,
        'output_size': 1,
        'dropout': 0.1,
        'energy_conservation_weight': 0.1,
        'thermodynamic_weight': 0.05,
        'uk_atmospheric_weight': 0.08
    }
    
    # Create model
    model = create_uk_solar_model(config)
    
    # Test with sample data
    batch_size = 32
    seq_len = 24  # 24 hours
    input_size = config['input_size']
    
    # Sample input data
    x = torch.randn(batch_size, seq_len, input_size)
    solar_geometry = torch.randn(batch_size, 4)  # zenith, azimuth, elevation, air_mass
    targets = torch.randn(batch_size, 1)
    
    # Forward pass
    outputs = model(x, solar_geometry)
    
    # Calculate loss
    total_loss, loss_components = model.compute_total_loss(outputs, targets, x)
    
    print(f"Model created successfully!")
    print(f"Output shape: {outputs['prediction'].shape}")
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss components: {[f'{k}: {v.item():.4f}' for k, v in loss_components.items()]}")
    
    # Test ensemble model
    ensemble = UKSolarEnsemble(input_size=input_size, num_models=3)
    ensemble_outputs = ensemble(x, solar_geometry)
    
    print(f"Ensemble prediction shape: {ensemble_outputs['prediction'].shape}")
    print(f"Prediction uncertainty shape: {ensemble_outputs['prediction_uncertainty'].shape}")

