"""
Comprehensive Test Suite for UK Solar Physics-Informed Model

This module provides extensive testing for the physics-informed neural network,
including unit tests, integration tests, and physics constraint validation.

Author: Manus AI
Date: 2025-08-16
"""

import unittest
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from models.physics_informed_model import (
    PhysicsInformedSolarModel, 
    UKSolarEnsemble,
    MinimalGatedUnit,
    UKAtmosphericPhysicsModule,
    SolarGeometryModule,
    PhysicsConstraints
)
from physics.solar_geometry import calculate_uk_solar_features
from physics.uk_atmosphere import UKAtmosphericModel, calculate_uk_atmospheric_features


class TestMinimalGatedUnit(unittest.TestCase):
    """Test cases for Minimal Gated Unit (MGU) cell."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.hidden_size = 20
        self.batch_size = 5
        self.mgu = MinimalGatedUnit(self.input_size, self.hidden_size)
        
    def test_mgu_initialization(self):
        """Test MGU initialization."""
        self.assertEqual(self.mgu.input_size, self.input_size)
        self.assertEqual(self.mgu.hidden_size, self.hidden_size)
        
        # Check layer dimensions
        self.assertEqual(self.mgu.forget_gate.in_features, self.input_size + self.hidden_size)
        self.assertEqual(self.mgu.forget_gate.out_features, self.hidden_size)
        self.assertEqual(self.mgu.candidate.in_features, self.input_size + self.hidden_size)
        self.assertEqual(self.mgu.candidate.out_features, self.hidden_size)
    
    def test_mgu_forward_pass(self):
        """Test MGU forward pass."""
        input_tensor = torch.randn(self.batch_size, self.input_size)
        
        # Test without initial hidden state
        output = self.mgu(input_tensor)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
        
        # Test with initial hidden state
        hidden = torch.randn(self.batch_size, self.hidden_size)
        output = self.mgu(input_tensor, hidden)
        self.assertEqual(output.shape, (self.batch_size, self.hidden_size))
    
    def test_mgu_gradient_flow(self):
        """Test gradient flow through MGU."""
        input_tensor = torch.randn(self.batch_size, self.input_size, requires_grad=True)
        hidden = torch.randn(self.batch_size, self.hidden_size, requires_grad=True)
        
        output = self.mgu(input_tensor, hidden)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(input_tensor.grad)
        self.assertIsNotNone(hidden.grad)
        
        # Check parameter gradients
        for param in self.mgu.parameters():
            self.assertIsNotNone(param.grad)


class TestUKAtmosphericPhysicsModule(unittest.TestCase):
    """Test cases for UK atmospheric physics module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_size = 50
        self.batch_size = 10
        self.atm_module = UKAtmosphericPhysicsModule(self.feature_size)
        
    def test_atmospheric_module_initialization(self):
        """Test atmospheric module initialization."""
        self.assertEqual(self.atm_module.feature_size, self.feature_size)
        
        # Check network architectures
        self.assertIsInstance(self.atm_module.atf_network, torch.nn.Sequential)
        self.assertIsInstance(self.atm_module.clear_sky_network, torch.nn.Sequential)
        self.assertIsInstance(self.atm_module.cloud_network, torch.nn.Sequential)
    
    def test_atmospheric_forward_pass(self):
        """Test atmospheric module forward pass."""
        features = torch.randn(self.batch_size, self.feature_size)
        
        outputs = self.atm_module(features)
        
        # Check output structure
        self.assertIn('atmospheric_transparency', outputs)
        self.assertIn('clear_sky_enhancement', outputs)
        self.assertIn('cloud_transmission', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['atmospheric_transparency'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['clear_sky_enhancement'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['cloud_transmission'].shape, (self.batch_size, 3))
        
        # Check output ranges
        atf = outputs['atmospheric_transparency']
        self.assertTrue(torch.all(atf >= 0) and torch.all(atf <= 1))
        
        cloud_trans = outputs['cloud_transmission']
        self.assertTrue(torch.all(cloud_trans >= 0) and torch.all(cloud_trans <= 1))


class TestSolarGeometryModule(unittest.TestCase):
    """Test cases for solar geometry module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 10
        self.solar_module = SolarGeometryModule()
        
    def test_solar_geometry_forward_pass(self):
        """Test solar geometry module forward pass."""
        # Create realistic solar geometry features
        zenith = torch.uniform(0, 90, (self.batch_size,))  # 0-90 degrees
        azimuth = torch.uniform(0, 360, (self.batch_size,))  # 0-360 degrees
        elevation = 90 - zenith  # Complementary to zenith
        air_mass = 1.0 / torch.cos(torch.deg2rad(zenith.clamp(0, 89)))  # Realistic air mass
        
        solar_features = torch.stack([zenith, azimuth, elevation, air_mass], dim=1)
        
        outputs = self.solar_module(solar_features)
        
        # Check output structure
        self.assertIn('elevation_constraint', outputs)
        self.assertIn('air_mass_constraint', outputs)
        self.assertIn('daylight_mask', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['elevation_constraint'].shape, (self.batch_size,))
        self.assertEqual(outputs['air_mass_constraint'].shape, (self.batch_size,))
        self.assertEqual(outputs['daylight_mask'].shape, (self.batch_size,))
        
        # Check constraint values are reasonable
        elevation_constraint = outputs['elevation_constraint']
        self.assertTrue(torch.all(elevation_constraint >= 0))
        
        # Daylight mask should be boolean
        daylight_mask = outputs['daylight_mask']
        self.assertTrue(daylight_mask.dtype == torch.bool)


class TestPhysicsInformedSolarModel(unittest.TestCase):
    """Test cases for the complete physics-informed solar model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.hidden_size = 64
        self.num_layers = 2
        self.batch_size = 8
        self.seq_len = 24
        
        self.model = PhysicsInformedSolarModel(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.1
        )
        
    def test_model_initialization(self):
        """Test model initialization."""
        self.assertEqual(self.model.input_size, self.input_size)
        self.assertEqual(self.model.hidden_size, self.hidden_size)
        self.assertEqual(self.model.num_layers, self.num_layers)
        
        # Check model components
        self.assertIsNotNone(self.model.input_projection)
        self.assertIsNotNone(self.model.mgu_layers)
        self.assertIsNotNone(self.model.atmospheric_physics)
        self.assertIsNotNone(self.model.solar_geometry)
        self.assertIsNotNone(self.model.output_layers)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        solar_geometry = torch.randn(self.batch_size, 4)
        
        outputs = self.model(x, solar_geometry)
        
        # Check output structure
        self.assertIn('prediction', outputs)
        self.assertIn('base_prediction', outputs)
        self.assertIn('physics_adjustment', outputs)
        self.assertIn('atmospheric_outputs', outputs)
        self.assertIn('geometry_outputs', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['prediction'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['base_prediction'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['physics_adjustment'].shape, (self.batch_size, 1))
    
    def test_model_without_solar_geometry(self):
        """Test model forward pass without solar geometry features."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
        outputs = self.model(x)
        
        # Should still produce valid outputs
        self.assertIn('prediction', outputs)
        self.assertEqual(outputs['prediction'].shape, (self.batch_size, 1))
    
    def test_physics_loss_calculation(self):
        """Test physics loss calculation."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        targets = torch.randn(self.batch_size, 1)
        
        outputs = self.model(x)
        physics_losses = self.model.calculate_physics_loss(outputs, targets, x)
        
        # Check loss components
        expected_losses = [
            'energy_conservation', 'thermodynamic', 'atmospheric', 
            'temporal_consistency', 'solar_geometry'
        ]
        
        for loss_name in expected_losses:
            self.assertIn(loss_name, physics_losses)
            self.assertIsInstance(physics_losses[loss_name], torch.Tensor)
            self.assertTrue(physics_losses[loss_name] >= 0)
    
    def test_total_loss_computation(self):
        """Test total loss computation."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        targets = torch.randn(self.batch_size, 1)
        
        outputs = self.model(x)
        total_loss, loss_components = self.model.compute_total_loss(outputs, targets, x)
        
        # Check total loss
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertTrue(total_loss >= 0)
        
        # Check loss components
        self.assertIn('data_loss', loss_components)
        self.assertIn('total_loss', loss_components)
        
        # Total loss should be sum of components
        self.assertAlmostEqual(
            total_loss.item(), 
            loss_components['total_loss'].item(), 
            places=5
        )
    
    def test_model_gradient_flow(self):
        """Test gradient flow through the model."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size, requires_grad=True)
        targets = torch.randn(self.batch_size, 1)
        
        outputs = self.model(x)
        total_loss, _ = self.model.compute_total_loss(outputs, targets, x)
        
        total_loss.backward()
        
        # Check input gradients
        self.assertIsNotNone(x.grad)
        
        # Check parameter gradients
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"No gradient for parameter: {name}")


class TestUKSolarEnsemble(unittest.TestCase):
    """Test cases for UK solar ensemble model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 50
        self.num_models = 3
        self.batch_size = 8
        self.seq_len = 24
        
        self.ensemble = UKSolarEnsemble(
            input_size=self.input_size,
            num_models=self.num_models
        )
        
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        self.assertEqual(self.ensemble.num_models, self.num_models)
        self.assertEqual(len(self.ensemble.models), self.num_models)
        
        # Check that models have different configurations
        hidden_sizes = [model.hidden_size for model in self.ensemble.models]
        self.assertEqual(len(set(hidden_sizes)), self.num_models)  # All different
    
    def test_ensemble_forward_pass(self):
        """Test ensemble forward pass."""
        x = torch.randn(self.batch_size, self.seq_len, self.input_size)
        
        outputs = self.ensemble(x)
        
        # Check output structure
        self.assertIn('prediction', outputs)
        self.assertIn('individual_predictions', outputs)
        self.assertIn('ensemble_weights', outputs)
        self.assertIn('prediction_uncertainty', outputs)
        
        # Check output shapes
        self.assertEqual(outputs['prediction'].shape, (self.batch_size, 1))
        self.assertEqual(outputs['individual_predictions'].shape, (self.batch_size, 1, self.num_models))
        self.assertEqual(outputs['ensemble_weights'].shape, (self.batch_size, self.num_models))
        self.assertEqual(outputs['prediction_uncertainty'].shape, (self.batch_size, 1))
        
        # Check ensemble weights sum to 1
        weights = outputs['ensemble_weights']
        weight_sums = torch.sum(weights, dim=1)
        torch.testing.assert_close(weight_sums, torch.ones(self.batch_size), rtol=1e-5, atol=1e-5)


class TestPhysicsConstraints(unittest.TestCase):
    """Test cases for physics constraints validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = PhysicsInformedSolarModel(
            input_size=20,
            hidden_size=32,
            num_layers=2
        )
        
    def test_energy_conservation_constraint(self):
        """Test energy conservation physics constraint."""
        batch_size = 5
        seq_len = 12
        
        # Create input with very high solar irradiance (should trigger constraint)
        x = torch.randn(batch_size, seq_len, 20)
        targets = torch.ones(batch_size, 1) * 2000  # Very high energy output
        
        outputs = self.model(x)
        physics_losses = self.model.calculate_physics_loss(outputs, targets, x)
        
        # Energy conservation loss should be calculated
        self.assertIn('energy_conservation', physics_losses)
        self.assertIsInstance(physics_losses['energy_conservation'], torch.Tensor)
    
    def test_thermodynamic_constraint(self):
        """Test thermodynamic physics constraint."""
        batch_size = 5
        seq_len = 12
        
        x = torch.randn(batch_size, seq_len, 20)
        targets = torch.randn(batch_size, 1)
        
        # Force negative predictions to test thermodynamic constraint
        outputs = self.model(x)
        outputs['prediction'] = -torch.abs(outputs['prediction'])  # Force negative
        
        physics_losses = self.model.calculate_physics_loss(outputs, targets, x)
        
        # Thermodynamic loss should be positive (penalizing negative energy)
        self.assertTrue(physics_losses['thermodynamic'] > 0)
    
    def test_atmospheric_transparency_constraint(self):
        """Test atmospheric transparency constraint."""
        batch_size = 5
        seq_len = 12
        
        x = torch.randn(batch_size, seq_len, 20)
        targets = torch.randn(batch_size, 1)
        
        outputs = self.model(x)
        
        # Check ATF is within reasonable bounds
        atf = outputs['atmospheric_outputs']['atmospheric_transparency']
        self.assertTrue(torch.all(atf >= 0))
        self.assertTrue(torch.all(atf <= 1))
    
    def test_temporal_consistency_constraint(self):
        """Test temporal consistency constraint."""
        batch_size = 10  # Need multiple samples for temporal constraint
        seq_len = 12
        
        x = torch.randn(batch_size, seq_len, 20)
        targets = torch.randn(batch_size, 1)
        
        outputs = self.model(x)
        physics_losses = self.model.calculate_physics_loss(outputs, targets, x)
        
        # Temporal consistency loss should be calculated
        self.assertIn('temporal_consistency', physics_losses)
        self.assertIsInstance(physics_losses['temporal_consistency'], torch.Tensor)


class TestUKAtmosphericModel(unittest.TestCase):
    """Test cases for UK atmospheric model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latitude = 51.5074  # London
        self.longitude = -0.1278
        self.elevation = 50.0
        
        self.atm_model = UKAtmosphericModel(
            self.latitude, self.longitude, self.elevation
        )
        
    def test_atmospheric_model_initialization(self):
        """Test atmospheric model initialization."""
        self.assertEqual(self.atm_model.latitude, self.latitude)
        self.assertEqual(self.atm_model.longitude, self.longitude)
        self.assertEqual(self.atm_model.elevation, self.elevation)
        
        # Check region determination
        self.assertIn(self.atm_model.region, [
            'scotland', 'northern_england', 'midlands', 'wales', 'southern_england'
        ])
    
    def test_linke_turbidity_calculation(self):
        """Test Linke turbidity calculation."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)  # Summer noon
        
        turbidity = self.atm_model.linke_turbidity(dt, 'average')
        
        # Check reasonable range for UK
        self.assertTrue(1.5 <= turbidity <= 8.0)
        
        # Test different atmospheric conditions
        clean_turbidity = self.atm_model.linke_turbidity(dt, 'clean')
        polluted_turbidity = self.atm_model.linke_turbidity(dt, 'polluted')
        
        self.assertTrue(clean_turbidity < polluted_turbidity)
    
    def test_clear_sky_irradiance_calculation(self):
        """Test clear sky irradiance calculation."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        zenith_angle = 30.0  # High sun
        air_mass = 1.15
        extraterrestrial_irradiance = 1361.0
        
        ghi, dni, dhi = self.atm_model.ineichen_perez_clear_sky(
            dt, zenith_angle, air_mass, extraterrestrial_irradiance
        )
        
        # Check reasonable values
        self.assertTrue(0 <= ghi <= 1200)
        self.assertTrue(0 <= dni <= 1000)
        self.assertTrue(0 <= dhi <= 500)
        
        # GHI should be sum of direct and diffuse components
        cos_zenith = np.cos(np.radians(zenith_angle))
        expected_ghi = dni * cos_zenith + dhi
        self.assertAlmostEqual(ghi, expected_ghi, delta=10)
    
    def test_atmospheric_transparency_factor(self):
        """Test atmospheric transparency factor calculation."""
        measured_ghi = 500.0
        clear_sky_ghi = 600.0
        
        atf = self.atm_model.atmospheric_transparency_factor(measured_ghi, clear_sky_ghi)
        
        # Should be ratio of measured to clear sky
        expected_atf = measured_ghi / clear_sky_ghi
        self.assertAlmostEqual(atf, expected_atf, places=3)
        
        # Test bounds
        self.assertTrue(0 <= atf <= 1.3)
    
    def test_uk_cloud_model(self):
        """Test UK cloud model."""
        dt = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        cloud_cover = 0.5
        
        cloud_factors = self.atm_model.uk_cloud_model(cloud_cover, dt)
        
        # Check output structure
        self.assertIn('direct_transmission', cloud_factors)
        self.assertIn('diffuse_enhancement', cloud_factors)
        self.assertIn('global_transmission', cloud_factors)
        
        # Check reasonable ranges
        self.assertTrue(0 <= cloud_factors['direct_transmission'] <= 1)
        self.assertTrue(0.5 <= cloud_factors['diffuse_enhancement'] <= 1.5)
        self.assertTrue(0.1 <= cloud_factors['global_transmission'] <= 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='H', tz='UTC'),
            'temperature': np.random.normal(10, 5, 100),
            'humidity': np.random.normal(75, 15, 100),
            'pressure': np.random.normal(1013, 10, 100),
            'wind_speed': np.random.uniform(0, 15, 100),
            'wind_direction': np.random.uniform(0, 360, 100),
            'cloud_cover': np.random.uniform(0, 1, 100),
            'ghi': np.random.uniform(0, 800, 100),
            'dni': np.random.uniform(0, 900, 100),
            'dhi': np.random.uniform(0, 400, 100),
            'energy_output': np.random.uniform(0, 40, 100)
        })
        
        self.latitude = 51.5074
        self.longitude = -0.1278
    
    def test_solar_geometry_integration(self):
        """Test integration with solar geometry calculations."""
        # Calculate solar features
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.latitude, self.longitude
        )
        
        # Check that solar features were added
        solar_columns = [
            'solar_zenith', 'solar_azimuth', 'solar_elevation', 
            'air_mass', 'extraterrestrial_irradiance'
        ]
        
        for col in solar_columns:
            self.assertIn(col, enhanced_data.columns)
            
        # Check reasonable ranges
        self.assertTrue(enhanced_data['solar_zenith'].between(0, 180).all())
        self.assertTrue(enhanced_data['solar_azimuth'].between(0, 360).all())
        self.assertTrue(enhanced_data['solar_elevation'].between(-90, 90).all())
    
    def test_atmospheric_features_integration(self):
        """Test integration with atmospheric features."""
        # First add solar features
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.latitude, self.longitude
        )
        
        # Then add atmospheric features
        final_data = calculate_uk_atmospheric_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        # Check atmospheric features were added
        atmospheric_columns = [
            'linke_turbidity', 'precipitable_water', 'aerosol_optical_depth',
            'cloud_direct_transmission', 'cloud_diffuse_enhancement'
        ]
        
        for col in atmospheric_columns:
            self.assertIn(col, final_data.columns)
    
    def test_end_to_end_prediction(self):
        """Test end-to-end prediction pipeline."""
        # Prepare data
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.latitude, self.longitude
        )
        final_data = calculate_uk_atmospheric_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        # Create model
        input_size = 20  # Simplified for testing
        model = PhysicsInformedSolarModel(
            input_size=input_size,
            hidden_size=32,
            num_layers=2
        )
        
        # Create dummy input
        batch_size = 5
        seq_len = 12
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Test prediction
        model.eval()
        with torch.no_grad():
            outputs = model(x)
        
        # Check outputs
        self.assertIn('prediction', outputs)
        self.assertEqual(outputs['prediction'].shape, (batch_size, 1))
        
        # Predictions should be non-negative (energy output)
        predictions = outputs['prediction']
        self.assertTrue(torch.all(predictions >= 0))


class TestModelPerformance(unittest.TestCase):
    """Performance and benchmark tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = PhysicsInformedSolarModel(
            input_size=50,
            hidden_size=128,
            num_layers=3
        )
        
    def test_inference_speed(self):
        """Test model inference speed."""
        import time
        
        batch_size = 32
        seq_len = 24
        input_size = 50
        
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Warm up
        with torch.no_grad():
            _ = self.model(x)
        
        # Time inference
        start_time = time.time()
        num_runs = 100
        
        with torch.no_grad():
            for _ in range(num_runs):
                _ = self.model(x)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_runs
        
        # Should be reasonably fast (< 100ms per batch)
        self.assertLess(avg_time, 0.1, f"Inference too slow: {avg_time:.3f}s per batch")
    
    def test_memory_usage(self):
        """Test model memory usage."""
        batch_size = 32
        seq_len = 24
        input_size = 50
        
        x = torch.randn(batch_size, seq_len, input_size)
        
        # Check model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Should have reasonable number of parameters (< 1M for this config)
        self.assertLess(total_params, 1_000_000)
        self.assertEqual(total_params, trainable_params)  # All parameters should be trainable
    
    def test_gradient_computation_speed(self):
        """Test gradient computation speed."""
        import time
        
        batch_size = 16
        seq_len = 24
        input_size = 50
        
        x = torch.randn(batch_size, seq_len, input_size, requires_grad=True)
        targets = torch.randn(batch_size, 1)
        
        # Time forward and backward pass
        start_time = time.time()
        
        outputs = self.model(x)
        loss, _ = self.model.compute_total_loss(outputs, targets, x)
        loss.backward()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete within reasonable time (< 1s)
        self.assertLess(total_time, 1.0, f"Gradient computation too slow: {total_time:.3f}s")


def run_all_tests():
    """Run all test suites."""
    # Create test suite
    test_classes = [
        TestMinimalGatedUnit,
        TestUKAtmosphericPhysicsModule,
        TestSolarGeometryModule,
        TestPhysicsInformedSolarModel,
        TestUKSolarEnsemble,
        TestPhysicsConstraints,
        TestUKAtmosphericModel,
        TestIntegration,
        TestModelPerformance
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    # Run all tests
    result = run_all_tests()
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) + len(result.errors)) == 0 else 1
    exit(exit_code)

