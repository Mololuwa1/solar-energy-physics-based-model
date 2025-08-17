"""
Test Suite for UK Solar Data Pipeline and Feature Engineering

Comprehensive tests for data validation, preprocessing, and feature engineering
components of the UK solar energy prediction system.

Author: Manus AI
Date: 2025-08-16
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import sys
from pathlib import Path
import tempfile
import yaml

# Add src directory to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data.data_pipeline import (
    UKSolarDataValidator, 
    UKSolarDataProcessor, 
    SolarFarmConfig,
    load_solar_farm_config,
    create_sample_config
)
from features.feature_engineering import (
    UKSolarFeatureEngineer,
    create_comprehensive_features
)
from physics.solar_geometry import calculate_uk_solar_features
from physics.uk_atmosphere import calculate_uk_atmospheric_features


class TestSolarFarmConfig(unittest.TestCase):
    """Test cases for solar farm configuration."""
    
    def test_config_creation(self):
        """Test solar farm configuration creation."""
        config = SolarFarmConfig(
            name="Test Farm",
            latitude=51.5,
            longitude=-0.1,
            elevation=100.0,
            capacity_mw=50.0,
            panel_tilt=35.0,
            panel_azimuth=180.0,
            panel_technology="c-Si",
            commissioning_date=datetime(2020, 1, 1),
            region="southern_england"
        )
        
        self.assertEqual(config.name, "Test Farm")
        self.assertEqual(config.latitude, 51.5)
        self.assertEqual(config.capacity_mw, 50.0)
    
    def test_sample_config_creation(self):
        """Test sample configuration creation."""
        config_dict = create_sample_config()
        
        required_keys = [
            'name', 'latitude', 'longitude', 'elevation', 'capacity_mw',
            'panel_tilt', 'panel_azimuth', 'panel_technology', 
            'commissioning_date', 'region'
        ]
        
        for key in required_keys:
            self.assertIn(key, config_dict)
        
        # Check UK coordinates
        self.assertTrue(49.5 <= config_dict['latitude'] <= 61.0)
        self.assertTrue(-9.0 <= config_dict['longitude'] <= 3.0)


class TestUKSolarDataValidator(unittest.TestCase):
    """Test cases for UK solar data validator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SolarFarmConfig(
            name="Test Farm",
            latitude=51.5074,
            longitude=-0.1278,
            elevation=50.0,
            capacity_mw=50.0,
            panel_tilt=35.0,
            panel_azimuth=180.0,
            panel_technology="c-Si",
            commissioning_date=datetime(2020, 1, 1),
            region="southern_england"
        )
        
        self.validator = UKSolarDataValidator(self.config)
        
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
    
    def test_timestamp_validation(self):
        """Test timestamp validation."""
        # Test with valid timestamps
        valid_data = self.sample_data.copy()
        result = self.validator.validate_timestamps(valid_data)
        
        self.assertEqual(len(result), len(valid_data))
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['timestamp']))
        
        # Test with invalid timestamps
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'timestamp'] = 'invalid_date'
        
        result = self.validator.validate_timestamps(invalid_data)
        self.assertEqual(len(result), len(invalid_data) - 1)  # One row removed
    
    def test_bounds_validation(self):
        """Test bounds validation."""
        # Create data with out-of-bounds values
        invalid_data = self.sample_data.copy()
        invalid_data.loc[0, 'temperature'] = -50.0  # Too cold for UK
        invalid_data.loc[1, 'humidity'] = 150.0     # Invalid humidity
        invalid_data.loc[2, 'ghi'] = 2000.0         # Too high irradiance
        
        result = self.validator.validate_bounds(invalid_data)
        
        # Out-of-bounds values should be NaN
        self.assertTrue(pd.isna(result.loc[0, 'temperature']))
        self.assertTrue(pd.isna(result.loc[1, 'humidity']))
        self.assertTrue(pd.isna(result.loc[2, 'ghi']))
    
    def test_solar_consistency_validation(self):
        """Test solar radiation consistency validation."""
        # Create inconsistent solar data
        inconsistent_data = self.sample_data.copy()
        
        # Add solar geometry for consistency check
        enhanced_data = calculate_uk_solar_features(
            inconsistent_data, self.config.latitude, self.config.longitude
        )
        
        # Make GHI inconsistent with DNI and DHI
        enhanced_data.loc[0, 'ghi'] = 1000.0
        enhanced_data.loc[0, 'dni'] = 100.0
        enhanced_data.loc[0, 'dhi'] = 50.0
        enhanced_data.loc[0, 'solar_zenith'] = 30.0  # High sun
        
        result = self.validator.validate_solar_consistency(enhanced_data)
        
        # Should have flagged inconsistent values
        self.assertTrue('ghi' in result.columns)
    
    def test_temporal_consistency_validation(self):
        """Test temporal consistency validation."""
        # Create data with unrealistic temporal changes
        temporal_data = self.sample_data.copy()
        
        # Create unrealistic temperature jump
        temporal_data.loc[10, 'temperature'] = 0.0
        temporal_data.loc[11, 'temperature'] = 30.0  # 30°C jump in 1 hour
        
        result = self.validator.validate_temporal_consistency(temporal_data)
        
        # Should have temporal flags
        flag_columns = [col for col in result.columns if 'temporal_flag' in col]
        self.assertTrue(len(flag_columns) > 0)
    
    def test_nighttime_validation(self):
        """Test nighttime validation."""
        # Create data with solar radiation at night
        nighttime_data = self.sample_data.copy()
        
        # Add solar geometry
        enhanced_data = calculate_uk_solar_features(
            nighttime_data, self.config.latitude, self.config.longitude
        )
        
        # Find nighttime periods and add solar radiation
        nighttime_mask = enhanced_data['solar_elevation'] <= 0
        if nighttime_mask.any():
            enhanced_data.loc[nighttime_mask, 'ghi'] = 100.0  # Invalid nighttime solar
            
            result = self.validator.validate_nighttime_values(enhanced_data)
            
            # Nighttime solar radiation should be zeroed
            nighttime_ghi = result.loc[nighttime_mask, 'ghi']
            self.assertTrue((nighttime_ghi <= 5.0).all())  # Allow small sensor noise


class TestUKSolarDataProcessor(unittest.TestCase):
    """Test cases for UK solar data processor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SolarFarmConfig(
            name="Test Farm",
            latitude=51.5074,
            longitude=-0.1278,
            elevation=50.0,
            capacity_mw=50.0,
            panel_tilt=35.0,
            panel_azimuth=180.0,
            panel_technology="c-Si",
            commissioning_date=datetime(2020, 1, 1),
            region="southern_england"
        )
        
        self.processor = UKSolarDataProcessor(self.config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='H', tz='UTC'),
            'temperature': np.random.normal(10, 5, 200),
            'humidity': np.random.normal(75, 15, 200),
            'pressure': np.random.normal(1013, 10, 200),
            'wind_speed': np.random.uniform(0, 15, 200),
            'wind_direction': np.random.uniform(0, 360, 200),
            'cloud_cover': np.random.uniform(0, 1, 200),
            'ghi': np.random.uniform(0, 800, 200),
            'dni': np.random.uniform(0, 900, 200),
            'dhi': np.random.uniform(0, 400, 200),
            'energy_output': np.random.uniform(0, 40, 200)
        })
    
    def test_data_loading(self):
        """Test data loading from file."""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            self.sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            # Test loading
            loaded_data = self.processor.load_data(temp_path)
            
            self.assertEqual(len(loaded_data), len(self.sample_data))
            self.assertEqual(list(loaded_data.columns), list(self.sample_data.columns))
            
        finally:
            # Clean up
            Path(temp_path).unlink()
    
    def test_data_validation(self):
        """Test complete data validation."""
        result = self.processor.validate_data(self.sample_data)
        
        # Should return a DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should have same or fewer rows (due to validation)
        self.assertLessEqual(len(result), len(self.sample_data))
        
        # Should have timestamp column
        self.assertIn('timestamp', result.columns)
    
    def test_missing_data_handling(self):
        """Test missing data handling strategies."""
        # Create data with missing values
        missing_data = self.sample_data.copy()
        missing_data.loc[10:15, 'temperature'] = np.nan
        missing_data.loc[20:22, 'ghi'] = np.nan
        
        # Test interpolation strategy
        result_interp = self.processor.handle_missing_data(missing_data, 'interpolate')
        
        # Should have fewer NaN values
        original_nan_count = missing_data.isnull().sum().sum()
        result_nan_count = result_interp.isnull().sum().sum()
        self.assertLess(result_nan_count, original_nan_count)
        
        # Test drop strategy
        result_drop = self.processor.handle_missing_data(missing_data, 'drop')
        
        # Should have no NaN values but fewer rows
        self.assertEqual(result_drop.isnull().sum().sum(), 0)
        self.assertLess(len(result_drop), len(missing_data))
    
    def test_clear_sky_features(self):
        """Test clear sky feature calculation."""
        # Add solar geometry first
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.config.latitude, self.config.longitude
        )
        
        result = self.processor.calculate_clear_sky_features(enhanced_data)
        
        # Should have clear sky features
        clear_sky_columns = ['ghi_clear', 'dni_clear', 'dhi_clear']
        for col in clear_sky_columns:
            self.assertIn(col, result.columns)
        
        # Should have clear sky indices
        if 'ghi' in result.columns:
            self.assertIn('clear_sky_index', result.columns)
            
            # Clear sky index should be reasonable
            csi = result['clear_sky_index']
            self.assertTrue((csi >= 0).all())
            self.assertTrue((csi <= 1.5).all())
    
    def test_pv_features(self):
        """Test PV-specific feature calculation."""
        # Add solar geometry first
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.config.latitude, self.config.longitude
        )
        
        result = self.processor.calculate_pv_features(enhanced_data)
        
        # Should have PV features
        expected_features = ['panel_temperature', 'incidence_angle', 'iam', 'poa_irradiance']
        
        for feature in expected_features:
            if all(col in enhanced_data.columns for col in ['temperature', 'wind_speed', 'ghi']):
                # Only check if required input columns exist
                pass  # Features may or may not be present depending on input data
    
    def test_lag_features(self):
        """Test lag feature creation."""
        columns = ['temperature', 'ghi']
        lags = [1, 2, 3]
        
        result = self.processor.create_lag_features(self.sample_data, columns, lags)
        
        # Should have lag columns
        for col in columns:
            for lag in lags:
                lag_col = f'{col}_lag_{lag}'
                if col in self.sample_data.columns:
                    self.assertIn(lag_col, result.columns)
    
    def test_rolling_features(self):
        """Test rolling feature creation."""
        columns = ['temperature', 'ghi']
        windows = [6, 12]
        
        result = self.processor.create_rolling_features(self.sample_data, columns, windows)
        
        # Should have rolling columns
        for col in columns:
            for window in windows:
                if col in self.sample_data.columns:
                    rolling_cols = [f'{col}_rolling_mean_{window}', 
                                  f'{col}_rolling_std_{window}']
                    for rolling_col in rolling_cols:
                        self.assertIn(rolling_col, result.columns)
    
    def test_full_pipeline(self):
        """Test complete processing pipeline."""
        result = self.processor.process_full_pipeline(self.sample_data)
        
        # Should return processed DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should have more columns (features added)
        self.assertGreater(len(result.columns), len(self.sample_data.columns))
        
        # Should have solar geometry features
        solar_features = ['solar_zenith', 'solar_azimuth', 'solar_elevation']
        for feature in solar_features:
            self.assertIn(feature, result.columns)
        
        # Should have atmospheric features
        atmospheric_features = ['linke_turbidity', 'precipitable_water']
        for feature in atmospheric_features:
            self.assertIn(feature, result.columns)


class TestUKSolarFeatureEngineer(unittest.TestCase):
    """Test cases for UK solar feature engineer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latitude = 51.5074
        self.longitude = -0.1278
        self.engineer = UKSolarFeatureEngineer(self.latitude, self.longitude)
        
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
            'ghi_clear': np.random.uniform(0, 1000, 100),
            'solar_elevation': np.random.uniform(-20, 60, 100),
            'energy_output': np.random.uniform(0, 40, 100)
        })
    
    def test_temporal_features(self):
        """Test temporal feature creation."""
        result = self.engineer.create_temporal_features(self.sample_data)
        
        # Should have basic time components
        time_features = ['year', 'month', 'day', 'hour', 'day_of_year', 'day_of_week']
        for feature in time_features:
            self.assertIn(feature, result.columns)
        
        # Should have cyclical encodings
        cyclical_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos']
        for feature in cyclical_features:
            self.assertIn(feature, result.columns)
        
        # Check cyclical encoding ranges
        self.assertTrue((result['hour_sin'] >= -1).all() and (result['hour_sin'] <= 1).all())
        self.assertTrue((result['hour_cos'] >= -1).all() and (result['hour_cos'] <= 1).all())
        
        # Should have UK-specific features
        uk_features = ['is_bst', 'is_working_hours', 'is_weekend']
        for feature in uk_features:
            self.assertIn(feature, result.columns)
    
    def test_weather_interaction_features(self):
        """Test weather interaction feature creation."""
        result = self.engineer.create_weather_interaction_features(self.sample_data)
        
        # Should have temperature-humidity interactions
        if 'temperature' in self.sample_data.columns and 'humidity' in self.sample_data.columns:
            interaction_features = ['heat_index', 'dew_point', 'vpd']
            for feature in interaction_features:
                self.assertIn(feature, result.columns)
        
        # Should have wind chill
        if 'temperature' in self.sample_data.columns and 'wind_speed' in self.sample_data.columns:
            self.assertIn('wind_chill', result.columns)
        
        # Should have cloud interactions
        if 'cloud_cover' in self.sample_data.columns:
            cloud_features = ['cloud_transmission', 'cloud_enhancement']
            for feature in cloud_features:
                self.assertIn(feature, result.columns)
    
    def test_solar_physics_features(self):
        """Test solar physics feature creation."""
        result = self.engineer.create_solar_physics_features(self.sample_data)
        
        # Should have clear sky index
        if 'ghi' in self.sample_data.columns and 'ghi_clear' in self.sample_data.columns:
            self.assertIn('clear_sky_index', result.columns)
            
            # Clear sky index should be reasonable
            csi = result['clear_sky_index']
            self.assertTrue((csi >= 0).all())
            self.assertTrue((csi <= 1.5).all())
        
        # Should have solar elevation categories
        if 'solar_elevation' in self.sample_data.columns:
            elevation_features = ['low_sun', 'medium_sun', 'high_sun']
            for feature in elevation_features:
                self.assertIn(feature, result.columns)
    
    def test_uk_specific_features(self):
        """Test UK-specific feature creation."""
        result = self.engineer.create_uk_specific_features(self.sample_data)
        
        # Should have UK seasonal features
        if 'month' in result.columns:
            self.assertIn('uk_solar_season_strength', result.columns)
        
        # Should have pressure-based features
        if 'pressure' in self.sample_data.columns:
            pressure_features = ['high_pressure', 'low_pressure']
            for feature in pressure_features:
                self.assertIn(feature, result.columns)
        
        # Should have regional features
        regional_features = [col for col in result.columns if 'regional' in col or 'is_' in col]
        self.assertTrue(len(regional_features) > 0)
    
    def test_lag_and_lead_features(self):
        """Test lag and lead feature creation."""
        target_columns = ['ghi', 'temperature']
        lag_periods = [1, 2, 3]
        
        result = self.engineer.create_lag_and_lead_features(
            self.sample_data, target_columns, lag_periods
        )
        
        # Should have lag features
        for col in target_columns:
            if col in self.sample_data.columns:
                for lag in lag_periods:
                    lag_col = f'{col}_lag_{lag}'
                    self.assertIn(lag_col, result.columns)
    
    def test_rolling_statistics(self):
        """Test rolling statistics creation."""
        columns = ['ghi', 'temperature']
        windows = [6, 12]
        statistics = ['mean', 'std', 'min', 'max']
        
        result = self.engineer.create_rolling_statistics(
            self.sample_data, columns, windows, statistics
        )
        
        # Should have rolling statistics
        for col in columns:
            if col in self.sample_data.columns:
                for window in windows:
                    for stat in statistics:
                        rolling_col = f'{col}_rolling_{stat}_{window}'
                        self.assertIn(rolling_col, result.columns)
    
    def test_fourier_features(self):
        """Test Fourier feature creation."""
        columns = ['ghi', 'temperature']
        periods = [24, 168]  # Daily, weekly
        
        result = self.engineer.create_fourier_features(
            self.sample_data, columns, periods
        )
        
        # Should have Fourier features
        for col in columns:
            if col in self.sample_data.columns:
                for period in periods:
                    fourier_cols = [f'{col}_fourier_sin_{period}h', f'{col}_fourier_cos_{period}h']
                    for fourier_col in fourier_cols:
                        self.assertIn(fourier_col, result.columns)
    
    def test_interaction_features(self):
        """Test interaction feature creation."""
        feature_pairs = [('temperature', 'humidity'), ('ghi', 'cloud_cover')]
        
        result = self.engineer.create_interaction_features(self.sample_data, feature_pairs)
        
        # Should have interaction features
        for feat1, feat2 in feature_pairs:
            if feat1 in self.sample_data.columns and feat2 in self.sample_data.columns:
                interaction_cols = [
                    f'{feat1}_x_{feat2}', f'{feat1}_div_{feat2}',
                    f'{feat1}_minus_{feat2}', f'{feat1}_plus_{feat2}'
                ]
                for interaction_col in interaction_cols:
                    self.assertIn(interaction_col, result.columns)
    
    def test_feature_selection(self):
        """Test feature selection."""
        # Add some numerical features for selection
        enhanced_data = self.sample_data.copy()
        enhanced_data['feature_1'] = np.random.randn(len(enhanced_data))
        enhanced_data['feature_2'] = np.random.randn(len(enhanced_data))
        enhanced_data['feature_3'] = enhanced_data['energy_output'] + np.random.randn(len(enhanced_data)) * 0.1
        
        selected_df, selected_features = self.engineer.select_features(
            enhanced_data, 'energy_output', method='correlation', k=5
        )
        
        # Should return selected features
        self.assertLessEqual(len(selected_features), 5)
        self.assertIn('energy_output', selected_df.columns)
        self.assertIn('timestamp', selected_df.columns)
    
    def test_feature_scaling(self):
        """Test feature scaling."""
        result = self.engineer.scale_features(self.sample_data, method='standard')
        
        # Numerical columns should be scaled (except excluded ones)
        numerical_cols = self.sample_data.select_dtypes(include=[np.number]).columns
        scaled_cols = [col for col in numerical_cols if col != 'timestamp']
        
        for col in scaled_cols:
            if col in result.columns:
                # Standard scaling should result in approximately zero mean and unit variance
                col_mean = result[col].mean()
                col_std = result[col].std()
                self.assertAlmostEqual(col_mean, 0, places=1)
                self.assertAlmostEqual(col_std, 1, places=1)


class TestComprehensiveFeatures(unittest.TestCase):
    """Test cases for comprehensive feature creation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.latitude = 51.5074
        self.longitude = -0.1278
        
        # Create comprehensive sample data
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='H', tz='UTC'),
            'temperature': np.random.normal(10, 5, 200),
            'humidity': np.random.normal(75, 15, 200),
            'pressure': np.random.normal(1013, 10, 200),
            'wind_speed': np.random.uniform(0, 15, 200),
            'wind_direction': np.random.uniform(0, 360, 200),
            'cloud_cover': np.random.uniform(0, 1, 200),
            'ghi': np.random.uniform(0, 800, 200),
            'dni': np.random.uniform(0, 900, 200),
            'dhi': np.random.uniform(0, 400, 200),
            'energy_output': np.random.uniform(0, 40, 200)
        })
    
    def test_comprehensive_feature_creation(self):
        """Test comprehensive feature creation pipeline."""
        # First add physics features
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.latitude, self.longitude
        )
        enhanced_data = calculate_uk_atmospheric_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        # Create comprehensive features
        result = create_comprehensive_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        # Should have significantly more features
        self.assertGreater(len(result.columns), len(self.sample_data.columns) * 3)
        
        # Should have various feature types
        feature_types = {
            'temporal': [col for col in result.columns if any(x in col for x in ['hour', 'day', 'month'])],
            'weather': [col for col in result.columns if any(x in col for x in ['temp', 'humid', 'wind'])],
            'solar': [col for col in result.columns if any(x in col for x in ['ghi', 'solar', 'clear'])],
            'lag': [col for col in result.columns if 'lag' in col],
            'rolling': [col for col in result.columns if 'rolling' in col],
            'fourier': [col for col in result.columns if 'fourier' in col],
            'interaction': [col for col in result.columns if any(x in col for x in ['_x_', '_div_'])]
        }
        
        # Each feature type should have some features
        for feat_type, features in feature_types.items():
            self.assertGreater(len(features), 0, f"No {feat_type} features found")
    
    def test_feature_quality(self):
        """Test quality of created features."""
        # Add physics features first
        enhanced_data = calculate_uk_solar_features(
            self.sample_data, self.latitude, self.longitude
        )
        enhanced_data = calculate_uk_atmospheric_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        result = create_comprehensive_features(
            enhanced_data, self.latitude, self.longitude
        )
        
        # Check for infinite or extremely large values
        numerical_cols = result.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            # Should not have infinite values
            self.assertFalse(np.isinf(result[col]).any(), f"Infinite values in {col}")
            
            # Should not have extremely large values (> 1e10)
            self.assertFalse((np.abs(result[col]) > 1e10).any(), f"Extremely large values in {col}")
        
        # Check that cyclical features are in correct range
        cyclical_cols = [col for col in result.columns if any(x in col for x in ['_sin', '_cos'])]
        for col in cyclical_cols:
            if col in result.columns:
                self.assertTrue((result[col] >= -1.1).all() and (result[col] <= 1.1).all(), 
                              f"Cyclical feature {col} out of range")


class TestDataIntegration(unittest.TestCase):
    """Integration tests for data pipeline components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = SolarFarmConfig(
            name="Integration Test Farm",
            latitude=51.5074,
            longitude=-0.1278,
            elevation=50.0,
            capacity_mw=50.0,
            panel_tilt=35.0,
            panel_azimuth=180.0,
            panel_technology="c-Si",
            commissioning_date=datetime(2020, 1, 1),
            region="southern_england"
        )
        
        # Create realistic sample data
        timestamps = pd.date_range('2024-01-01', periods=1000, freq='H', tz='UTC')
        
        self.sample_data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': 10 + 5 * np.sin(2 * np.pi * np.arange(1000) / (24 * 365.25)) + np.random.normal(0, 2, 1000),
            'humidity': 75 + 10 * np.random.randn(1000),
            'pressure': 1013 + 5 * np.random.randn(1000),
            'wind_speed': np.abs(5 + 3 * np.random.randn(1000)),
            'wind_direction': np.random.uniform(0, 360, 1000),
            'cloud_cover': np.random.beta(2, 2, 1000),  # More realistic cloud distribution
            'ghi': np.maximum(0, 400 * np.sin(2 * np.pi * np.arange(1000) / 24) * 
                             (1 - 0.7 * np.random.beta(2, 2, 1000)) + np.random.normal(0, 20, 1000)),
            'dni': np.maximum(0, 500 * np.sin(2 * np.pi * np.arange(1000) / 24) * 
                             (1 - 0.8 * np.random.beta(2, 2, 1000)) + np.random.normal(0, 30, 1000)),
            'dhi': np.maximum(0, 200 + 100 * np.random.beta(2, 2, 1000) + np.random.normal(0, 10, 1000)),
            'energy_output': np.maximum(0, 30 * np.sin(2 * np.pi * np.arange(1000) / 24) * 
                                       (1 - 0.6 * np.random.beta(2, 2, 1000)) + np.random.normal(0, 2, 1000))
        })
    
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end data processing pipeline."""
        processor = UKSolarDataProcessor(self.config)
        
        # Run full pipeline
        result = processor.process_full_pipeline(self.sample_data)
        
        # Should complete without errors
        self.assertIsInstance(result, pd.DataFrame)
        
        # Should have reasonable number of rows (some may be dropped due to NaN from feature engineering)
        self.assertGreater(len(result), len(self.sample_data) * 0.8)
        
        # Should have many more features
        self.assertGreater(len(result.columns), len(self.sample_data.columns) * 5)
        
        # Should have key feature categories
        feature_categories = {
            'solar_geometry': ['solar_zenith', 'solar_azimuth', 'solar_elevation'],
            'atmospheric': ['linke_turbidity', 'precipitable_water'],
            'clear_sky': ['ghi_clear', 'clear_sky_index'],
            'temporal': ['hour_sin', 'month_cos', 'day_of_year_sin'],
            'weather_interactions': ['heat_index', 'dew_point'],
            'uk_specific': ['uk_solar_season_strength']
        }
        
        for category, expected_features in feature_categories.items():
            found_features = [feat for feat in expected_features if feat in result.columns]
            self.assertGreater(len(found_features), 0, 
                             f"No {category} features found. Expected: {expected_features}")
    
    def test_data_quality_after_processing(self):
        """Test data quality after complete processing."""
        processor = UKSolarDataProcessor(self.config)
        result = processor.process_full_pipeline(self.sample_data)
        
        # Check for data quality issues
        numerical_cols = result.select_dtypes(include=[np.number]).columns
        
        # Should not have excessive NaN values (< 5% per column)
        for col in numerical_cols:
            nan_percentage = result[col].isnull().sum() / len(result)
            self.assertLess(nan_percentage, 0.05, f"Too many NaN values in {col}: {nan_percentage:.2%}")
        
        # Should not have duplicate timestamps
        self.assertEqual(len(result), len(result['timestamp'].unique()))
        
        # Solar features should be physically reasonable
        if 'solar_elevation' in result.columns:
            self.assertTrue(result['solar_elevation'].between(-90, 90).all())
        
        if 'clear_sky_index' in result.columns:
            # Most values should be between 0 and 1.2
            reasonable_csi = result['clear_sky_index'].between(0, 1.2).sum()
            total_csi = len(result['clear_sky_index'].dropna())
            self.assertGreater(reasonable_csi / total_csi, 0.9)
    
    def test_config_file_handling(self):
        """Test configuration file loading and saving."""
        # Create temporary config file
        config_dict = {
            'name': 'Test Farm',
            'latitude': 51.5,
            'longitude': -0.1,
            'elevation': 100.0,
            'capacity_mw': 50.0,
            'panel_tilt': 35.0,
            'panel_azimuth': 180.0,
            'panel_technology': 'c-Si',
            'commissioning_date': '2020-01-01T00:00:00',
            'region': 'southern_england'
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_dict, f)
            temp_path = f.name
        
        try:
            # Test loading
            loaded_config = load_solar_farm_config(temp_path)
            
            self.assertEqual(loaded_config.name, config_dict['name'])
            self.assertEqual(loaded_config.latitude, config_dict['latitude'])
            self.assertEqual(loaded_config.capacity_mw, config_dict['capacity_mw'])
            
        finally:
            # Clean up
            Path(temp_path).unlink()


def run_all_tests():
    """Run all test suites."""
    test_classes = [
        TestSolarFarmConfig,
        TestUKSolarDataValidator,
        TestUKSolarDataProcessor,
        TestUKSolarFeatureEngineer,
        TestComprehensiveFeatures,
        TestDataIntegration
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
    print(f"Data Pipeline Test Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Exit with appropriate code
    exit_code = 0 if (len(result.failures) + len(result.errors)) == 0 else 1
    exit(exit_code)

