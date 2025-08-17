"""
UK Solar Farm Data Pipeline

Comprehensive data processing pipeline for UK solar farms, including data ingestion,
validation, preprocessing, and feature engineering optimized for physics-based ML models.

Author: Manus AI
Date: 2025-08-16
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from pathlib import Path
import yaml
import json
from dataclasses import dataclass
import warnings

# Import our physics modules
import sys
sys.path.append('/home/ubuntu/uk_solar_prediction/src')
from physics.solar_geometry import calculate_uk_solar_features
from physics.uk_atmosphere import calculate_uk_atmospheric_features


@dataclass
class SolarFarmConfig:
    """Configuration for a UK solar farm."""
    name: str
    latitude: float
    longitude: float
    elevation: float
    capacity_mw: float
    panel_tilt: float
    panel_azimuth: float
    panel_technology: str
    commissioning_date: datetime
    region: str


class UKSolarDataValidator:
    """
    Data validation class for UK solar farm data.
    
    Implements comprehensive quality control checks specific to UK conditions
    and solar energy systems.
    """
    
    # UK-specific validation bounds
    UK_BOUNDS = {
        'latitude': (49.5, 61.0),      # UK + buffer
        'longitude': (-9.0, 3.0),      # UK + buffer
        'temperature': (-20.0, 40.0),   # UK temperature range (°C)
        'humidity': (0.0, 100.0),       # Relative humidity (%)
        'pressure': (950.0, 1050.0),   # Atmospheric pressure (hPa)
        'wind_speed': (0.0, 50.0),      # Wind speed (m/s)
        'wind_direction': (0.0, 360.0), # Wind direction (degrees)
        'cloud_cover': (0.0, 1.0),     # Cloud cover fraction
        'ghi': (0.0, 1400.0),          # Global horizontal irradiance (W/m²)
        'dni': (0.0, 1200.0),          # Direct normal irradiance (W/m²)
        'dhi': (0.0, 800.0),           # Diffuse horizontal irradiance (W/m²)
        'energy_output': (0.0, None),  # Energy output (kWh, no upper bound)
    }
    
    def __init__(self, config: SolarFarmConfig):
        """
        Initialize validator with solar farm configuration.
        
        Args:
            config: Solar farm configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def validate_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean timestamp data.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with validated timestamps
        """
        if 'timestamp' not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' column")
        
        # Convert to datetime
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove invalid timestamps
        invalid_timestamps = df['timestamp'].isna()
        if invalid_timestamps.any():
            self.logger.warning(f"Removed {invalid_timestamps.sum()} invalid timestamps")
            df = df[~invalid_timestamps]
        
        # Ensure UTC timezone
        if df['timestamp'].dt.tz is None:
            df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
        else:
            df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['timestamp'])
        if duplicates.any():
            self.logger.warning(f"Removed {duplicates.sum()} duplicate timestamps")
            df = df[~duplicates]
        
        return df
    
    def validate_bounds(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data against physical bounds.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with out-of-bounds values flagged
        """
        df = df.copy()
        
        for column, (min_val, max_val) in self.UK_BOUNDS.items():
            if column in df.columns:
                # Flag out-of-bounds values
                if min_val is not None:
                    below_min = df[column] < min_val
                    if below_min.any():
                        self.logger.warning(f"{column}: {below_min.sum()} values below {min_val}")
                        df.loc[below_min, column] = np.nan
                
                if max_val is not None:
                    above_max = df[column] > max_val
                    if above_max.any():
                        self.logger.warning(f"{column}: {above_max.sum()} values above {max_val}")
                        df.loc[above_max, column] = np.nan
        
        return df
    
    def validate_solar_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate consistency between solar radiation components.
        
        Args:
            df: Input dataframe with GHI, DNI, DHI columns
            
        Returns:
            Dataframe with inconsistent values flagged
        """
        df = df.copy()
        
        # Check if required columns exist
        solar_cols = ['ghi', 'dni', 'dhi']
        available_cols = [col for col in solar_cols if col in df.columns]
        
        if len(available_cols) < 2:
            return df
        
        # Add solar geometry if not present
        if 'solar_zenith' not in df.columns:
            df = calculate_uk_solar_features(df, self.config.latitude, self.config.longitude)
        
        # Calculate expected GHI from DNI and DHI
        if all(col in df.columns for col in ['dni', 'dhi', 'solar_zenith']):
            cos_zenith = np.cos(np.radians(df['solar_zenith']))
            expected_ghi = df['dni'] * cos_zenith + df['dhi']
            
            if 'ghi' in df.columns:
                # Check consistency
                ghi_diff = np.abs(df['ghi'] - expected_ghi)
                ghi_relative_error = ghi_diff / (df['ghi'] + 1e-6)
                
                # Flag inconsistent values (>20% relative error)
                inconsistent = (ghi_relative_error > 0.2) & (df['ghi'] > 50)
                if inconsistent.any():
                    self.logger.warning(f"Found {inconsistent.sum()} inconsistent GHI values")
                    df.loc[inconsistent, 'ghi'] = np.nan
        
        # Check for negative values during daylight
        daylight = df['solar_elevation'] > 0
        for col in available_cols:
            if col in df.columns:
                negative_daylight = (df[col] < 0) & daylight
                if negative_daylight.any():
                    self.logger.warning(f"{col}: {negative_daylight.sum()} negative values during daylight")
                    df.loc[negative_daylight, col] = 0.0
        
        return df
    
    def validate_temporal_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate temporal consistency of measurements.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with temporally inconsistent values flagged
        """
        df = df.copy()
        
        # Check for unrealistic changes in measurements
        temporal_checks = {
            'temperature': 10.0,    # Max change per hour (°C)
            'humidity': 30.0,       # Max change per hour (%)
            'pressure': 5.0,        # Max change per hour (hPa)
            'ghi': 500.0,          # Max change per minute (W/m²)
        }
        
        for column, max_change in temporal_checks.items():
            if column in df.columns:
                # Calculate time differences
                time_diff = df['timestamp'].diff().dt.total_seconds() / 3600.0  # hours
                value_diff = df[column].diff().abs()
                
                # Calculate rate of change
                rate_of_change = value_diff / (time_diff + 1e-6)
                
                # Adjust threshold based on time interval
                if column == 'ghi':
                    # For GHI, use per-minute threshold
                    threshold = max_change * (time_diff * 60)
                else:
                    threshold = max_change * time_diff
                
                # Flag unrealistic changes
                unrealistic = (rate_of_change > threshold) & (time_diff < 24)  # Ignore large gaps
                if unrealistic.any():
                    self.logger.warning(f"{column}: {unrealistic.sum()} unrealistic temporal changes")
                    # Don't automatically remove, just flag for review
                    df.loc[unrealistic, f'{column}_temporal_flag'] = True
        
        return df
    
    def validate_nighttime_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate nighttime measurements.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with nighttime validation applied
        """
        df = df.copy()
        
        # Ensure solar geometry is calculated
        if 'solar_elevation' not in df.columns:
            df = calculate_uk_solar_features(df, self.config.latitude, self.config.longitude)
        
        # Define nighttime (sun below horizon)
        nighttime = df['solar_elevation'] <= 0
        
        # Solar radiation should be zero at night
        solar_cols = ['ghi', 'dni', 'dhi']
        for col in solar_cols:
            if col in df.columns:
                nighttime_solar = (df[col] > 5.0) & nighttime  # Allow small sensor noise
                if nighttime_solar.any():
                    self.logger.warning(f"{col}: {nighttime_solar.sum()} non-zero nighttime values")
                    df.loc[nighttime_solar, col] = 0.0
        
        # Energy output should be zero or very low at night
        if 'energy_output' in df.columns:
            nighttime_energy = (df['energy_output'] > 0.01 * self.config.capacity_mw) & nighttime
            if nighttime_energy.any():
                self.logger.warning(f"energy_output: {nighttime_energy.sum()} significant nighttime values")
                # Don't automatically zero - might be legitimate (e.g., auxiliary power)
                df.loc[nighttime_energy, 'energy_output_nighttime_flag'] = True
        
        return df


class UKSolarDataProcessor:
    """
    Main data processing class for UK solar farm data.
    
    Handles data ingestion, validation, preprocessing, and feature engineering
    for physics-based machine learning models.
    """
    
    def __init__(self, config: SolarFarmConfig):
        """
        Initialize data processor.
        
        Args:
            config: Solar farm configuration
        """
        self.config = config
        self.validator = UKSolarDataValidator(config)
        self.logger = logging.getLogger(__name__)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def load_data(self, file_path: Union[str, Path], 
                  file_format: str = 'auto') -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to data file
            file_format: File format ('csv', 'parquet', 'json', 'auto')
            
        Returns:
            Loaded dataframe
        """
        file_path = Path(file_path)
        
        if file_format == 'auto':
            file_format = file_path.suffix.lower().lstrip('.')
        
        self.logger.info(f"Loading data from {file_path} (format: {file_format})")
        
        try:
            if file_format == 'csv':
                df = pd.read_csv(file_path)
            elif file_format == 'parquet':
                df = pd.read_parquet(file_path)
            elif file_format == 'json':
                df = pd.read_json(file_path)
            elif file_format in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply comprehensive data validation.
        
        Args:
            df: Input dataframe
            
        Returns:
            Validated dataframe
        """
        self.logger.info("Starting data validation")
        
        # Validate timestamps
        df = self.validator.validate_timestamps(df)
        
        # Validate bounds
        df = self.validator.validate_bounds(df)
        
        # Validate solar consistency
        df = self.validator.validate_solar_consistency(df)
        
        # Validate temporal consistency
        df = self.validator.validate_temporal_consistency(df)
        
        # Validate nighttime values
        df = self.validator.validate_nighttime_values(df)
        
        self.logger.info(f"Validation complete. {len(df)} rows remaining")
        return df
    
    def handle_missing_data(self, df: pd.DataFrame, 
                           strategy: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing data using various strategies.
        
        Args:
            df: Input dataframe
            strategy: Missing data strategy ('interpolate', 'forward_fill', 'drop')
            
        Returns:
            Dataframe with missing data handled
        """
        df = df.copy()
        
        # Report missing data statistics
        missing_stats = df.isnull().sum()
        missing_pct = (missing_stats / len(df)) * 100
        
        for col in missing_stats[missing_stats > 0].index:
            self.logger.info(f"{col}: {missing_stats[col]} missing ({missing_pct[col]:.1f}%)")
        
        if strategy == 'interpolate':
            # Time-aware interpolation
            df = df.set_index('timestamp')
            
            # Interpolate numerical columns
            numerical_cols = df.select_dtypes(include=[np.number]).columns
            for col in numerical_cols:
                if df[col].isnull().any():
                    # Use time-based interpolation with limits
                    df[col] = df[col].interpolate(
                        method='time', 
                        limit=12,  # Max 12 consecutive missing values
                        limit_direction='both'
                    )
            
            df = df.reset_index()
            
        elif strategy == 'forward_fill':
            # Forward fill with limits
            df = df.fillna(method='ffill', limit=6)
            
        elif strategy == 'drop':
            # Drop rows with any missing values
            df = df.dropna()
            
        else:
            raise ValueError(f"Unknown missing data strategy: {strategy}")
        
        self.logger.info(f"Missing data handling complete. {len(df)} rows remaining")
        return df
    
    def calculate_clear_sky_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate clear sky model features.
        
        Args:
            df: Input dataframe with solar geometry
            
        Returns:
            Dataframe with clear sky features
        """
        from physics.uk_atmosphere import UKAtmosphericModel
        
        df = df.copy()
        atm_model = UKAtmosphericModel(
            self.config.latitude, 
            self.config.longitude, 
            self.config.elevation
        )
        
        clear_sky_results = []
        
        for idx, row in df.iterrows():
            dt = pd.to_datetime(row['timestamp'])
            zenith = row.get('solar_zenith', 90.0)
            air_mass = row.get('air_mass', 40.0)
            extra_irr = row.get('extraterrestrial_irradiance', 0.0)
            
            # Calculate clear sky irradiance
            ghi_clear, dni_clear, dhi_clear = atm_model.ineichen_perez_clear_sky(
                dt, zenith, air_mass, extra_irr
            )
            
            clear_sky_results.append({
                'ghi_clear': ghi_clear,
                'dni_clear': dni_clear,
                'dhi_clear': dhi_clear
            })
        
        # Add clear sky features
        for key in clear_sky_results[0].keys():
            df[key] = [r[key] for r in clear_sky_results]
        
        # Calculate clear sky indices
        if 'ghi' in df.columns:
            df['clear_sky_index'] = df['ghi'] / (df['ghi_clear'] + 1e-6)
            df['clear_sky_index'] = df['clear_sky_index'].clip(0, 1.5)
        
        if 'dni' in df.columns:
            df['dni_clear_sky_index'] = df['dni'] / (df['dni_clear'] + 1e-6)
            df['dni_clear_sky_index'] = df['dni_clear_sky_index'].clip(0, 1.5)
        
        return df
    
    def calculate_pv_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate photovoltaic system specific features.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with PV features
        """
        df = df.copy()
        
        # Panel temperature model (simplified)
        if 'temperature' in df.columns and 'wind_speed' in df.columns and 'ghi' in df.columns:
            # NOCT-based model
            noct = 45.0  # Nominal Operating Cell Temperature (°C)
            df['panel_temperature'] = (
                df['temperature'] + 
                (noct - 20) * df['ghi'] / 800.0 * 
                (1 - 0.05 * df['wind_speed'])
            )
        
        # Incidence angle for tilted panels
        if all(col in df.columns for col in ['solar_zenith', 'solar_azimuth']):
            panel_tilt_rad = np.radians(self.config.panel_tilt)
            panel_azimuth_rad = np.radians(self.config.panel_azimuth)
            solar_zenith_rad = np.radians(df['solar_zenith'])
            solar_azimuth_rad = np.radians(df['solar_azimuth'])
            
            # Incidence angle calculation
            cos_incidence = (
                np.sin(solar_zenith_rad) * np.sin(panel_tilt_rad) * 
                np.cos(solar_azimuth_rad - panel_azimuth_rad) +
                np.cos(solar_zenith_rad) * np.cos(panel_tilt_rad)
            )
            
            df['incidence_angle'] = np.degrees(np.arccos(np.clip(cos_incidence, -1, 1)))
            
            # Incidence angle modifier (simple model)
            df['iam'] = 1 - 0.05 * ((df['incidence_angle'] / 90.0) ** 2)
            df['iam'] = df['iam'].clip(0, 1)
        
        # Plane of array irradiance (simplified)
        if 'ghi' in df.columns and 'dhi' in df.columns:
            # Simplified POA calculation
            tilt_factor = np.cos(np.radians(self.config.panel_tilt))
            df['poa_irradiance'] = df['ghi'] * tilt_factor + df['dhi'] * (1 + tilt_factor) / 2
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           columns: List[str], 
                           lags: List[int]) -> pd.DataFrame:
        """
        Create lagged features for time series modeling.
        
        Args:
            df: Input dataframe
            columns: Columns to create lags for
            lags: List of lag periods (in time steps)
            
        Returns:
            Dataframe with lag features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for lag in lags:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame,
                               columns: List[str],
                               windows: List[int]) -> pd.DataFrame:
        """
        Create rolling window features.
        
        Args:
            df: Input dataframe
            columns: Columns to create rolling features for
            windows: List of window sizes
            
        Returns:
            Dataframe with rolling features
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_rolling_std_{window}'] = df[col].rolling(window).std()
                    df[f'{col}_rolling_min_{window}'] = df[col].rolling(window).min()
                    df[f'{col}_rolling_max_{window}'] = df[col].rolling(window).max()
        
        return df
    
    def process_full_pipeline(self, df: pd.DataFrame,
                             include_lags: bool = True,
                             include_rolling: bool = True) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Args:
            df: Input dataframe
            include_lags: Whether to include lag features
            include_rolling: Whether to include rolling features
            
        Returns:
            Fully processed dataframe
        """
        self.logger.info("Starting full data processing pipeline")
        
        # Step 1: Validate data
        df = self.validate_data(df)
        
        # Step 2: Handle missing data
        df = self.handle_missing_data(df)
        
        # Step 3: Calculate solar geometry features
        self.logger.info("Calculating solar geometry features")
        df = calculate_uk_solar_features(
            df, self.config.latitude, self.config.longitude, self.config.elevation
        )
        
        # Step 4: Calculate atmospheric features
        self.logger.info("Calculating atmospheric features")
        df = calculate_uk_atmospheric_features(
            df, self.config.latitude, self.config.longitude, self.config.elevation
        )
        
        # Step 5: Calculate clear sky features
        self.logger.info("Calculating clear sky features")
        df = self.calculate_clear_sky_features(df)
        
        # Step 6: Calculate PV-specific features
        self.logger.info("Calculating PV features")
        df = self.calculate_pv_features(df)
        
        # Step 7: Create lag features
        if include_lags:
            self.logger.info("Creating lag features")
            lag_columns = ['ghi', 'temperature', 'humidity', 'cloud_cover']
            lags = [1, 2, 3, 6, 12]  # Various lag periods
            df = self.create_lag_features(df, lag_columns, lags)
        
        # Step 8: Create rolling features
        if include_rolling:
            self.logger.info("Creating rolling features")
            rolling_columns = ['ghi', 'clear_sky_index', 'temperature']
            windows = [6, 12, 24]  # Various window sizes
            df = self.create_rolling_features(df, rolling_columns, windows)
        
        # Step 9: Final cleanup
        df = df.dropna()  # Remove rows with NaN from feature engineering
        
        self.logger.info(f"Pipeline complete. Final dataset: {len(df)} rows, {len(df.columns)} columns")
        
        return df


def load_solar_farm_config(config_path: Union[str, Path]) -> SolarFarmConfig:
    """
    Load solar farm configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Solar farm configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert commissioning_date string to datetime
    if isinstance(config_dict['commissioning_date'], str):
        config_dict['commissioning_date'] = datetime.fromisoformat(
            config_dict['commissioning_date']
        )
    
    return SolarFarmConfig(**config_dict)


def create_sample_config() -> Dict[str, Any]:
    """
    Create a sample configuration for a UK solar farm.
    
    Returns:
        Sample configuration dictionary
    """
    return {
        'name': 'UK Solar Farm Example',
        'latitude': 51.5074,
        'longitude': -0.1278,
        'elevation': 50.0,
        'capacity_mw': 50.0,
        'panel_tilt': 35.0,
        'panel_azimuth': 180.0,
        'panel_technology': 'c-Si',
        'commissioning_date': '2020-01-01T00:00:00',
        'region': 'southern_england'
    }


if __name__ == "__main__":
    # Example usage
    import tempfile
    
    # Create sample configuration
    config_dict = create_sample_config()
    config = SolarFarmConfig(**config_dict)
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=100, freq='H', tz='UTC')
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
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
    
    # Process data
    processor = UKSolarDataProcessor(config)
    processed_data = processor.process_full_pipeline(sample_data)
    
    print(f"Processed {len(processed_data)} rows with {len(processed_data.columns)} features")
    print("\nSample features:")
    print(processed_data.columns.tolist()[:20])  # Show first 20 features

