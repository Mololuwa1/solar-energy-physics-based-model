"""
Advanced Feature Engineering for UK Solar Energy Prediction

This module implements sophisticated feature engineering techniques specifically
designed for physics-based machine learning models for UK solar farms.

Author: Manus AI
Date: 2025-08-16
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
import warnings


class UKSolarFeatureEngineer:
    """
    Advanced feature engineering class for UK solar energy prediction.
    
    Implements physics-informed feature creation, temporal pattern extraction,
    and UK-specific meteorological feature engineering.
    """
    
    def __init__(self, latitude: float, longitude: float):
        """
        Initialize feature engineer.
        
        Args:
            latitude: Site latitude
            longitude: Site longitude
        """
        self.latitude = latitude
        self.longitude = longitude
        self.logger = logging.getLogger(__name__)
        
        # Feature scaling objects
        self.scalers = {}
        self.feature_selectors = {}
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal features for UK solar prediction.
        
        Args:
            df: Input dataframe with timestamp column
            
        Returns:
            Dataframe with temporal features
        """
        df = df.copy()
        
        # Ensure timestamp is datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic time components
        df['year'] = df['timestamp'].dt.year
        df['month'] = df['timestamp'].dt.month
        df['day'] = df['timestamp'].dt.day
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        df['week_of_year'] = df['timestamp'].dt.isocalendar().week
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # UK-specific seasonal features
        # Meteorological seasons for UK
        df['season'] = df['month'].map({
            12: 0, 1: 0, 2: 0,  # Winter
            3: 1, 4: 1, 5: 1,   # Spring
            6: 2, 7: 2, 8: 2,   # Summer
            9: 3, 10: 3, 11: 3  # Autumn
        })
        
        # Cyclical encoding for temporal features
        # Hour of day (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of year (365.25-day cycle)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of week (7-day cycle)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Week of year (52-week cycle)
        df['week_of_year_sin'] = np.sin(2 * np.pi * df['week_of_year'] / 52)
        df['week_of_year_cos'] = np.cos(2 * np.pi * df['week_of_year'] / 52)
        
        # UK-specific time features
        # British Summer Time (BST) indicator
        df['is_bst'] = df['timestamp'].dt.tz_convert('Europe/London').dt.dst.astype(int)
        
        # Working hours indicator (UK business hours)
        df['is_working_hours'] = (
            (df['hour'] >= 9) & (df['hour'] <= 17) & 
            (df['day_of_week'] < 5)  # Monday-Friday
        ).astype(int)
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # UK holiday seasons (approximate)
        df['is_summer_holiday'] = (
            (df['month'] == 7) | (df['month'] == 8)
        ).astype(int)
        
        df['is_christmas_period'] = (
            (df['month'] == 12) & (df['day'] >= 20)
        ).astype(int)
        
        return df
    
    def create_weather_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather interaction features specific to UK conditions.
        
        Args:
            df: Input dataframe with weather variables
            
        Returns:
            Dataframe with weather interaction features
        """
        df = df.copy()
        
        # Temperature-humidity interactions
        if 'temperature' in df.columns and 'humidity' in df.columns:
            # Heat index (feels-like temperature)
            df['heat_index'] = self._calculate_heat_index(df['temperature'], df['humidity'])
            
            # Dew point temperature
            df['dew_point'] = self._calculate_dew_point(df['temperature'], df['humidity'])
            
            # Vapor pressure deficit
            df['vpd'] = self._calculate_vpd(df['temperature'], df['humidity'])
        
        # Wind chill (important for UK winter conditions)
        if 'temperature' in df.columns and 'wind_speed' in df.columns:
            df['wind_chill'] = self._calculate_wind_chill(df['temperature'], df['wind_speed'])
        
        # Atmospheric stability indicators
        if all(col in df.columns for col in ['temperature', 'pressure', 'humidity']):
            # Potential temperature (simplified)
            df['potential_temperature'] = df['temperature'] * (1000 / df['pressure']) ** 0.286
            
            # Equivalent potential temperature
            df['equiv_potential_temp'] = self._calculate_equivalent_potential_temperature(
                df['temperature'], df['pressure'], df['humidity']
            )
        
        # Cloud-radiation interactions
        if 'cloud_cover' in df.columns:
            # Cloud transmission factor (physics-based)
            df['cloud_transmission'] = 1 - 0.75 * df['cloud_cover']
            
            # Cloud enhancement factor (for diffuse radiation)
            df['cloud_enhancement'] = 1 + 0.3 * df['cloud_cover'] * (1 - df['cloud_cover'])
        
        # Pressure tendency (if multiple time points available)
        if 'pressure' in df.columns and len(df) > 1:
            df['pressure_tendency'] = df['pressure'].diff()
            df['pressure_tendency_3h'] = df['pressure'].diff(3)  # 3-hour tendency
        
        # Wind direction categories (UK-specific)
        if 'wind_direction' in df.columns:
            df['wind_from_atlantic'] = (
                (df['wind_direction'] >= 225) & (df['wind_direction'] <= 315)
            ).astype(int)
            
            df['wind_from_continent'] = (
                (df['wind_direction'] >= 45) & (df['wind_direction'] <= 135)
            ).astype(int)
            
            df['wind_from_arctic'] = (
                (df['wind_direction'] >= 315) | (df['wind_direction'] <= 45)
            ).astype(int)
        
        return df
    
    def create_solar_physics_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create physics-based solar features.
        
        Args:
            df: Input dataframe with solar and atmospheric data
            
        Returns:
            Dataframe with physics-based solar features
        """
        df = df.copy()
        
        # Atmospheric transparency features
        if 'ghi' in df.columns and 'ghi_clear' in df.columns:
            # Clear sky index (atmospheric transparency factor)
            df['clear_sky_index'] = df['ghi'] / (df['ghi_clear'] + 1e-6)
            df['clear_sky_index'] = df['clear_sky_index'].clip(0, 1.5)
            
            # Clear sky index variability
            df['csi_variability'] = df['clear_sky_index'].rolling(6).std()
            
            # Clear sky index persistence
            df['csi_persistence'] = df['clear_sky_index'].shift(1)
        
        # Diffuse fraction
        if 'ghi' in df.columns and 'dhi' in df.columns:
            df['diffuse_fraction'] = df['dhi'] / (df['ghi'] + 1e-6)
            df['diffuse_fraction'] = df['diffuse_fraction'].clip(0, 1)
        
        # Direct normal irradiance fraction
        if 'dni' in df.columns and 'ghi' in df.columns and 'solar_zenith' in df.columns:
            cos_zenith = np.cos(np.radians(df['solar_zenith']))
            df['dni_fraction'] = (df['dni'] * cos_zenith) / (df['ghi'] + 1e-6)
            df['dni_fraction'] = df['dni_fraction'].clip(0, 1)
        
        # Clearness index (Kt)
        if 'ghi' in df.columns and 'extraterrestrial_irradiance' in df.columns:
            cos_zenith = np.cos(np.radians(df['solar_zenith']))
            extraterrestrial_horizontal = df['extraterrestrial_irradiance'] * cos_zenith
            df['clearness_index'] = df['ghi'] / (extraterrestrial_horizontal + 1e-6)
            df['clearness_index'] = df['clearness_index'].clip(0, 1.2)
        
        # Solar elevation categories
        if 'solar_elevation' in df.columns:
            df['low_sun'] = (df['solar_elevation'] < 15).astype(int)
            df['medium_sun'] = (
                (df['solar_elevation'] >= 15) & (df['solar_elevation'] < 45)
            ).astype(int)
            df['high_sun'] = (df['solar_elevation'] >= 45).astype(int)
        
        # Air mass categories
        if 'air_mass' in df.columns:
            df['low_air_mass'] = (df['air_mass'] < 2).astype(int)
            df['medium_air_mass'] = (
                (df['air_mass'] >= 2) & (df['air_mass'] < 5)
            ).astype(int)
            df['high_air_mass'] = (df['air_mass'] >= 5).astype(int)
        
        return df
    
    def create_lag_and_lead_features(self, df: pd.DataFrame, 
                                   target_columns: List[str],
                                   lag_periods: List[int],
                                   lead_periods: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Create lag and lead features for time series modeling.
        
        Args:
            df: Input dataframe
            target_columns: Columns to create lags/leads for
            lag_periods: List of lag periods
            lead_periods: List of lead periods (optional)
            
        Returns:
            Dataframe with lag and lead features
        """
        df = df.copy()
        
        for col in target_columns:
            if col in df.columns:
                # Lag features
                for lag in lag_periods:
                    df[f'{col}_lag_{lag}'] = df[col].shift(lag)
                
                # Lead features (for training with future information)
                if lead_periods:
                    for lead in lead_periods:
                        df[f'{col}_lead_{lead}'] = df[col].shift(-lead)
        
        return df
    
    def create_rolling_statistics(self, df: pd.DataFrame,
                                columns: List[str],
                                windows: List[int],
                                statistics: List[str] = ['mean', 'std', 'min', 'max']) -> pd.DataFrame:
        """
        Create rolling window statistics.
        
        Args:
            df: Input dataframe
            columns: Columns to calculate statistics for
            windows: Window sizes
            statistics: Statistics to calculate
            
        Returns:
            Dataframe with rolling statistics
        """
        df = df.copy()
        
        for col in columns:
            if col in df.columns:
                for window in windows:
                    rolling = df[col].rolling(window, min_periods=1)
                    
                    if 'mean' in statistics:
                        df[f'{col}_rolling_mean_{window}'] = rolling.mean()
                    if 'std' in statistics:
                        df[f'{col}_rolling_std_{window}'] = rolling.std()
                    if 'min' in statistics:
                        df[f'{col}_rolling_min_{window}'] = rolling.min()
                    if 'max' in statistics:
                        df[f'{col}_rolling_max_{window}'] = rolling.max()
                    if 'median' in statistics:
                        df[f'{col}_rolling_median_{window}'] = rolling.median()
                    if 'skew' in statistics:
                        df[f'{col}_rolling_skew_{window}'] = rolling.skew()
                    if 'kurt' in statistics:
                        df[f'{col}_rolling_kurt_{window}'] = rolling.kurt()
        
        return df
    
    def create_fourier_features(self, df: pd.DataFrame, 
                              columns: List[str],
                              periods: List[float]) -> pd.DataFrame:
        """
        Create Fourier transform features for periodic patterns.
        
        Args:
            df: Input dataframe
            columns: Columns to create Fourier features for
            periods: List of periods (in hours)
            
        Returns:
            Dataframe with Fourier features
        """
        df = df.copy()
        
        # Create time index in hours from start
        if 'timestamp' in df.columns:
            start_time = df['timestamp'].iloc[0]
            df['hours_from_start'] = (df['timestamp'] - start_time).dt.total_seconds() / 3600
        else:
            df['hours_from_start'] = np.arange(len(df))
        
        for col in columns:
            if col in df.columns:
                for period in periods:
                    # Create sine and cosine components
                    df[f'{col}_fourier_sin_{period}h'] = np.sin(
                        2 * np.pi * df['hours_from_start'] / period
                    )
                    df[f'{col}_fourier_cos_{period}h'] = np.cos(
                        2 * np.pi * df['hours_from_start'] / period
                    )
        
        # Remove temporary column
        df = df.drop('hours_from_start', axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                  feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between variable pairs.
        
        Args:
            df: Input dataframe
            feature_pairs: List of (feature1, feature2) tuples
            
        Returns:
            Dataframe with interaction features
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio interaction (with small epsilon to avoid division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
                
                # Difference interaction
                df[f'{feat1}_minus_{feat2}'] = df[feat1] - df[feat2]
                
                # Sum interaction
                df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
        
        return df
    
    def create_uk_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to UK solar conditions and patterns.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dataframe with UK-specific features
        """
        df = df.copy()
        
        # UK seasonal solar patterns
        if 'month' in df.columns:
            # UK solar season strength
            uk_solar_strength = {
                1: 0.1, 2: 0.2, 3: 0.4, 4: 0.6, 5: 0.8, 6: 1.0,
                7: 1.0, 8: 0.9, 9: 0.7, 10: 0.5, 11: 0.3, 12: 0.1
            }
            df['uk_solar_season_strength'] = df['month'].map(uk_solar_strength)
        
        # UK weather pattern indicators
        if 'pressure' in df.columns:
            # High pressure system (typically clearer skies)
            df['high_pressure'] = (df['pressure'] > 1020).astype(int)
            
            # Low pressure system (typically cloudier)
            df['low_pressure'] = (df['pressure'] < 1000).astype(int)
        
        # UK cloud patterns
        if 'cloud_cover' in df.columns and 'month' in df.columns:
            # Expected cloud cover for UK by month
            uk_expected_cloud = {
                1: 0.75, 2: 0.70, 3: 0.68, 4: 0.65, 5: 0.62, 6: 0.60,
                7: 0.58, 8: 0.60, 9: 0.65, 10: 0.70, 11: 0.73, 12: 0.76
            }
            df['uk_expected_cloud'] = df['month'].map(uk_expected_cloud)
            df['cloud_anomaly'] = df['cloud_cover'] - df['uk_expected_cloud']
        
        # UK daylight patterns
        if 'day_of_year' in df.columns:
            # Approximate UK daylight hours by day of year
            df['uk_daylight_hours'] = 12 + 4 * np.sin(
                2 * np.pi * (df['day_of_year'] - 81) / 365.25
            )
        
        # UK regional weather patterns (simplified)
        # This would be more sophisticated with actual location data
        if self.latitude > 55:  # Scotland
            df['is_scotland'] = 1
            df['regional_cloud_factor'] = 1.1  # More cloudy
        elif self.latitude > 53:  # Northern England
            df['is_northern_england'] = 1
            df['regional_cloud_factor'] = 1.05
        else:  # Southern regions
            df['is_southern_uk'] = 1
            df['regional_cloud_factor'] = 1.0
        
        return df
    
    def select_features(self, df: pd.DataFrame, 
                       target_column: str,
                       method: str = 'correlation',
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select most relevant features for prediction.
        
        Args:
            df: Input dataframe
            target_column: Target variable column name
            method: Feature selection method ('correlation', 'f_test', 'mutual_info')
            k: Number of features to select
            
        Returns:
            Tuple of (selected dataframe, selected feature names)
        """
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataframe")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column and col != 'timestamp']
        X = df[feature_columns].select_dtypes(include=[np.number])
        y = df[target_column]
        
        # Remove rows with NaN values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if method == 'correlation':
            # Correlation-based selection
            correlations = X_clean.corrwith(y_clean).abs().sort_values(ascending=False)
            selected_features = correlations.head(k).index.tolist()
            
        elif method == 'f_test':
            # F-test based selection
            selector = SelectKBest(score_func=f_regression, k=min(k, X_clean.shape[1]))
            selector.fit(X_clean, y_clean)
            selected_features = X_clean.columns[selector.get_support()].tolist()
            
        else:
            raise ValueError(f"Unknown feature selection method: {method}")
        
        # Create selected dataframe
        selected_df = df[['timestamp', target_column] + selected_features].copy()
        
        self.logger.info(f"Selected {len(selected_features)} features using {method}")
        
        return selected_df, selected_features
    
    def scale_features(self, df: pd.DataFrame, 
                      method: str = 'standard',
                      exclude_columns: List[str] = None) -> pd.DataFrame:
        """
        Scale numerical features.
        
        Args:
            df: Input dataframe
            method: Scaling method ('standard', 'minmax', 'robust')
            exclude_columns: Columns to exclude from scaling
            
        Returns:
            Dataframe with scaled features
        """
        if exclude_columns is None:
            exclude_columns = ['timestamp']
        
        df = df.copy()
        
        # Select numerical columns for scaling
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        columns_to_scale = [col for col in numerical_columns if col not in exclude_columns]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Fit and transform
        df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])
        
        # Store scaler for later use
        self.scalers[method] = scaler
        
        return df
    
    # Helper methods for weather calculations
    def _calculate_heat_index(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate heat index from temperature and humidity."""
        temp_f = temp_c * 9/5 + 32  # Convert to Fahrenheit
        
        hi = 0.5 * (temp_f + 61.0 + ((temp_f - 68.0) * 1.2) + (humidity * 0.094))
        
        # Use more complex formula for higher temperatures
        mask = hi >= 80
        if mask.any():
            hi_complex = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity -
                         0.22475541 * temp_f * humidity - 0.00683783 * temp_f**2 -
                         0.05481717 * humidity**2 + 0.00122874 * temp_f**2 * humidity +
                         0.00085282 * temp_f * humidity**2 - 0.00000199 * temp_f**2 * humidity**2)
            hi[mask] = hi_complex[mask]
        
        return (hi - 32) * 5/9  # Convert back to Celsius
    
    def _calculate_dew_point(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate dew point temperature."""
        a = 17.27
        b = 237.7
        
        alpha = ((a * temp_c) / (b + temp_c)) + np.log(humidity / 100.0)
        dew_point = (b * alpha) / (a - alpha)
        
        return dew_point
    
    def _calculate_vpd(self, temp_c: pd.Series, humidity: pd.Series) -> pd.Series:
        """Calculate vapor pressure deficit."""
        # Saturation vapor pressure (kPa)
        es = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
        
        # Actual vapor pressure
        ea = es * humidity / 100.0
        
        # Vapor pressure deficit
        vpd = es - ea
        
        return vpd
    
    def _calculate_wind_chill(self, temp_c: pd.Series, wind_speed: pd.Series) -> pd.Series:
        """Calculate wind chill temperature."""
        # Convert to mph and Fahrenheit for standard formula
        wind_mph = wind_speed * 2.237
        temp_f = temp_c * 9/5 + 32
        
        # Wind chill formula (valid for temp <= 50°F and wind >= 3 mph)
        mask = (temp_f <= 50) & (wind_mph >= 3)
        
        wind_chill = temp_c.copy()  # Default to actual temperature
        
        if mask.any():
            wc_f = (35.74 + 0.6215 * temp_f - 35.75 * wind_mph**0.16 + 
                   0.4275 * temp_f * wind_mph**0.16)
            wind_chill[mask] = (wc_f[mask] - 32) * 5/9  # Convert back to Celsius
        
        return wind_chill
    
    def _calculate_equivalent_potential_temperature(self, temp_c: pd.Series, 
                                                  pressure: pd.Series, 
                                                  humidity: pd.Series) -> pd.Series:
        """Calculate equivalent potential temperature."""
        # Convert to Kelvin
        temp_k = temp_c + 273.15
        
        # Mixing ratio (simplified)
        es = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))  # hPa
        e = es * humidity / 100.0
        mixing_ratio = 0.622 * e / (pressure - e)
        
        # Potential temperature
        theta = temp_k * (1000 / pressure) ** 0.286
        
        # Equivalent potential temperature (simplified)
        theta_e = theta * np.exp((2.5e6 * mixing_ratio) / (1004 * temp_k))
        
        return theta_e - 273.15  # Convert back to Celsius


def create_comprehensive_features(df: pd.DataFrame, 
                                latitude: float, 
                                longitude: float,
                                target_column: str = 'energy_output') -> pd.DataFrame:
    """
    Create comprehensive feature set for UK solar prediction.
    
    Args:
        df: Input dataframe
        latitude: Site latitude
        longitude: Site longitude
        target_column: Target variable column name
        
    Returns:
        Dataframe with comprehensive features
    """
    engineer = UKSolarFeatureEngineer(latitude, longitude)
    
    # Create all feature types
    df = engineer.create_temporal_features(df)
    df = engineer.create_weather_interaction_features(df)
    df = engineer.create_solar_physics_features(df)
    df = engineer.create_uk_specific_features(df)
    
    # Create lag features for key variables
    lag_columns = ['ghi', 'clear_sky_index', 'temperature', 'cloud_cover']
    lag_periods = [1, 2, 3, 6, 12, 24]
    df = engineer.create_lag_and_lead_features(df, lag_columns, lag_periods)
    
    # Create rolling statistics
    rolling_columns = ['ghi', 'clear_sky_index', 'temperature']
    windows = [3, 6, 12, 24]
    df = engineer.create_rolling_statistics(df, rolling_columns, windows)
    
    # Create Fourier features for periodic patterns
    fourier_columns = ['ghi', 'temperature']
    periods = [24, 168, 8760]  # Daily, weekly, yearly patterns
    df = engineer.create_fourier_features(df, fourier_columns, periods)
    
    # Create interaction features
    important_pairs = [
        ('temperature', 'humidity'),
        ('ghi', 'cloud_cover'),
        ('clear_sky_index', 'solar_elevation'),
        ('wind_speed', 'temperature')
    ]
    df = engineer.create_interaction_features(df, important_pairs)
    
    return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    timestamps = pd.date_range('2024-01-01', periods=1000, freq='H', tz='UTC')
    sample_data = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': np.random.normal(10, 5, 1000),
        'humidity': np.random.normal(75, 15, 1000),
        'pressure': np.random.normal(1013, 10, 1000),
        'wind_speed': np.random.uniform(0, 15, 1000),
        'wind_direction': np.random.uniform(0, 360, 1000),
        'cloud_cover': np.random.uniform(0, 1, 1000),
        'ghi': np.random.uniform(0, 800, 1000),
        'ghi_clear': np.random.uniform(0, 1000, 1000),
        'solar_elevation': np.random.uniform(-20, 60, 1000),
        'energy_output': np.random.uniform(0, 40, 1000)
    })
    
    # Create comprehensive features
    result_df = create_comprehensive_features(sample_data, 51.5, -0.1)
    
    print(f"Created {len(result_df.columns)} features from {len(sample_data.columns)} original columns")
    print(f"Feature categories created:")
    
    feature_types = {
        'temporal': [col for col in result_df.columns if any(x in col for x in ['hour', 'day', 'month', 'season'])],
        'weather': [col for col in result_df.columns if any(x in col for x in ['temp', 'humid', 'wind', 'pressure'])],
        'solar': [col for col in result_df.columns if any(x in col for x in ['ghi', 'solar', 'clear'])],
        'lag': [col for col in result_df.columns if 'lag' in col],
        'rolling': [col for col in result_df.columns if 'rolling' in col],
        'interaction': [col for col in result_df.columns if any(x in col for x in ['_x_', '_div_', '_minus_', '_plus_'])]
    }
    
    for feat_type, features in feature_types.items():
        print(f"  {feat_type}: {len(features)} features")

