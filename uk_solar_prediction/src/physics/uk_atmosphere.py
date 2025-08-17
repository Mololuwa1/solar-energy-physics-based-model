"""
UK Atmospheric Modeling for Solar Energy Prediction

This module implements atmospheric models specifically calibrated for UK conditions,
including clear sky models, atmospheric transparency calculations, and weather pattern
analysis optimized for British Isles meteorology.

Author: Manus AI
Date: 2025-08-16
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import math


class UKAtmosphericModel:
    """
    Atmospheric model calibrated for UK conditions.
    
    Incorporates UK-specific atmospheric parameters, seasonal variations,
    and typical weather patterns for accurate solar radiation modeling.
    """
    
    # UK atmospheric constants
    UK_TURBIDITY_COEFFICIENTS = {
        'winter': {'clean': 2.5, 'average': 3.2, 'polluted': 4.1},
        'spring': {'clean': 2.8, 'average': 3.6, 'polluted': 4.8},
        'summer': {'clean': 3.1, 'average': 4.2, 'polluted': 5.5},
        'autumn': {'clean': 2.7, 'average': 3.4, 'polluted': 4.3}
    }
    
    # UK precipitation patterns (mm/month averages)
    UK_PRECIPITATION_PATTERNS = {
        1: 84, 2: 60, 3: 67, 4: 57, 5: 56, 6: 62,
        7: 56, 8: 67, 9: 73, 10: 84, 11: 84, 12: 84
    }
    
    # UK cloud cover statistics (fraction, 0-1)
    UK_CLOUD_COVER_MONTHLY = {
        1: 0.75, 2: 0.70, 3: 0.68, 4: 0.65, 5: 0.62, 6: 0.60,
        7: 0.58, 8: 0.60, 9: 0.65, 10: 0.70, 11: 0.73, 12: 0.76
    }
    
    # UK atmospheric pressure variations (hPa)
    UK_PRESSURE_SEASONAL = {
        'winter': {'mean': 1013.2, 'std': 15.8},
        'spring': {'mean': 1015.1, 'std': 12.4},
        'summer': {'mean': 1016.8, 'std': 8.9},
        'autumn': {'mean': 1014.5, 'std': 13.7}
    }
    
    def __init__(self, latitude: float, longitude: float, elevation: float = 0.0):
        """
        Initialize UK atmospheric model.
        
        Args:
            latitude: Site latitude (50-60°N for UK)
            longitude: Site longitude (-8 to 2° for UK)
            elevation: Site elevation in meters
        """
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        
        # Determine UK region for localized parameters
        self.region = self._determine_uk_region(latitude, longitude)
        
    def _determine_uk_region(self, lat: float, lon: float) -> str:
        """Determine UK region for localized atmospheric parameters."""
        if lat >= 56.0:
            return 'scotland'
        elif lat >= 53.0:
            return 'northern_england'
        elif lat >= 51.5:
            return 'midlands'
        elif lon <= -3.0:
            return 'wales'
        else:
            return 'southern_england'
    
    def _get_season(self, month: int) -> str:
        """Get meteorological season for UK."""
        if month in [12, 1, 2]:
            return 'winter'
        elif month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        else:
            return 'autumn'
    
    def linke_turbidity(self, dt: datetime, atmospheric_condition: str = 'average') -> float:
        """
        Calculate Linke turbidity factor for UK conditions.
        
        Args:
            dt: Datetime object
            atmospheric_condition: 'clean', 'average', or 'polluted'
            
        Returns:
            Linke turbidity factor
        """
        season = self._get_season(dt.month)
        base_turbidity = self.UK_TURBIDITY_COEFFICIENTS[season][atmospheric_condition]
        
        # Regional adjustments
        regional_factors = {
            'scotland': 0.9,        # Cleaner air
            'northern_england': 1.1, # Industrial areas
            'midlands': 1.2,        # Urban/industrial
            'wales': 0.95,          # Rural, mountainous
            'southern_england': 1.05 # Mixed urban/rural
        }
        
        turbidity = base_turbidity * regional_factors.get(self.region, 1.0)
        
        # Elevation correction (cleaner air at higher elevations)
        elevation_factor = 1.0 - (self.elevation / 10000.0) * 0.1
        turbidity *= elevation_factor
        
        return max(1.5, min(turbidity, 8.0))  # Reasonable bounds
    
    def ineichen_perez_clear_sky(self, dt: datetime, zenith_angle: float, 
                                air_mass: float, extraterrestrial_irradiance: float,
                                atmospheric_condition: str = 'average') -> Tuple[float, float, float]:
        """
        Calculate clear sky irradiance using Ineichen-Perez model adapted for UK.
        
        Args:
            dt: Datetime object
            zenith_angle: Solar zenith angle in degrees
            air_mass: Air mass value
            extraterrestrial_irradiance: Extraterrestrial irradiance in W/m²
            atmospheric_condition: Atmospheric clarity condition
            
        Returns:
            Tuple of (GHI, DNI, DHI) clear sky irradiances in W/m²
        """
        if zenith_angle >= 90:
            return 0.0, 0.0, 0.0
        
        # Get Linke turbidity for UK conditions
        tl = self.linke_turbidity(dt, atmospheric_condition)
        
        # Ineichen-Perez coefficients
        cos_zenith = math.cos(math.radians(zenith_angle))
        
        # Beam normal irradiance
        delta_r = 1.0 / (6.6296 + 1.7513 * air_mass - 0.1202 * air_mass**2 + 
                        0.0065 * air_mass**3 - 0.00013 * air_mass**4)
        
        # UK-specific atmospheric transmission
        tau_b = 0.664 + 0.163 / delta_r - 0.006 / delta_r**2 - 0.0002 / delta_r**3
        tau_b = tau_b * math.exp(-0.09 * air_mass * tl)
        
        # Direct normal irradiance
        dni_clear = extraterrestrial_irradiance * tau_b
        
        # Diffuse horizontal irradiance (UK has high diffuse component)
        tau_d = tau_b * (0.271 - 0.294 * tau_b) * math.sin(math.radians(88.0 - zenith_angle))**2.3
        tau_d = max(0.0, min(tau_d, 0.8))  # UK-specific bounds
        
        dhi_clear = extraterrestrial_irradiance * cos_zenith * tau_d
        
        # Global horizontal irradiance
        ghi_clear = dni_clear * cos_zenith + dhi_clear
        
        # UK atmospheric corrections for high latitude
        if self.latitude > 55.0:  # Scotland corrections
            ghi_clear *= 0.98  # Slightly reduced due to atmospheric path
            dhi_clear *= 1.02  # Increased diffuse due to scattering
        
        return max(0.0, ghi_clear), max(0.0, dni_clear), max(0.0, dhi_clear)
    
    def atmospheric_transparency_factor(self, measured_ghi: float, clear_sky_ghi: float) -> float:
        """
        Calculate atmospheric transparency factor (ATF).
        
        Args:
            measured_ghi: Measured global horizontal irradiance
            clear_sky_ghi: Clear sky global horizontal irradiance
            
        Returns:
            Atmospheric transparency factor (0-1.2 typical range)
        """
        if clear_sky_ghi <= 0:
            return 0.0
        
        atf = measured_ghi / clear_sky_ghi
        
        # UK-specific bounds (can exceed 1.0 due to cloud enhancement)
        return max(0.0, min(atf, 1.3))
    
    def uk_cloud_model(self, cloud_cover: float, dt: datetime) -> Dict[str, float]:
        """
        Model cloud effects on solar radiation for UK conditions.
        
        Args:
            cloud_cover: Cloud cover fraction (0-1)
            dt: Datetime object
            
        Returns:
            Dictionary with cloud transmission factors
        """
        # UK cloud types and their transmission characteristics
        season = self._get_season(dt.month)
        
        # Seasonal cloud transmission factors for UK
        seasonal_factors = {
            'winter': {'base': 0.15, 'enhancement': 0.05},
            'spring': {'base': 0.25, 'enhancement': 0.08},
            'summer': {'base': 0.35, 'enhancement': 0.12},
            'autumn': {'base': 0.20, 'enhancement': 0.06}
        }
        
        base_transmission = seasonal_factors[season]['base']
        enhancement_factor = seasonal_factors[season]['enhancement']
        
        # Cloud transmission model for UK conditions
        if cloud_cover <= 0.1:
            # Clear sky
            direct_transmission = 1.0
            diffuse_enhancement = 1.0
        elif cloud_cover <= 0.5:
            # Partly cloudy - UK often has broken clouds
            direct_transmission = 1.0 - 0.6 * cloud_cover
            diffuse_enhancement = 1.0 + enhancement_factor * cloud_cover
        elif cloud_cover <= 0.8:
            # Mostly cloudy - common UK condition
            direct_transmission = base_transmission + (1.0 - base_transmission) * (0.8 - cloud_cover) / 0.3
            diffuse_enhancement = 1.0 + enhancement_factor * 0.5
        else:
            # Overcast - very common in UK
            direct_transmission = base_transmission * (1.0 - cloud_cover) / 0.2
            diffuse_enhancement = 0.8 + 0.2 * (1.0 - cloud_cover)
        
        return {
            'direct_transmission': max(0.0, min(direct_transmission, 1.0)),
            'diffuse_enhancement': max(0.5, min(diffuse_enhancement, 1.5)),
            'global_transmission': max(0.1, min(direct_transmission * 0.7 + 0.3, 1.0))
        }
    
    def uk_aerosol_model(self, visibility: Optional[float] = None, 
                        urban_proximity: float = 0.5) -> float:
        """
        Model aerosol effects for UK conditions.
        
        Args:
            visibility: Visibility in km (if available)
            urban_proximity: Urban proximity factor (0-1, 1=urban center)
            
        Returns:
            Aerosol optical depth at 550nm
        """
        # UK baseline aerosol optical depth
        baseline_aod = {
            'scotland': 0.08,
            'northern_england': 0.12,
            'midlands': 0.15,
            'wales': 0.09,
            'southern_england': 0.13
        }
        
        base_aod = baseline_aod.get(self.region, 0.12)
        
        # Urban enhancement
        urban_enhancement = 1.0 + urban_proximity * 0.5
        
        # Visibility-based correction if available
        if visibility is not None:
            # Koschmieder relationship adapted for UK
            if visibility > 0:
                visibility_aod = 3.912 / visibility - 0.01159
                visibility_aod = max(0.05, min(visibility_aod, 0.5))
                # Blend with baseline
                aod = 0.7 * visibility_aod + 0.3 * base_aod * urban_enhancement
            else:
                aod = base_aod * urban_enhancement * 2.0  # Heavy pollution/fog
        else:
            aod = base_aod * urban_enhancement
        
        return max(0.05, min(aod, 0.8))
    
    def precipitable_water(self, temperature: float, humidity: float, 
                          pressure: float = 1013.25) -> float:
        """
        Calculate precipitable water for UK atmospheric conditions.
        
        Args:
            temperature: Air temperature in °C
            humidity: Relative humidity in %
            pressure: Atmospheric pressure in hPa
            
        Returns:
            Precipitable water in cm
        """
        # Saturation vapor pressure (Tetens formula)
        es = 6.1078 * math.exp(17.27 * temperature / (temperature + 237.3))
        
        # Actual vapor pressure
        e = es * humidity / 100.0
        
        # Precipitable water (simplified model for UK)
        # UK typically has high humidity, so use enhanced formula
        pw = 0.14 * e * pressure / 1013.25
        
        # UK seasonal adjustments
        month = datetime.now().month
        seasonal_factor = {
            1: 0.8, 2: 0.8, 3: 0.9, 4: 1.0, 5: 1.1, 6: 1.2,
            7: 1.3, 8: 1.2, 9: 1.1, 10: 1.0, 11: 0.9, 12: 0.8
        }
        
        pw *= seasonal_factor.get(month, 1.0)
        
        return max(0.5, min(pw, 6.0))  # UK typical range
    
    def uk_atmospheric_correction(self, ghi: float, dni: float, dhi: float,
                                 temperature: float, humidity: float,
                                 pressure: float, cloud_cover: float,
                                 dt: datetime) -> Tuple[float, float, float]:
        """
        Apply comprehensive UK atmospheric corrections to solar irradiance.
        
        Args:
            ghi, dni, dhi: Solar irradiance components in W/m²
            temperature: Air temperature in °C
            humidity: Relative humidity in %
            pressure: Atmospheric pressure in hPa
            cloud_cover: Cloud cover fraction (0-1)
            dt: Datetime object
            
        Returns:
            Corrected (GHI, DNI, DHI) in W/m²
        """
        # Cloud effects
        cloud_factors = self.uk_cloud_model(cloud_cover, dt)
        
        # Aerosol effects (estimate urban proximity from region)
        urban_factors = {
            'scotland': 0.2, 'northern_england': 0.6, 'midlands': 0.8,
            'wales': 0.3, 'southern_england': 0.5
        }
        urban_proximity = urban_factors.get(self.region, 0.5)
        aod = self.uk_aerosol_model(urban_proximity=urban_proximity)
        
        # Water vapor effects
        pw = self.precipitable_water(temperature, humidity, pressure)
        water_transmission = math.exp(-0.077 * pw**0.3)
        
        # Apply corrections
        corrected_dni = dni * cloud_factors['direct_transmission'] * \
                       math.exp(-aod * 1.5) * water_transmission
        
        corrected_dhi = dhi * cloud_factors['diffuse_enhancement'] * \
                       (1.0 + aod * 0.5) * water_transmission
        
        corrected_ghi = ghi * cloud_factors['global_transmission'] * \
                       math.exp(-aod * 0.8) * water_transmission
        
        # UK-specific bounds checking
        corrected_ghi = max(0.0, min(corrected_ghi, 1200.0))
        corrected_dni = max(0.0, min(corrected_dni, 1000.0))
        corrected_dhi = max(0.0, min(corrected_dhi, 500.0))
        
        return corrected_ghi, corrected_dni, corrected_dhi


class UKWeatherPatterns:
    """
    UK-specific weather pattern analysis for solar forecasting.
    """
    
    # UK weather pattern classifications
    LAMB_WEATHER_TYPES = [
        'A', 'AE', 'AS', 'AW', 'AN',  # Anticyclonic types
        'C', 'CE', 'CS', 'CW', 'CN',  # Cyclonic types
        'E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'  # Directional types
    ]
    
    # Solar radiation characteristics by weather type
    WEATHER_TYPE_SOLAR = {
        'A': {'ghi_factor': 0.85, 'variability': 0.1},    # Anticyclonic
        'C': {'ghi_factor': 0.35, 'variability': 0.4},    # Cyclonic
        'E': {'ghi_factor': 0.45, 'variability': 0.3},    # Easterly
        'W': {'ghi_factor': 0.40, 'variability': 0.35},   # Westerly
        'N': {'ghi_factor': 0.50, 'variability': 0.25},   # Northerly
        'S': {'ghi_factor': 0.65, 'variability': 0.2},    # Southerly
    }
    
    def __init__(self):
        """Initialize UK weather pattern analyzer."""
        pass
    
    def classify_weather_pattern(self, pressure_data: Dict[str, float],
                               wind_direction: float, wind_speed: float) -> str:
        """
        Classify current weather pattern using simplified Lamb classification.
        
        Args:
            pressure_data: Dictionary with pressure readings from different locations
            wind_direction: Wind direction in degrees
            wind_speed: Wind speed in m/s
            
        Returns:
            Lamb weather type classification
        """
        # Simplified classification based on available data
        if wind_speed < 2.0:
            return 'A'  # Anticyclonic (light winds)
        elif wind_speed > 8.0:
            return 'C'  # Cyclonic (strong winds)
        else:
            # Directional type based on wind direction
            if 337.5 <= wind_direction or wind_direction < 22.5:
                return 'N'
            elif 22.5 <= wind_direction < 67.5:
                return 'NE'
            elif 67.5 <= wind_direction < 112.5:
                return 'E'
            elif 112.5 <= wind_direction < 157.5:
                return 'SE'
            elif 157.5 <= wind_direction < 202.5:
                return 'S'
            elif 202.5 <= wind_direction < 247.5:
                return 'SW'
            elif 247.5 <= wind_direction < 292.5:
                return 'W'
            else:
                return 'NW'
    
    def solar_forecast_adjustment(self, base_forecast: float, 
                                weather_type: str, hour: int) -> float:
        """
        Adjust solar forecast based on UK weather pattern.
        
        Args:
            base_forecast: Base solar irradiance forecast
            weather_type: Lamb weather type
            hour: Hour of day (0-23)
            
        Returns:
            Adjusted solar irradiance forecast
        """
        # Get weather type characteristics
        pattern_char = self.WEATHER_TYPE_SOLAR.get(
            weather_type[0], {'ghi_factor': 0.5, 'variability': 0.3}
        )
        
        # Apply base adjustment
        adjusted_forecast = base_forecast * pattern_char['ghi_factor']
        
        # Add diurnal variation based on weather type
        if weather_type.startswith('A'):  # Anticyclonic - clear skies
            diurnal_factor = 1.0 + 0.3 * math.sin(math.pi * (hour - 6) / 12)
        elif weather_type.startswith('C'):  # Cyclonic - variable clouds
            diurnal_factor = 1.0 + 0.1 * math.sin(math.pi * (hour - 6) / 12)
        else:  # Directional - moderate variation
            diurnal_factor = 1.0 + 0.2 * math.sin(math.pi * (hour - 6) / 12)
        
        diurnal_factor = max(0.5, min(diurnal_factor, 1.5))
        
        return adjusted_forecast * diurnal_factor


def calculate_uk_atmospheric_features(df: pd.DataFrame, latitude: float, 
                                    longitude: float, elevation: float = 0.0) -> pd.DataFrame:
    """
    Calculate UK-specific atmospheric features for solar prediction.
    
    Args:
        df: DataFrame with weather and solar data
        latitude: Site latitude
        longitude: Site longitude
        elevation: Site elevation
        
    Returns:
        DataFrame with added atmospheric features
    """
    atm_model = UKAtmosphericModel(latitude, longitude, elevation)
    weather_patterns = UKWeatherPatterns()
    
    df = df.copy()
    
    # Required columns check
    required_cols = ['timestamp', 'temperature', 'humidity', 'pressure', 'cloud_cover']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate atmospheric features
    results = []
    
    for idx, row in df.iterrows():
        dt = pd.to_datetime(row['timestamp'])
        
        # Linke turbidity
        turbidity = atm_model.linke_turbidity(dt)
        
        # Precipitable water
        pw = atm_model.precipitable_water(
            row['temperature'], row['humidity'], row['pressure']
        )
        
        # Aerosol optical depth
        aod = atm_model.uk_aerosol_model()
        
        # Cloud transmission factors
        cloud_factors = atm_model.uk_cloud_model(row['cloud_cover'], dt)
        
        # Weather pattern classification (if wind data available)
        if 'wind_direction' in df.columns and 'wind_speed' in df.columns:
            weather_type = weather_patterns.classify_weather_pattern(
                {'pressure': row['pressure']},
                row['wind_direction'],
                row['wind_speed']
            )
        else:
            weather_type = 'unknown'
        
        results.append({
            'linke_turbidity': turbidity,
            'precipitable_water': pw,
            'aerosol_optical_depth': aod,
            'cloud_direct_transmission': cloud_factors['direct_transmission'],
            'cloud_diffuse_enhancement': cloud_factors['diffuse_enhancement'],
            'cloud_global_transmission': cloud_factors['global_transmission'],
            'weather_pattern': weather_type
        })
    
    # Add results to dataframe
    for key in results[0].keys():
        df[key] = [r[key] for r in results]
    
    return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from datetime import datetime, timezone
    
    # Test data for a UK location
    test_data = {
        'timestamp': pd.date_range('2024-01-01', periods=24, freq='H', tz='UTC'),
        'temperature': np.random.normal(8, 5, 24),  # UK winter temperatures
        'humidity': np.random.normal(80, 10, 24),   # UK humidity levels
        'pressure': np.random.normal(1013, 10, 24), # UK pressure
        'cloud_cover': np.random.uniform(0.3, 0.9, 24),  # UK cloud cover
        'wind_direction': np.random.uniform(0, 360, 24),
        'wind_speed': np.random.uniform(2, 12, 24)
    }
    
    df = pd.DataFrame(test_data)
    
    # Calculate atmospheric features
    result_df = calculate_uk_atmospheric_features(df, 51.5, -0.1)
    
    print("UK atmospheric features calculated:")
    print(result_df[['timestamp', 'linke_turbidity', 'precipitable_water', 
                    'cloud_global_transmission']].head())

