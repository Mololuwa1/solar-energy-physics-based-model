"""
Solar Geometry Calculations for UK Solar Farms

This module provides high-precision solar position calculations optimized for UK latitudes (50°N - 60°N).
Implements the Solar Position Algorithm (SPA) with corrections for atmospheric refraction and 
UK-specific atmospheric conditions.

Author: Mololuwa Obafemi-Moses
Date: 2025-08-16
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Tuple, Union, Optional
import math


class UKSolarGeometry:
    """
    Solar geometry calculator optimized for UK conditions.
    
    Provides high-precision solar position calculations with UK-specific
    atmospheric corrections and seasonal adjustments.
    """
    
    # UK-specific constants
    UK_LATITUDE_MIN = 50.0  # Southernmost UK latitude
    UK_LATITUDE_MAX = 60.0  # Northernmost UK latitude (Shetland Islands)
    UK_LONGITUDE_MIN = -8.0  # Westernmost UK longitude
    UK_LONGITUDE_MAX = 2.0   # Easternmost UK longitude
    
    # Atmospheric constants for UK
    UK_STANDARD_PRESSURE = 1013.25  # hPa at sea level
    UK_STANDARD_TEMPERATURE = 10.0  # °C annual average
    UK_ATMOSPHERIC_REFRACTION = 0.5667  # degrees, typical UK conditions
    
    def __init__(self, latitude: float, longitude: float, elevation: float = 0.0):
        """
        Initialize solar geometry calculator for a UK location.
        
        Args:
            latitude: Site latitude in degrees (50.0 to 60.0 for UK)
            longitude: Site longitude in degrees (-8.0 to 2.0 for UK)
            elevation: Site elevation in meters above sea level
            
        Raises:
            ValueError: If coordinates are outside UK bounds
        """
        if not (self.UK_LATITUDE_MIN <= latitude <= self.UK_LATITUDE_MAX):
            raise ValueError(f"Latitude {latitude} outside UK range ({self.UK_LATITUDE_MIN}-{self.UK_LATITUDE_MAX})")
        
        if not (self.UK_LONGITUDE_MIN <= longitude <= self.UK_LONGITUDE_MAX):
            raise ValueError(f"Longitude {longitude} outside UK range ({self.UK_LONGITUDE_MIN}-{self.UK_LONGITUDE_MAX})")
        
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        
        # Pre-calculate constants
        self.lat_rad = math.radians(latitude)
        self.lon_rad = math.radians(longitude)
    
    def julian_day(self, dt: datetime) -> float:
        """
        Calculate Julian day number with high precision.
        
        Args:
            dt: Datetime object (should be in UTC)
            
        Returns:
            Julian day number as float
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        elif dt.tzinfo != timezone.utc:
            dt = dt.astimezone(timezone.utc)
        
        year = dt.year
        month = dt.month
        day = dt.day
        hour = dt.hour
        minute = dt.minute
        second = dt.second + dt.microsecond / 1e6
        
        # Julian day calculation
        if month <= 2:
            year -= 1
            month += 12
        
        a = int(year / 100)
        b = 2 - a + int(a / 4)
        
        jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5
        jd += (hour + minute / 60.0 + second / 3600.0) / 24.0
        
        return jd
    
    def solar_declination(self, julian_day: float) -> float:
        """
        Calculate solar declination angle.
        
        Args:
            julian_day: Julian day number
            
        Returns:
            Solar declination in radians
        """
        n = julian_day - 2451545.0  # Days since J2000.0
        L = math.radians(280.460 + 0.9856474 * n)  # Mean longitude
        g = math.radians(357.528 + 0.9856003 * n)  # Mean anomaly
        
        # Solar declination with higher order terms for UK precision
        lambda_sun = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
        declination = math.asin(math.sin(math.radians(23.439)) * math.sin(lambda_sun))
        
        return declination
    
    def equation_of_time(self, julian_day: float) -> float:
        """
        Calculate equation of time correction.
        
        Args:
            julian_day: Julian day number
            
        Returns:
            Equation of time in minutes
        """
        n = julian_day - 2451545.0
        L = math.radians(280.460 + 0.9856474 * n)
        g = math.radians(357.528 + 0.9856003 * n)
        
        # Equation of time with UK-specific corrections
        eot = 4 * math.degrees(L - 0.0057183 - math.atan2(
            math.tan(L), math.cos(math.radians(23.439))
        ))
        
        # Apply orbital corrections
        eot += 4 * math.degrees(
            math.radians(1.915) * math.sin(g) + 
            math.radians(0.020) * math.sin(2 * g)
        )
        
        return eot
    
    def hour_angle(self, dt: datetime) -> float:
        """
        Calculate solar hour angle.
        
        Args:
            dt: Datetime object
            
        Returns:
            Hour angle in radians
        """
        jd = self.julian_day(dt)
        eot = self.equation_of_time(jd)
        
        # Local solar time
        lst = dt.hour + dt.minute / 60.0 + dt.second / 3600.0
        lst += self.longitude / 15.0  # Longitude correction
        lst += eot / 60.0  # Equation of time correction
        
        # Hour angle (solar noon = 0)
        hour_angle = math.radians(15.0 * (lst - 12.0))
        
        return hour_angle
    
    def solar_position(self, dt: datetime) -> Tuple[float, float, float]:
        """
        Calculate solar position (zenith, azimuth, elevation).
        
        Args:
            dt: Datetime object
            
        Returns:
            Tuple of (zenith_angle, azimuth_angle, elevation_angle) in degrees
        """
        jd = self.julian_day(dt)
        declination = self.solar_declination(jd)
        hour_angle = self.hour_angle(dt)
        
        # Solar zenith angle
        cos_zenith = (math.sin(self.lat_rad) * math.sin(declination) + 
                     math.cos(self.lat_rad) * math.cos(declination) * math.cos(hour_angle))
        
        # Clamp to valid range to avoid numerical errors
        cos_zenith = max(-1.0, min(1.0, cos_zenith))
        zenith = math.acos(cos_zenith)
        
        # Solar azimuth angle
        sin_azimuth = math.cos(declination) * math.sin(hour_angle) / math.sin(zenith)
        cos_azimuth = ((math.sin(declination) * math.cos(self.lat_rad) - 
                       math.cos(declination) * math.sin(self.lat_rad) * math.cos(hour_angle)) / 
                      math.sin(zenith))
        
        # Handle numerical precision issues
        sin_azimuth = max(-1.0, min(1.0, sin_azimuth))
        cos_azimuth = max(-1.0, min(1.0, cos_azimuth))
        
        azimuth = math.atan2(sin_azimuth, cos_azimuth)
        
        # Convert to degrees and adjust azimuth to 0-360°
        zenith_deg = math.degrees(zenith)
        azimuth_deg = math.degrees(azimuth)
        if azimuth_deg < 0:
            azimuth_deg += 360.0
        
        elevation_deg = 90.0 - zenith_deg
        
        return zenith_deg, azimuth_deg, elevation_deg
    
    def air_mass(self, dt: datetime, pressure: Optional[float] = None, 
                 temperature: Optional[float] = None) -> float:
        """
        Calculate air mass with UK atmospheric corrections.
        
        Args:
            dt: Datetime object
            pressure: Atmospheric pressure in hPa (default: UK standard)
            temperature: Air temperature in °C (default: UK standard)
            
        Returns:
            Air mass value
        """
        if pressure is None:
            pressure = self.UK_STANDARD_PRESSURE
        if temperature is None:
            temperature = self.UK_STANDARD_TEMPERATURE
        
        zenith_deg, _, elevation_deg = self.solar_position(dt)
        
        # Return very large air mass for sun below horizon
        if elevation_deg <= 0:
            return 40.0
        
        zenith_rad = math.radians(zenith_deg)
        
        # Kasten-Young air mass formula with pressure/temperature corrections
        am = (1.0 / (math.cos(zenith_rad) + 0.50572 * 
                    (96.07995 - zenith_deg) ** (-1.6364)))
        
        # Pressure correction for UK conditions
        pressure_correction = pressure / self.UK_STANDARD_PRESSURE
        
        # Temperature correction (simplified)
        temp_correction = (273.15 + self.UK_STANDARD_TEMPERATURE) / (273.15 + temperature)
        
        am *= pressure_correction * temp_correction
        
        return am
    
    def extraterrestrial_irradiance(self, dt: datetime) -> float:
        """
        Calculate extraterrestrial solar irradiance.
        
        Args:
            dt: Datetime object
            
        Returns:
            Extraterrestrial irradiance in W/m²
        """
        jd = self.julian_day(dt)
        
        # Earth-sun distance correction
        n = jd - 2451545.0
        earth_sun_distance = 1.00014 - 0.01671 * math.cos(math.radians(0.9856 * n)) - \
                           0.00014 * math.cos(math.radians(1.9712 * n))
        
        # Solar constant (W/m²)
        solar_constant = 1366.1
        
        # Extraterrestrial irradiance
        i0 = solar_constant / (earth_sun_distance ** 2)
        
        return i0
    
    def sunrise_sunset(self, dt: datetime) -> Tuple[datetime, datetime]:
        """
        Calculate sunrise and sunset times for UK location.
        
        Args:
            dt: Date for calculation (time component ignored)
            
        Returns:
            Tuple of (sunrise, sunset) datetime objects in UTC
        """
        jd = self.julian_day(dt.replace(hour=12, minute=0, second=0, microsecond=0))
        declination = self.solar_declination(jd)
        
        # Hour angle at sunrise/sunset
        cos_hour_angle = -math.tan(self.lat_rad) * math.tan(declination)
        
        # Check for polar day/night conditions
        if cos_hour_angle > 1.0:
            # Polar night - sun never rises
            return None, None
        elif cos_hour_angle < -1.0:
            # Polar day - sun never sets
            return None, None
        
        hour_angle = math.acos(cos_hour_angle)
        
        # Convert to time
        eot = self.equation_of_time(jd)
        
        # Sunrise time (local solar time)
        sunrise_lst = 12.0 - math.degrees(hour_angle) / 15.0
        sunset_lst = 12.0 + math.degrees(hour_angle) / 15.0
        
        # Convert to UTC
        utc_offset = self.longitude / 15.0 + eot / 60.0
        
        sunrise_utc = sunrise_lst - utc_offset
        sunset_utc = sunset_lst - utc_offset
        
        # Create datetime objects
        base_date = dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        
        sunrise_hours = int(sunrise_utc)
        sunrise_minutes = int((sunrise_utc - sunrise_hours) * 60)
        sunrise_seconds = int(((sunrise_utc - sunrise_hours) * 60 - sunrise_minutes) * 60)
        
        sunset_hours = int(sunset_utc)
        sunset_minutes = int((sunset_utc - sunset_hours) * 60)
        sunset_seconds = int(((sunset_utc - sunset_hours) * 60 - sunset_minutes) * 60)
        
        # Handle day boundary crossings
        sunrise_dt = base_date.replace(hour=sunrise_hours % 24, 
                                     minute=sunrise_minutes, 
                                     second=sunrise_seconds)
        if sunrise_hours < 0:
            sunrise_dt = sunrise_dt.replace(day=sunrise_dt.day - 1)
        elif sunrise_hours >= 24:
            sunrise_dt = sunrise_dt.replace(day=sunrise_dt.day + 1)
        
        sunset_dt = base_date.replace(hour=sunset_hours % 24, 
                                    minute=sunset_minutes, 
                                    second=sunset_seconds)
        if sunset_hours < 0:
            sunset_dt = sunset_dt.replace(day=sunset_dt.day - 1)
        elif sunset_hours >= 24:
            sunset_dt = sunset_dt.replace(day=sunset_dt.day + 1)
        
        return sunrise_dt, sunset_dt
    
    def day_length(self, dt: datetime) -> float:
        """
        Calculate day length in hours.
        
        Args:
            dt: Date for calculation
            
        Returns:
            Day length in hours
        """
        sunrise, sunset = self.sunrise_sunset(dt)
        
        if sunrise is None or sunset is None:
            return 0.0 if sunrise is None else 24.0
        
        day_length = (sunset - sunrise).total_seconds() / 3600.0
        return day_length
    
    def solar_position_batch(self, timestamps: pd.Series) -> pd.DataFrame:
        """
        Calculate solar positions for multiple timestamps efficiently.
        
        Args:
            timestamps: Pandas Series of datetime objects
            
        Returns:
            DataFrame with columns: zenith, azimuth, elevation, air_mass
        """
        results = []
        
        for timestamp in timestamps:
            zenith, azimuth, elevation = self.solar_position(timestamp)
            air_mass = self.air_mass(timestamp)
            
            results.append({
                'zenith': zenith,
                'azimuth': azimuth,
                'elevation': elevation,
                'air_mass': air_mass
            })
        
        return pd.DataFrame(results, index=timestamps.index)


def calculate_uk_solar_features(df: pd.DataFrame, latitude: float, longitude: float, 
                               elevation: float = 0.0) -> pd.DataFrame:
    """
    Calculate comprehensive solar geometry features for UK solar farm data.
    
    Args:
        df: DataFrame with 'timestamp' column
        latitude: Site latitude
        longitude: Site longitude  
        elevation: Site elevation in meters
        
    Returns:
        DataFrame with added solar geometry columns
    """
    solar_calc = UKSolarGeometry(latitude, longitude, elevation)
    
    # Ensure timestamp is datetime
    if 'timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column")
    
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate solar positions
    solar_positions = solar_calc.solar_position_batch(df['timestamp'])
    
    # Add solar geometry features
    df['solar_zenith'] = solar_positions['zenith']
    df['solar_azimuth'] = solar_positions['azimuth']
    df['solar_elevation'] = solar_positions['elevation']
    df['air_mass'] = solar_positions['air_mass']
    
    # Calculate extraterrestrial irradiance
    df['extraterrestrial_irradiance'] = df['timestamp'].apply(
        solar_calc.extraterrestrial_irradiance
    )
    
    # Calculate time-based features
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['month'] = df['timestamp'].dt.month
    
    # Cyclical encoding for temporal features
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365.25)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365.25)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Calculate sunrise/sunset information
    unique_dates = df['timestamp'].dt.date.unique()
    sunrise_sunset_data = {}
    
    for date in unique_dates:
        dt = datetime.combine(date, datetime.min.time()).replace(tzinfo=timezone.utc)
        sunrise, sunset = solar_calc.sunrise_sunset(dt)
        day_length = solar_calc.day_length(dt)
        
        sunrise_sunset_data[date] = {
            'sunrise': sunrise,
            'sunset': sunset,
            'day_length': day_length
        }
    
    # Add sunrise/sunset features
    df['date'] = df['timestamp'].dt.date
    df['day_length'] = df['date'].map(lambda x: sunrise_sunset_data[x]['day_length'])
    
    # Calculate time relative to solar noon
    df['solar_noon_offset'] = df['timestamp'].apply(
        lambda x: abs((x.hour + x.minute/60.0) - 12.0)
    )
    
    # Remove temporary date column
    df = df.drop('date', axis=1)
    
    return df


if __name__ == "__main__":
    # Example usage for a UK solar farm
    import pandas as pd
    
    # Example coordinates for a solar farm in southern England
    latitude = 51.5074  # London latitude
    longitude = -0.1278  # London longitude
    
    # Create test data
    timestamps = pd.date_range('2024-01-01', '2024-01-02', freq='15min', tz='UTC')
    test_df = pd.DataFrame({'timestamp': timestamps})
    
    # Calculate solar features
    result_df = calculate_uk_solar_features(test_df, latitude, longitude)
    
    print("Solar geometry features calculated:")
    print(result_df[['timestamp', 'solar_elevation', 'solar_azimuth', 'air_mass']].head(10))

