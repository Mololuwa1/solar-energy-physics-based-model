#!/usr/bin/env python3
"""
Environment Setup Script for UK Solar Energy Prediction System

This script sets up the complete environment for the physics-informed solar
energy prediction model, including dependencies, configuration, and validation.

Author: Manus AI
Date: 2025-08-16
"""

import os
import sys
import subprocess
import json
import yaml
from pathlib import Path
import argparse
import shutil
import tempfile
from datetime import datetime


class EnvironmentSetup:
    """Handles environment setup and configuration."""
    
    def __init__(self, project_root=None):
        """
        Initialize environment setup.
        
        Args:
            project_root: Path to project root directory
        """
        self.project_root = Path(project_root) if project_root else Path(__file__).parent.parent
        self.setup_log = []
        
    def log(self, message, level='INFO'):
        """Log setup messages."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        self.setup_log.append(log_entry)
        print(log_entry)
    
    def check_python_version(self):
        """Check Python version compatibility."""
        self.log("Checking Python version...")
        
        version = sys.version_info
        if version.major != 3 or version.minor < 8:
            self.log(f"Python {version.major}.{version.minor} detected. Python 3.8+ required.", 'ERROR')
            return False
        
        self.log(f"Python {version.major}.{version.minor}.{version.micro} - Compatible ✓")
        return True
    
    def install_dependencies(self, dev_mode=False):
        """Install Python dependencies."""
        self.log("Installing Python dependencies...")
        
        requirements_file = self.project_root / 'requirements.txt'
        if not requirements_file.exists():
            self.log("requirements.txt not found", 'ERROR')
            return False
        
        try:
            # Install main requirements
            cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.log(f"Failed to install requirements: {result.stderr}", 'ERROR')
                return False
            
            self.log("Main dependencies installed ✓")
            
            # Install development dependencies if requested
            if dev_mode:
                dev_requirements = self.project_root / 'requirements-dev.txt'
                if dev_requirements.exists():
                    cmd = [sys.executable, '-m', 'pip', 'install', '-r', str(dev_requirements)]
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        self.log("Development dependencies installed ✓")
                    else:
                        self.log(f"Failed to install dev requirements: {result.stderr}", 'WARNING')
            
            return True
            
        except Exception as e:
            self.log(f"Error installing dependencies: {e}", 'ERROR')
            return False
    
    def setup_directories(self):
        """Create necessary directories."""
        self.log("Setting up directory structure...")
        
        directories = [
            'data/raw',
            'data/processed',
            'data/external',
            'models/trained',
            'models/checkpoints',
            'notebooks/experiments',
            'logs',
            'outputs',
            'configs'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            self.log(f"Created directory: {directory}")
        
        self.log("Directory structure created ✓")
        return True
    
    def create_sample_config(self):
        """Create sample configuration files."""
        self.log("Creating sample configuration files...")
        
        # Sample solar farm configuration
        sample_config = {
            'name': 'UK Sample Solar Farm',
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
        
        config_path = self.project_root / 'configs' / 'sample_solar_farm.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(sample_config, f, default_flow_style=False)
        
        self.log(f"Sample config created: {config_path}")
        
        # Training configuration
        training_config = {
            'model': {
                'input_size': 50,
                'hidden_size': 128,
                'num_layers': 3,
                'dropout': 0.1,
                'sequence_length': 24,
                'prediction_horizon': 1
            },
            'training': {
                'batch_size': 64,
                'num_epochs': 100,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'early_stopping_patience': 10,
                'validation_split': 0.2
            },
            'physics': {
                'energy_conservation_weight': 1.0,
                'thermodynamic_weight': 0.5,
                'atmospheric_weight': 0.3,
                'temporal_consistency_weight': 0.2,
                'solar_geometry_weight': 0.4
            }
        }
        
        training_config_path = self.project_root / 'configs' / 'training_config.yaml'
        with open(training_config_path, 'w') as f:
            yaml.dump(training_config, f, default_flow_style=False)
        
        self.log(f"Training config created: {training_config_path}")
        
        return True
    
    def setup_environment_variables(self):
        """Setup environment variables."""
        self.log("Setting up environment variables...")
        
        env_file = self.project_root / '.env'
        
        env_vars = {
            'PYTHONPATH': str(self.project_root / 'src'),
            'UK_SOLAR_PROJECT_ROOT': str(self.project_root),
            'UK_SOLAR_DATA_DIR': str(self.project_root / 'data'),
            'UK_SOLAR_MODEL_DIR': str(self.project_root / 'models'),
            'UK_SOLAR_LOG_DIR': str(self.project_root / 'logs'),
            'UK_SOLAR_CONFIG_DIR': str(self.project_root / 'configs')
        }
        
        with open(env_file, 'w') as f:
            f.write("# UK Solar Energy Prediction Environment Variables\n")
            f.write(f"# Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        self.log(f"Environment file created: {env_file}")
        
        # Set environment variables for current session
        for key, value in env_vars.items():
            os.environ[key] = value
        
        return True
    
    def validate_installation(self):
        """Validate the installation."""
        self.log("Validating installation...")
        
        # Check if key modules can be imported
        test_imports = [
            'numpy',
            'pandas',
            'torch',
            'sklearn',
            'matplotlib',
            'seaborn',
            'yaml',
            'requests'
        ]
        
        failed_imports = []
        
        for module in test_imports:
            try:
                __import__(module)
                self.log(f"✓ {module}")
            except ImportError:
                self.log(f"✗ {module}", 'ERROR')
                failed_imports.append(module)
        
        if failed_imports:
            self.log(f"Failed to import: {', '.join(failed_imports)}", 'ERROR')
            return False
        
        # Check project structure
        required_dirs = [
            'src',
            'tests',
            'sagemaker',
            'scripts',
            'data',
            'models'
        ]
        
        missing_dirs = []
        for directory in required_dirs:
            if not (self.project_root / directory).exists():
                missing_dirs.append(directory)
        
        if missing_dirs:
            self.log(f"Missing directories: {', '.join(missing_dirs)}", 'ERROR')
            return False
        
        self.log("Installation validation completed ✓")
        return True
    
    def run_basic_tests(self):
        """Run basic functionality tests."""
        self.log("Running basic functionality tests...")
        
        try:
            # Test data pipeline
            sys.path.insert(0, str(self.project_root / 'src'))
            
            from data.data_pipeline import SolarFarmConfig
            from physics.solar_geometry import calculate_uk_solar_features
            
            # Create test configuration
            config = SolarFarmConfig(
                name="Test Farm",
                latitude=51.5,
                longitude=-0.1,
                elevation=50.0,
                capacity_mw=50.0,
                panel_tilt=35.0,
                panel_azimuth=180.0,
                panel_technology="c-Si",
                commissioning_date=datetime(2020, 1, 1),
                region="uk"
            )
            
            self.log("✓ Data pipeline components")
            
            # Test model creation
            from models.physics_informed_model import PhysicsInformedSolarModel
            
            model = PhysicsInformedSolarModel(
                input_size=10,
                hidden_size=32,
                num_layers=2
            )
            
            self.log("✓ Model creation")
            
            # Test basic forward pass
            import torch
            x = torch.randn(1, 24, 10)
            outputs = model(x)
            
            if 'prediction' in outputs:
                self.log("✓ Model forward pass")
            else:
                self.log("✗ Model forward pass", 'ERROR')
                return False
            
            self.log("Basic functionality tests completed ✓")
            return True
            
        except Exception as e:
            self.log(f"Basic tests failed: {e}", 'ERROR')
            return False
    
    def create_quick_start_guide(self):
        """Create quick start guide."""
        self.log("Creating quick start guide...")
        
        guide_content = f"""# UK Solar Energy Prediction - Quick Start Guide

## Environment Setup Complete!

Your UK Solar Energy Prediction environment has been successfully set up on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.

## Project Structure

```
uk_solar_prediction/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── physics/           # Physics calculations
├── tests/                 # Test suites
├── sagemaker/            # AWS SageMaker deployment
├── scripts/              # Utility scripts
├── configs/              # Configuration files
├── data/                 # Data directories
└── models/               # Model storage
```

## Quick Start Commands

### 1. Run Tests
```bash
# Run all tests
python scripts/run_tests.py

# Run specific test module
python scripts/run_tests.py -m test_physics_model

# Run with coverage
python scripts/run_tests.py --coverage
```

### 2. Train a Model
```bash
# Local training
python src/models/trainer.py --config configs/training_config.yaml

# SageMaker training
python sagemaker/train.py --role YOUR_SAGEMAKER_ROLE
```

### 3. Deploy to SageMaker
```bash
python sagemaker/deploy.py \\
    --role YOUR_SAGEMAKER_ROLE \\
    --model-data-url s3://your-bucket/model.tar.gz \\
    --endpoint-name uk-solar-endpoint
```

### 4. Process Data
```bash
python -c "
from src.data.data_pipeline import UKSolarDataProcessor
from src.data.data_pipeline import load_solar_farm_config

config = load_solar_farm_config('configs/sample_solar_farm.yaml')
processor = UKSolarDataProcessor(config)
# Add your data processing code here
"
```

## Configuration

### Solar Farm Configuration
Edit `configs/sample_solar_farm.yaml` with your solar farm details:
- Location (latitude, longitude, elevation)
- Capacity and panel specifications
- Regional settings

### Training Configuration
Edit `configs/training_config.yaml` for model training:
- Model architecture parameters
- Training hyperparameters
- Physics constraint weights

## Environment Variables

The following environment variables have been set:
- `PYTHONPATH`: {os.environ.get('PYTHONPATH', 'Not set')}
- `UK_SOLAR_PROJECT_ROOT`: {os.environ.get('UK_SOLAR_PROJECT_ROOT', 'Not set')}
- `UK_SOLAR_DATA_DIR`: {os.environ.get('UK_SOLAR_DATA_DIR', 'Not set')}

## Next Steps

1. **Prepare Your Data**: Place your solar farm data in `data/raw/`
2. **Configure Your Farm**: Update `configs/sample_solar_farm.yaml`
3. **Train Your Model**: Run training with your data
4. **Deploy to AWS**: Use SageMaker deployment scripts
5. **Monitor Performance**: Use the testing framework

## Getting Help

- Check the main README.md for detailed documentation
- Run tests to validate your setup: `python scripts/run_tests.py`
- Review example notebooks in `notebooks/`

## Troubleshooting

If you encounter issues:
1. Check the setup log in `setup_log.txt`
2. Verify all dependencies are installed
3. Ensure AWS credentials are configured (for SageMaker)
4. Run the validation tests

Happy predicting! 🌞⚡
"""
        
        guide_path = self.project_root / 'QUICK_START.md'
        with open(guide_path, 'w') as f:
            f.write(guide_content)
        
        self.log(f"Quick start guide created: {guide_path}")
        return True
    
    def save_setup_log(self):
        """Save setup log to file."""
        log_path = self.project_root / 'setup_log.txt'
        
        with open(log_path, 'w') as f:
            f.write("UK Solar Energy Prediction System - Setup Log\n")
            f.write("=" * 50 + "\n\n")
            
            for entry in self.setup_log:
                f.write(entry + "\n")
        
        self.log(f"Setup log saved: {log_path}")
    
    def run_full_setup(self, dev_mode=False, skip_tests=False):
        """Run complete environment setup."""
        self.log("Starting UK Solar Energy Prediction environment setup...")
        
        success = True
        
        # Check Python version
        if not self.check_python_version():
            success = False
        
        # Install dependencies
        if success and not self.install_dependencies(dev_mode):
            success = False
        
        # Setup directories
        if success and not self.setup_directories():
            success = False
        
        # Create configurations
        if success and not self.create_sample_config():
            success = False
        
        # Setup environment variables
        if success and not self.setup_environment_variables():
            success = False
        
        # Validate installation
        if success and not self.validate_installation():
            success = False
        
        # Run basic tests
        if success and not skip_tests and not self.run_basic_tests():
            success = False
        
        # Create quick start guide
        if success and not self.create_quick_start_guide():
            success = False
        
        # Save setup log
        self.save_setup_log()
        
        if success:
            self.log("🎉 Environment setup completed successfully!")
            self.log("Check QUICK_START.md for next steps.")
        else:
            self.log("❌ Environment setup failed. Check setup_log.txt for details.", 'ERROR')
        
        return success


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Setup UK Solar Prediction environment')
    
    parser.add_argument('--dev', action='store_true',
                       help='Install development dependencies')
    parser.add_argument('--skip-tests', action='store_true',
                       help='Skip basic functionality tests')
    parser.add_argument('--project-root', type=str,
                       help='Project root directory path')
    
    args = parser.parse_args()
    
    # Create setup instance
    setup = EnvironmentSetup(args.project_root)
    
    try:
        # Run setup
        success = setup.run_full_setup(
            dev_mode=args.dev,
            skip_tests=args.skip_tests
        )
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"Setup failed with error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

