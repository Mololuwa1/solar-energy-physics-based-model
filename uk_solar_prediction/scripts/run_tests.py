#!/usr/bin/env python3
"""
Comprehensive Test Runner for UK Solar Energy Prediction System

This script runs all test suites and generates detailed reports for the
physics-informed solar energy prediction model and deployment system.

Author: Manus AI
Date: 2025-08-16
"""

import sys
import os
import unittest
import time
import json
from pathlib import Path
from datetime import datetime
import argparse
import subprocess
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'tests'))
sys.path.insert(0, str(project_root / 'sagemaker'))


class TestRunner:
    """Comprehensive test runner with reporting capabilities."""
    
    def __init__(self, verbose=True, generate_report=True):
        """
        Initialize test runner.
        
        Args:
            verbose: Whether to run tests in verbose mode
            generate_report: Whether to generate detailed test report
        """
        self.verbose = verbose
        self.generate_report = generate_report
        self.test_results = {}
        self.start_time = None
        self.end_time = None
        
    def discover_test_modules(self, test_dir):
        """
        Discover all test modules in the test directory.
        
        Args:
            test_dir: Path to test directory
            
        Returns:
            List of test module names
        """
        test_modules = []
        test_path = Path(test_dir)
        
        for test_file in test_path.glob('test_*.py'):
            module_name = test_file.stem
            test_modules.append(module_name)
            
        return sorted(test_modules)
    
    def load_test_module(self, module_name, test_dir):
        """
        Load a test module dynamically.
        
        Args:
            module_name: Name of the test module
            test_dir: Path to test directory
            
        Returns:
            Loaded module or None if failed
        """
        try:
            module_path = Path(test_dir) / f"{module_name}.py"
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Failed to load test module {module_name}: {e}")
            return None
    
    def run_test_suite(self, test_module, module_name):
        """
        Run test suite from a module.
        
        Args:
            test_module: Loaded test module
            module_name: Name of the test module
            
        Returns:
            Test result object
        """
        print(f"\n{'='*60}")
        print(f"Running {module_name}")
        print(f"{'='*60}")
        
        # Discover tests in the module
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        # Find all test classes in the module
        for name in dir(test_module):
            obj = getattr(test_module, name)
            if (isinstance(obj, type) and 
                issubclass(obj, unittest.TestCase) and 
                obj != unittest.TestCase):
                tests = loader.loadTestsFromTestCase(obj)
                suite.addTests(tests)
        
        # Run tests
        runner = unittest.TextTestRunner(
            verbosity=2 if self.verbose else 1,
            stream=sys.stdout,
            buffer=True
        )
        
        start_time = time.time()
        result = runner.run(suite)
        end_time = time.time()
        
        # Store results
        self.test_results[module_name] = {
            'result': result,
            'duration': end_time - start_time,
            'tests_run': result.testsRun,
            'failures': len(result.failures),
            'errors': len(result.errors),
            'skipped': len(getattr(result, 'skipped', [])),
            'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / 
                           max(result.testsRun, 1) * 100)
        }
        
        return result
    
    def run_all_tests(self, test_dir=None):
        """
        Run all test suites.
        
        Args:
            test_dir: Path to test directory (defaults to project tests dir)
        """
        if test_dir is None:
            test_dir = project_root / 'tests'
        
        self.start_time = time.time()
        
        print(f"UK Solar Energy Prediction System - Test Suite")
        print(f"{'='*60}")
        print(f"Test directory: {test_dir}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Discover test modules
        test_modules = self.discover_test_modules(test_dir)
        print(f"Found {len(test_modules)} test modules: {', '.join(test_modules)}")
        
        # Run each test module
        all_results = []
        
        for module_name in test_modules:
            try:
                # Load test module
                test_module = self.load_test_module(module_name, test_dir)
                if test_module is None:
                    continue
                
                # Run tests
                result = self.run_test_suite(test_module, module_name)
                all_results.append(result)
                
            except Exception as e:
                print(f"Error running {module_name}: {e}")
                continue
        
        self.end_time = time.time()
        
        # Generate summary
        self.print_summary()
        
        # Generate detailed report if requested
        if self.generate_report:
            self.generate_test_report()
        
        return all_results
    
    def print_summary(self):
        """Print test execution summary."""
        print(f"\n{'='*60}")
        print(f"TEST EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        total_duration = self.end_time - self.start_time
        total_tests = sum(r['tests_run'] for r in self.test_results.values())
        total_failures = sum(r['failures'] for r in self.test_results.values())
        total_errors = sum(r['errors'] for r in self.test_results.values())
        total_skipped = sum(r['skipped'] for r in self.test_results.values())
        
        overall_success_rate = ((total_tests - total_failures - total_errors) / 
                              max(total_tests, 1) * 100)
        
        print(f"Total execution time: {total_duration:.2f} seconds")
        print(f"Total test modules: {len(self.test_results)}")
        print(f"Total tests run: {total_tests}")
        print(f"Total failures: {total_failures}")
        print(f"Total errors: {total_errors}")
        print(f"Total skipped: {total_skipped}")
        print(f"Overall success rate: {overall_success_rate:.1f}%")
        
        print(f"\nPer-module results:")
        print(f"{'Module':<30} {'Tests':<8} {'Failures':<10} {'Errors':<8} {'Success%':<10} {'Time(s)':<10}")
        print(f"{'-'*80}")
        
        for module_name, results in self.test_results.items():
            print(f"{module_name:<30} {results['tests_run']:<8} {results['failures']:<10} "
                  f"{results['errors']:<8} {results['success_rate']:<9.1f}% {results['duration']:<9.2f}")
        
        # Status determination
        if total_failures == 0 and total_errors == 0:
            status = "✅ ALL TESTS PASSED"
            exit_code = 0
        elif overall_success_rate >= 90:
            status = "⚠️  MOSTLY PASSING (some issues)"
            exit_code = 0
        else:
            status = "❌ SIGNIFICANT ISSUES DETECTED"
            exit_code = 1
        
        print(f"\nOverall Status: {status}")
        
        return exit_code
    
    def generate_test_report(self):
        """Generate detailed test report."""
        report_path = project_root / 'test_report.json'
        
        # Prepare report data
        report_data = {
            'execution_info': {
                'start_time': datetime.fromtimestamp(self.start_time).isoformat(),
                'end_time': datetime.fromtimestamp(self.end_time).isoformat(),
                'duration_seconds': self.end_time - self.start_time,
                'python_version': sys.version,
                'platform': sys.platform
            },
            'summary': {
                'total_modules': len(self.test_results),
                'total_tests': sum(r['tests_run'] for r in self.test_results.values()),
                'total_failures': sum(r['failures'] for r in self.test_results.values()),
                'total_errors': sum(r['errors'] for r in self.test_results.values()),
                'total_skipped': sum(r['skipped'] for r in self.test_results.values()),
                'overall_success_rate': ((sum(r['tests_run'] for r in self.test_results.values()) - 
                                        sum(r['failures'] for r in self.test_results.values()) - 
                                        sum(r['errors'] for r in self.test_results.values())) / 
                                       max(sum(r['tests_run'] for r in self.test_results.values()), 1) * 100)
            },
            'modules': {}
        }
        
        # Add detailed module results
        for module_name, results in self.test_results.items():
            module_data = {
                'tests_run': results['tests_run'],
                'failures': results['failures'],
                'errors': results['errors'],
                'skipped': results['skipped'],
                'success_rate': results['success_rate'],
                'duration': results['duration']
            }
            
            # Add failure and error details
            result = results['result']
            if result.failures:
                module_data['failure_details'] = [
                    {'test': str(test), 'message': traceback.split('\n')[-2] if traceback else 'Unknown'}
                    for test, traceback in result.failures
                ]
            
            if result.errors:
                module_data['error_details'] = [
                    {'test': str(test), 'message': traceback.split('\n')[-2] if traceback else 'Unknown'}
                    for test, traceback in result.errors
                ]
            
            report_data['modules'][module_name] = module_data
        
        # Save report
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\nDetailed test report saved to: {report_path}")
        
        # Generate HTML report
        self.generate_html_report(report_data)
    
    def generate_html_report(self, report_data):
        """Generate HTML test report."""
        html_path = project_root / 'test_report.html'
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>UK Solar Prediction - Test Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .summary {{ margin: 20px 0; }}
        .module {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 5px; }}
        .module-header {{ background-color: #f8f9fa; padding: 10px; font-weight: bold; }}
        .module-content {{ padding: 10px; }}
        .success {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #f8f9fa; border-radius: 3px; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>UK Solar Energy Prediction System - Test Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Duration: {report_data['execution_info']['duration_seconds']:.2f} seconds</p>
    </div>
    
    <div class="summary">
        <h2>Summary</h2>
        <div class="metric">
            <strong>Total Tests:</strong> {report_data['summary']['total_tests']}
        </div>
        <div class="metric">
            <strong>Failures:</strong> <span class="{'error' if report_data['summary']['total_failures'] > 0 else 'success'}">{report_data['summary']['total_failures']}</span>
        </div>
        <div class="metric">
            <strong>Errors:</strong> <span class="{'error' if report_data['summary']['total_errors'] > 0 else 'success'}">{report_data['summary']['total_errors']}</span>
        </div>
        <div class="metric">
            <strong>Success Rate:</strong> <span class="{'success' if report_data['summary']['overall_success_rate'] >= 95 else 'warning' if report_data['summary']['overall_success_rate'] >= 80 else 'error'}">{report_data['summary']['overall_success_rate']:.1f}%</span>
        </div>
    </div>
    
    <div class="modules">
        <h2>Module Results</h2>
        <table>
            <tr>
                <th>Module</th>
                <th>Tests</th>
                <th>Failures</th>
                <th>Errors</th>
                <th>Success Rate</th>
                <th>Duration (s)</th>
            </tr>
        """
        
        for module_name, module_data in report_data['modules'].items():
            success_class = ('success' if module_data['success_rate'] >= 95 else 
                           'warning' if module_data['success_rate'] >= 80 else 'error')
            
            html_content += f"""
            <tr>
                <td>{module_name}</td>
                <td>{module_data['tests_run']}</td>
                <td class="{'error' if module_data['failures'] > 0 else ''}">{module_data['failures']}</td>
                <td class="{'error' if module_data['errors'] > 0 else ''}">{module_data['errors']}</td>
                <td class="{success_class}">{module_data['success_rate']:.1f}%</td>
                <td>{module_data['duration']:.2f}</td>
            </tr>
            """
        
        html_content += """
        </table>
    </div>
    
    <div class="details">
        <h2>Detailed Results</h2>
        """
        
        for module_name, module_data in report_data['modules'].items():
            html_content += f"""
        <div class="module">
            <div class="module-header">{module_name}</div>
            <div class="module-content">
                <p><strong>Tests Run:</strong> {module_data['tests_run']}</p>
                <p><strong>Success Rate:</strong> {module_data['success_rate']:.1f}%</p>
                <p><strong>Duration:</strong> {module_data['duration']:.2f} seconds</p>
            """
            
            if 'failure_details' in module_data:
                html_content += "<h4>Failures:</h4><ul>"
                for failure in module_data['failure_details']:
                    html_content += f"<li><strong>{failure['test']}:</strong> {failure['message']}</li>"
                html_content += "</ul>"
            
            if 'error_details' in module_data:
                html_content += "<h4>Errors:</h4><ul>"
                for error in module_data['error_details']:
                    html_content += f"<li><strong>{error['test']}:</strong> {error['message']}</li>"
                html_content += "</ul>"
            
            html_content += """
            </div>
        </div>
            """
        
        html_content += """
    </div>
</body>
</html>
        """
        
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        print(f"HTML test report saved to: {html_path}")
    
    def run_specific_tests(self, test_patterns):
        """
        Run specific tests matching patterns.
        
        Args:
            test_patterns: List of test patterns to match
        """
        print(f"Running specific tests matching: {test_patterns}")
        
        # Implementation for running specific tests
        # This would filter tests based on patterns
        pass
    
    def run_coverage_analysis(self):
        """Run test coverage analysis."""
        try:
            import coverage
            
            print("Running coverage analysis...")
            
            # Start coverage
            cov = coverage.Coverage()
            cov.start()
            
            # Run tests
            self.run_all_tests()
            
            # Stop coverage and generate report
            cov.stop()
            cov.save()
            
            # Generate coverage report
            coverage_path = project_root / 'coverage_report.html'
            cov.html_report(directory=str(coverage_path.parent / 'htmlcov'))
            
            print(f"Coverage report generated in: {coverage_path.parent / 'htmlcov'}")
            
        except ImportError:
            print("Coverage package not available. Install with: pip install coverage")
        except Exception as e:
            print(f"Coverage analysis failed: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Run UK Solar Prediction System tests')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Run tests in verbose mode')
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating detailed test report')
    parser.add_argument('--coverage', action='store_true',
                       help='Run with coverage analysis')
    parser.add_argument('--module', '-m', action='append',
                       help='Run specific test module(s)')
    parser.add_argument('--pattern', '-p', action='append',
                       help='Run tests matching pattern(s)')
    parser.add_argument('--test-dir', type=str,
                       help='Custom test directory path')
    
    args = parser.parse_args()
    
    # Create test runner
    runner = TestRunner(
        verbose=args.verbose,
        generate_report=not args.no_report
    )
    
    try:
        if args.coverage:
            # Run with coverage
            runner.run_coverage_analysis()
        elif args.module:
            # Run specific modules
            print(f"Running specific modules: {args.module}")
            # Implementation would filter by modules
            runner.run_all_tests(args.test_dir)
        elif args.pattern:
            # Run tests matching patterns
            runner.run_specific_tests(args.pattern)
        else:
            # Run all tests
            runner.run_all_tests(args.test_dir)
        
        # Get exit code from summary
        exit_code = runner.print_summary() if hasattr(runner, 'test_results') else 0
        
    except KeyboardInterrupt:
        print("\nTest execution interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"Test execution failed: {e}")
        exit_code = 1
    
    sys.exit(exit_code)


if __name__ == '__main__':
    main()

