"""
Main runner script for tests and benchmarks.
"""
import subprocess
import sys
import os


def run_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Basic KVStore Tests")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_kvstore.py", "-v", "--tb=short", "-x"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print("Basic tests failed!")
        return False
    
    print("\n" + "="*60)
    print("Running Cluster Tests")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "test_cluster.py", "-v", "--tb=short", "-x"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    if result.returncode != 0:
        print("Cluster tests failed!")
        return False
    
    return True


def run_benchmarks():
    """Run benchmarks."""
    print("\n" + "="*60)
    print("Running Benchmarks")
    print("="*60)
    
    result = subprocess.run(
        [sys.executable, "benchmark.py"],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    
    return result.returncode == 0


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KVStore Test Runner")
    parser.add_argument("--tests", action="store_true", help="Run tests")
    parser.add_argument("--benchmarks", action="store_true", help="Run benchmarks")
    parser.add_argument("--all", action="store_true", help="Run tests and benchmarks")
    
    args = parser.parse_args()
    
    if not any([args.tests, args.benchmarks, args.all]):
        args.all = True
    
    success = True
    
    if args.tests or args.all:
        success = run_tests() and success
    
    if args.benchmarks or args.all:
        success = run_benchmarks() and success
    
    if success:
        print("\n" + "="*60)
        print("All operations completed successfully!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Some operations failed!")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()
