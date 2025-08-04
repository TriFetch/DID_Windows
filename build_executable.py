#!/usr/bin/env python3
"""
Build script for creating DID.py executable using PyInstaller
"""

import os
import subprocess
import sys

def build_executable():
    """Build the DID.py executable using PyInstaller"""
    
    # Check if PyInstaller is installed
    try:
        subprocess.run(["pyinstaller", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Build command
    build_cmd = [
        "pyinstaller",
        "--onefile",                    # Create single executable
        "--name=DID_Tool",             # Name of executable
        "--distpath=dist",             # Output directory
        "--workpath=build",            # Build directory
        "--specpath=build",            # Spec file directory
        "--clean",                     # Clean build cache
        "--noconfirm",                 # Don't ask for confirmation
        "DID.py"
    ]
    
    print("Building executable...")
    print(f"Command: {' '.join(build_cmd)}")
    
    try:
        result = subprocess.run(build_cmd, check=True, capture_output=True, text=True)
        print("Build successful!")
        print(f"Executable created: dist/DID_Tool")
        if os.name == 'nt':  # Windows
            print(f"Executable created: dist/DID_Tool.exe")
    except subprocess.CalledProcessError as e:
        print(f"Build failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    
    return True

if __name__ == "__main__":
    success = build_executable()
    if not success:
        sys.exit(1)