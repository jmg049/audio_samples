#!/bin/bash
echo "Generating integrated stub file..."

# Generate single integrated stub with all submodules
python generate_integrated_stub.py

# Create py.typed marker file
touch py.typed

echo "Generated stub files:"
ls -la audio_samples.pyi py.typed
