#!/bin/bash
# Script to run the full cleanup phase analysis

echo "Starting cleanup phase analysis..."
echo "This will take approximately 30-60 minutes on a GPU"
echo ""

# Make sure we're in the right directory
cd /workspace/grokking

# Run the analysis
python run_cleanup_analysis.py

echo ""
echo "Analysis complete! Check the following files:"
echo "- cleanup_phase_analysis_complete.png (main visualization)"
echo "- cleanup_metrics_final.pkl (raw data)"
echo "- cleanup_metrics_checkpoint_*.pkl (intermediate checkpoints)"