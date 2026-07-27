"""Utility functions for visualization"""

from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.padding import Padding
from ..richlog.core import INDENT

import re

from ..richlog import StepHandle


def save_plot(step: StepHandle, fig, experiment_name: str, filename: str,
              save_dir: str = 'results', run_timestamp: Optional[str] = None) -> None:
    """Save plot as PNG file in organized directory structure
    
    Args:
        step: open StepHandle to log the save confirmation into.
        fig: Matplotlib figure to save
        experiment_name: Name of the experiment
        filename: Filename for the plot (e.g., 'training_history.png')
        save_dir: Base directory to save plots in
        run_timestamp: Optional timestamp for the main run (YYYY-MM-DD_HHhMM format)
                      If None, creates a new timestamp
    """
    # Create results directory if it doesn't exist
    results_path = Path(save_dir)
    results_path.mkdir(exist_ok=True)
    
    # Use provided timestamp or create new one
    if run_timestamp:
        date_str = run_timestamp
    else:
        date_str = datetime.now().strftime('%Y-%m-%d_%Hh%M')
    
    date_dir = results_path / date_str
    date_dir.mkdir(exist_ok=True)
    
    # Create experiment-specific directory
    experiment_dir = date_dir / re.sub(r"( - |\. | )", "_", experiment_name.lower())
    experiment_dir.mkdir(exist_ok=True)
    
    # Save the plot
    filepath = experiment_dir / filename
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    if "prediction" in filename:
        message = f"prediction plot showing model outputs vs actual values:\n    {filepath}"
    else: 
        message = f"saved plot:\n    {filepath}"

    step.block(Padding(message, (0, 0, 0, len(INDENT) * 3), style="dim"))
