"""WaferPulse application package.

The package follows a small-agent/crew structure:

* ``agents`` own one analytical responsibility;
* ``crews`` orchestrate agent execution;
* ``tools`` contain deterministic data/ML functions;
* ``core`` contains shared contracts and data-lane policy;
* ``dashboard`` contains Streamlit presentation code only.
"""

__version__ = "1.3.0"
