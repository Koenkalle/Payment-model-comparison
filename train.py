"""Compatibility entry point for the offline model trainers."""
from training.train import *  # Re-export the existing Python training API.

if __name__ == "__main__":
    main()
