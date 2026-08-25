"""Regenerate Ridgecrest Fig. 6 panels (topo background)."""
import matplotlib

matplotlib.use("Agg")

from figures import fig03_ridgecrest_models

if __name__ == "__main__":
    fig03_ridgecrest_models()
    print("fig03 ridgecrest models done")
