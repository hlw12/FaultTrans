"""Regenerate Chi-Chi Fig. 8 panels (topo background, Chi model only)."""
import matplotlib

matplotlib.use("Agg")

from figures import fig05_chichi

if __name__ == "__main__":
    fig05_chichi()
    print("fig05 chichi done")
