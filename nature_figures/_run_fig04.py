"""Regenerate Ridgecrest Fig. 7 timing panels (continuous terrain)."""
import matplotlib

matplotlib.use("Agg")

from figures import fig04_ridgecrest_timing

if __name__ == "__main__":
    fig04_ridgecrest_timing()
    print("fig04 ridgecrest timing done")
