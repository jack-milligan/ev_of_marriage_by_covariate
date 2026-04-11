import os
import matplotlib.pyplot as plt


def save_visual(fig: plt.Figure, filename: str, folder: str = "visuals") -> None:
    """Save matplotlib Figure to the visuals/ directory, creating it if needed."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved visualization: {path}")