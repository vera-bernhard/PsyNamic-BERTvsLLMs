import matplotlib.pyplot as plt
import matplotlib.patches as Patch
import numpy as np
from matplotlib.legend_handler import HandlerPatch
from matplotlib.colors import LinearSegmentedColormap

class GradientLegendHandler(HandlerPatch):
    """Custom handler to display a gradient in a legend entry."""
    def __init__(self, colors, **kwargs):
        super().__init__()
        self.colors = colors

    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        # Make a custom colormap from the provided colors
        cmap = LinearSegmentedColormap.from_list("custom_cmap", self.colors)

        # Fake image of a horizontal gradient
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        im = plt.imshow(
            gradient,
            aspect="auto",
            cmap=cmap,
            extent=[xdescent, xdescent + width, ydescent, ydescent + height],
            transform=trans
        )
        return [im]

class GradientLegend:
    """Custom legend builder for multiple gradient entries."""
    def __init__(self, entries):
        """
        entries: list of tuples like
          [
            ("OpenAI models", ["#17a583", "#48c9b0"]),
            ("Llama-2 models", ["#33a02c", "#b2df8a", "#a6cee3", "#1f78b4"])
          ]
        """
        self.entries = entries

    def add_to_axis(self, ax=None, loc="upper right"):
        if ax is None:
            ax = plt.gca()

        # Dummy handles for each entry
        handles = [Patch.Rectangle((0,0),1,1) for _ in self.entries]
        labels = [label for label, _ in self.entries]
        handler_map = {
            h: GradientLegendHandler(colors)
            for h, (_, colors) in zip(handles, self.entries)
        }

        ax.legend(handles, labels, handler_map=handler_map, loc=loc)

# ------------------- Example usage -------------------

if __name__ == "__main__":
    # Fake plot
    x = np.linspace(0, 10, 100)
    plt.plot(x, np.sin(x), color="#17a583")
    plt.plot(x, np.cos(x), color="#33a02c")

    legend_data = [
        ("OpenAI models", ["#17a583", "#48c9b0"]),
        ("Llama-2-70B models", ["#33a02c", "#b2df8a"]),
        ("Llama-2-13B models", ["#1f78b4", "#a6cee3"]),
        ("LLaMA3-8B models", ["#e31a1c", "#fb9a99"]),
        ("LLaMA3.1-8B models", ["#ff7f00", "#fdbf6f"]),
    ]

    GradientLegend(legend_data).add_to_axis(loc="upper left")

    plt.show()
