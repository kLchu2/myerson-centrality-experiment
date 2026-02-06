import sys
from pathlib import Path

import networkx as nx

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from centrality.revised_myerson import all_revised_myerson_centralities  # noqa: E402


def main():
    # Define the graph from the image
    G = nx.Graph()
    G.add_nodes_from(["a", "b", "c", "d", "e", "f", "g"])
    G.add_edges_from(
        [
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "e"),
            ("e", "c"),  # Triangle 1
            ("c", "f"),
            ("f", "g"),
            ("g", "c"),  # Triangle 2
        ]
    )

    print("Graph Info:")
    print(f"Nodes: {G.nodes()}")
    print(f"Edges: {G.edges()}")

    # Calculate Revised Myerson Centrality
    # Using default influence constant r=1.0 as per library defaults
    # If a different r is needed, it can be passed as an argument.
    centralities = all_revised_myerson_centralities(G, verbose=True, use_cache=False)

    print("\nRevised Myerson Centrality Results:")
    for node in sorted(centralities.keys()):
        print(f"Node {node}: {centralities[node]:.6f}")


if __name__ == "__main__":
    main()
