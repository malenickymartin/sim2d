import argparse
import torch
from torch_geometric.data import HeteroData
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def visualize_hetero_graph(file_path):
    # 1. Load the data
    try:
        data = torch.load(file_path, weights_only=False)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not isinstance(data, HeteroData):
        print("Error: Loaded data is not a HeteroData object.")
        return

    print(f"Loaded graph from {file_path}")

    # 2. Setup NetworkX Graph and Color Maps
    G = nx.MultiDiGraph()

    node_color_map = {
        "world": "#88FF00",
        "object": "#006EFF",
        "floor": "#32532E",
    }
    default_node_color = "#C0C0C0"

    # Colors for specific interaction types
    CONTACT_OBJ_OBJ = "#FF0000"
    CONTACT_FLOOR_OBJ = "#FC802E"

    edge_color_map = {
        "w2o": "#006EFF",
        "w2f": "#32532E",
    }
    default_edge_color = "#000000"

    # 3. Create Global IDs and Build Graph
    global_node_index = 0
    node_mapping = {}
    global_labels = {}

    # Add Nodes
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        for local_idx in range(num_nodes):
            gid = global_node_index
            node_mapping[(node_type, local_idx)] = gid

            label = f"{node_type.capitalize()[0]}{local_idx}"

            G.add_node(gid, type=node_type, label=label)
            global_labels[gid] = label
            global_node_index += 1

    # Add Edges
    for edge_type_tuple in data.edge_types:
        src_type, relation, dst_type = edge_type_tuple
        edge_index = data[edge_type_tuple].edge_index

        if edge_index.numel() == 0:
            continue

        for i in range(edge_index.shape[1]):
            src_local = edge_index[0, i].item()
            dst_local = edge_index[1, i].item()
            src_global = node_mapping.get((src_type, src_local))
            dst_global = node_mapping.get((dst_type, dst_local))

            if src_global is not None and dst_global is not None:
                # Store relation type on the edge
                G.add_edge(src_global, dst_global, type=relation)

    # 4. Visualization Layout
    plt.figure(figsize=(10, 10))  # Increased figure size for more space

    # --- ADJUSTED LAYOUT FOR DISPERSION ---
    # k: Optimal distance between nodes. Higher = more spread out.
    # iterations: More steps allow the physics sim to find a better global optimum.
    # seed: Fixes the random state so the layout is consistent.
    pos = nx.spring_layout(G, k=5.5, iterations=50, seed=42)

    NODE_SIZE = 1000
    ARROW_SIZE = 25

    # Draw Nodes
    seen_node_types = set()
    for node_type in data.node_types:
        nodelist = [n for n, attr in G.nodes(data=True) if attr["type"] == node_type]
        if not nodelist:
            continue

        seen_node_types.add(node_type)
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=nodelist,
            node_color=node_color_map.get(node_type, default_node_color),
            node_size=NODE_SIZE,
            alpha=1.0,
            edgecolors="black",
            linewidths=1.5,
        )

    # Group Edges by Style/Color for Batch Drawing
    edge_batches = {}

    for u, v, attr in G.edges(data=True):
        relation = attr.get("type")

        # Determine Color and Group Key
        if relation == "contact":
            src_type = G.nodes[u]["type"]
            dst_type = G.nodes[v]["type"]

            if src_type == "floor" or dst_type == "floor":
                group_name = "contact_floor"
                color = CONTACT_FLOOR_OBJ
            else:
                group_name = "contact_object"
                color = CONTACT_OBJ_OBJ
        else:
            group_name = relation
            color = edge_color_map.get(relation, default_edge_color)

        # Determine Style
        style = "dashed" if "w2" in relation else "solid"
        width = 2.0

        # Initialize list for this group
        if group_name not in edge_batches:
            edge_batches[group_name] = {"color": color, "style": style, "width": width, "edges": []}

        edge_batches[group_name]["edges"].append((u, v))

    # Draw Edge Batches
    for group_name, batch in edge_batches.items():
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=batch["edges"],
            edge_color=batch["color"],
            style=batch["style"],
            width=batch["width"],
            arrowsize=ARROW_SIZE,
            node_size=NODE_SIZE,
            connectionstyle="arc3,rad=0.1",
        )

    # Draw Labels
    nx.draw_networkx_labels(G, pos, labels=global_labels, font_size=10, font_weight="bold")

    # Legend
    legend_patches = []
    for nt in sorted(list(seen_node_types)):
        color = node_color_map.get(nt, default_node_color)
        legend_patches.append(mpatches.Patch(color=color, label=f"Node: {nt}"))

    for group_name, batch in edge_batches.items():
        label_text = group_name.replace("_", " ").title()
        legend_patches.append(mpatches.Patch(color=batch["color"], label=f"Edge: {label_text}"))

    plt.legend(handles=legend_patches, loc="upper right")
    plt.title(f"Graph Visualization: {file_path}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    path = "data/gnn_datasets/test_dataset/processed/data_0_75.pt"
    visualize_hetero_graph(path)
