from typing import List, Any, Tuple, Optional
import os
import os.path as osp
import sys
import random

import cv2
from colorama import init, Fore, Style

import matplotlib.pyplot as plt
from matplotlib.colors import hex2color
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib import colors
import networkx as nx
import numpy as np
import networkx as nx

init()
# Create a custom exception hook
def colored_exception_hook(exc_type, exc_value, tb):
    # Format the exception and traceback
    import traceback
    exception_text = ''.join(traceback.format_exception(exc_type, exc_value, tb))
    
    if 'CUDARuntime' in exception_text:
        print(Fore.RED + exception_text + Style.RESET_ALL, file=sys.stderr)
    else:
        print(Fore.BLUE + exception_text + Style.RESET_ALL, file=sys.stderr)

# Set the custom exception hook
sys.excepthook = colored_exception_hook


# Keep the existing get_edge_colors function as it is
def get_edge_colors(graph):
    edge_colors = []
    edge_types = set()
    color_map = {}
    for _, _, data in graph.edges(data=True):
        edge_type = str(data.get('param_type', 'Unknown'))
        if edge_type == 'Unknown':
            edge_type = 'Hierarchy'
        edge_types.add(edge_type)
        if edge_type not in color_map:
            color_map[edge_type] = get_color_for_type(edge_type)
        edge_colors.append(color_map[edge_type])
    return edge_colors, list(edge_types), color_map

# Make sure this function is defined
def get_color_for_type(type_str):
    import hashlib
    hash_object = hashlib.md5(type_str.encode())
    hash_hex = hash_object.hexdigest()
    r = int(hash_hex[:2], 16) / 255.0
    g = int(hash_hex[2:4], 16) / 255.0
    b = int(hash_hex[4:6], 16) / 255.0
    return (r, g, b)

def visualize_dynamic_graph(graph_manager, title: str = "Transformation Graph", figsize: Tuple[int, int] = (24, 20), highlight_nodes: Optional[List[Any]] = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Create a main axes for the graph, using more of the figure space
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96])
    
    # Separate nodes by type
    input_nodes = [n for n, d in graph_manager.graph.nodes(data=True) if d['type'] == 'input']
    output_nodes = [n for n, d in graph_manager.graph.nodes(data=True) if d['type'] in ['output', 'solution']]
    primitive_nodes = [n for n, d in graph_manager.graph.nodes(data=True) if d['type'] == 'primitive']
    
    # Create a layout for the primitive nodes
    main_graph = graph_manager.graph.subgraph(primitive_nodes)
    pos = nx.spring_layout(main_graph, k=0.9, iterations=50)  # Increased k for more spread
    
    # Scale and shift the primitive node positions to use more space
    x_values, y_values = zip(*pos.values())
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    scale_factor = 2.5  # Increased scale factor
    for node in pos:
        pos[node] = (
            (pos[node][0] - x_min) / (x_max - x_min) * scale_factor - 0.5,
            (pos[node][1] - y_min) / (y_max - y_min) * scale_factor - 0.5
        )
    
    # Position dynamic nodes in separate pools
    dynamic_pos = {}
    num_pairs = len(graph_manager.pair_dynamic_nodes)
    pool_width, pool_height = 1.2, 0.5  # Adjusted pool size
    pool_spacing = 0.1  # Reduced spacing
    
    # Calculate the leftmost x-coordinate for the pools
    leftmost_x = -2.0  # Moved pools further left

    for i, (pair_index, pair_nodes) in enumerate(graph_manager.pair_dynamic_nodes.items()):
        pool_center_x = leftmost_x
        pool_center_y = 1.2 - (i * (pool_height + pool_spacing))
        
        for node_type, nodes in pair_nodes.items():
            for node in nodes:
                x = pool_center_x + random.uniform(-pool_width/2, pool_width/2)
                y = pool_center_y + random.uniform(-pool_height/2, pool_height/2)
                dynamic_pos[node.name] = (x, y)
        
        # Position input and output nodes for this pair
        input_node = f'train_input_{pair_index}'
        output_node = f'train_output_{pair_index}'
        pos[input_node] = (pool_center_x - pool_width/2 - 0.2, pool_center_y + 0.1)
        pos[output_node] = (pool_center_x - pool_width/2 - 0.2, pool_center_y - 0.1)
    
    # Combine positions
    pos.update(dynamic_pos)
    
    # Ensure all nodes have positions
    for node in graph_manager.graph.nodes():
        if node not in pos:
            pos[node] = (random.uniform(-2, 2), random.uniform(-2, 2))
    
    # Prepare node colors, sizes, and alphas
    node_colors = []
    node_sizes = []
    node_alphas = []
    for node in graph_manager.graph.nodes():
        node_type = graph_manager.graph.nodes[node].get('type', '')
        if node_type == 'input':
            node_colors.append('lightblue')
            node_sizes.append(150)
        elif node_type in ['output', 'solution']:
            node_colors.append('lightgreen')
            node_sizes.append(150)
        elif node_type == 'primitive':
            node_colors.append('orange')
            node_sizes.append(250)  # Slightly reduced size
        elif node_type == 'dynamic':
            node_colors.append('yellow')
            node_sizes.append(80)  # Reduced size for dynamic nodes
        else:
            node_colors.append('gray')
            node_sizes.append(80)
        
        if highlight_nodes is None or node in highlight_nodes:
            node_alphas.append(0.8)
        else:
            node_alphas.append(0.5)
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(graph_manager.graph, pos, 
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=node_alphas,
                           ax=ax)
                           
    # Prepare edge colors and alphas
    edge_colors = []
    edge_alphas = []
    param_types = set()
    for u, v, key, data in graph_manager.graph.edges(keys=True, data=True):
        if key == 'dependency':
            param_type = data.get('param_type', 'unknown')
            param_types.add(param_type)
            edge_colors.append(param_type)
        elif key == 'primitive':
            edge_colors.append('red')
        elif key == 'features':
            edge_colors.append('blue')
        elif key == 'input':
            edge_colors.append('green')
        elif key == 'transformation':
            edge_colors.append('aqua')
        else:
            edge_colors.append('gray')
        
        if highlight_nodes and (u in highlight_nodes or v in highlight_nodes):
            edge_alphas.append(1.0)
        else:
            edge_alphas.append(0.3)
    
    # Create a color map for param types
    param_types = list(param_types)
    n_colors = len(param_types)
    cmap = LinearSegmentedColormap.from_list("param_types", plt.cm.rainbow(np.linspace(0, 1, n_colors)))
    color_dict = {param_type: cmap(i/n_colors) for i, param_type in enumerate(param_types)}
    color_dict.update({
        'red': 'red',
        'blue': 'blue',
        'green': 'green',
        'aqua': 'aqua',
        'gray': 'gray'
    })
    
    edge_colors = [color_dict[color] for color in edge_colors]
    
    # Draw edges
    edges = nx.draw_networkx_edges(graph_manager.graph, pos, 
                           edge_color=edge_colors,
                           width=0.5,
                           alpha=edge_alphas,
                           ax=ax)
    
    # Add labels with a slight offset for better readability
    label_pos = {k: (v[0], v[1]+0.02) for k, v in pos.items()}
    nx.draw_networkx_labels(graph_manager.graph, label_pos, font_size=6, font_weight='bold', font_family='sans-serif', ax=ax)
    
    # Draw rectangles around the dynamic nodes pools
    for i in range(num_pairs):
        pool_center_y = 1.0 - (i * (pool_height + pool_spacing))
        pool_rect = plt.Rectangle((leftmost_x - pool_width/2, pool_center_y - pool_height/2),
                                  pool_width, pool_height,
                                  fill=False, edgecolor='gray', linestyle='--', linewidth=1)
        ax.add_patch(pool_rect)
        ax.text(leftmost_x, pool_center_y + pool_height/2 + 0.05, f'Pair {i} Dynamic Nodes',
                horizontalalignment='center', fontsize=10, fontweight='bold')

    # Add a title
    ax.set_title(title, fontsize=24, fontweight='bold')
    
    # Remove axis
    ax.axis('off')
    
    # Add a legend for node and edge types
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Input', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Output/Solution', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Primitive', markerfacecolor='orange', markersize=12),
        plt.Line2D([0], [0], marker='o', color='w', label='Dynamic', markerfacecolor='yellow', markersize=8),
        plt.Line2D([0], [0], color='red', lw=1, label='Primitive Application'),
        plt.Line2D([0], [0], color='blue', lw=1, label='Feature Comparison'),
        plt.Line2D([0], [0], color='green', lw=1, label='Input Argument'),
        plt.Line2D([0], [0], color='aqua', lw=1, label='Transformation')
    ]
    
    # Add legend elements for param types
    for param_type in param_types:
        legend_elements.append(plt.Line2D([0], [0], color=color_dict[param_type], lw=1, label=f'Param: {param_type}'))
    
    # Add a single title with increased font size
    ax.set_title(title, fontsize=24, fontweight='bold', pad=20)
    
    # Adjust the legend position and font size
    legend_ax = fig.add_axes([0.88, 0.08, 0.1, 0.87])
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, title="Node and Edge Types", 
                              loc="center left", fontsize='small')
    legend.get_title().set_fontsize('medium')
    
    plt.tight_layout()
    plt.show()

def visualize_graph(graph: nx.Graph, title: str = "Knowledge Graph", figsize: Tuple[int, int] = (20, 20), highlight_nodes: Optional[List[Any]] = None):
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Create a main axes for the graph
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9])  # [left, bottom, width, height]
    
    # Use Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(graph)
    
    # Add problem node to layout if it's in highlight_nodes but not in graph
    if highlight_nodes:
        for node in highlight_nodes:
            if node not in pos:
                # Find a random position near the center for the new node
                center_x = sum(x for x, y in pos.values()) / len(pos)
                center_y = sum(y for x, y in pos.values()) / len(pos)
                pos[node] = (center_x + random.uniform(-0.1, 0.1), 
                             center_y + random.uniform(-0.1, 0.1))
    
    # Calculate node degrees for color mapping
    degrees = dict(graph.degree())
    
    # Prepare node colors, sizes, and alphas
    node_colors = []
    node_sizes = []
    node_alphas = []
    for node in graph.nodes():
        if node in degrees:
            node_colors.append(degrees[node])
        else:
            node_colors.append(0)  # Assign 0 degree to nodes not in the original graph
        
        if highlight_nodes is None or node in highlight_nodes:
            node_sizes.append(300)
            node_alphas.append(0.8)
        else:
            node_sizes.append(100)
            node_alphas.append(0.2)
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(graph, pos, 
                           node_color=node_colors,
                           node_size=node_sizes,
                           alpha=node_alphas,
                           cmap=plt.cm.viridis,
                           ax=ax)
    
    # Color-code edges
    edge_colors, edge_types, color_map = get_edge_colors(graph)
    
    if highlight_nodes:
        # Prepare edge colors and alphas
        edge_alphas = []
        for u, v in graph.edges():
            if u in highlight_nodes and v in highlight_nodes:
                edge_alphas.append(1.0)
            else:
                edge_alphas.append(0.1)
        
        edges = nx.draw_networkx_edges(graph, pos, 
                               edge_color=edge_colors,
                               width=0.5,
                               alpha=edge_alphas,
                               ax=ax)
    else:
        edges = nx.draw_networkx_edges(graph, pos, 
                               edge_color=edge_colors,
                               width=0.5,
                               alpha=0.6,
                               ax=ax)
    
    # Add labels with a slight offset for better readability
    label_pos = {k: (v[0], v[1]+0.01) for k, v in pos.items()}
    if highlight_nodes:
        # Only label highlighted nodes
        labels = {node: node for node in highlight_nodes if node in pos}
        nx.draw_networkx_labels(graph, label_pos, labels, font_size=8, font_weight='bold', font_family='sans-serif', ax=ax)
    else:
        nx.draw_networkx_labels(graph, label_pos, font_size=4, font_weight='bold', font_family='sans-serif', ax=ax)
    
    # Add a title
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Remove axis
    ax.axis('off')
    
    # Add a colorbar for node degrees
    cax = fig.add_axes([0.82, 0.1, 0.02, 0.3])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values())))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Node Degree', rotation=270, labelpad=15)
    
    # Add a legend for edge types
    legend_elements = [plt.Line2D([0], [0], color=color_map[edge_type], 
                                  lw=2, label=edge_type) 
                       for edge_type in edge_types]
    
    # Create a separate axes for the legend
    legend_ax = fig.add_axes([0.82, 0.5, 0.15, 0.4])  # [left, bottom, width, height]
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, title="Edge Types", 
                              loc="center left", fontsize='x-small')
    legend.get_title().set_fontsize('small')
    
    plt.show()

def visualize_subgraph(graph: nx.Graph, nodes: List[Any], title: str = "Subgraph of Relevant Nodes", figsize: Tuple[int, int] = (15, 15)):
    fig = plt.figure(figsize=figsize, facecolor='white')
    
    # Create a main axes for the graph
    ax = fig.add_axes([0.05, 0.05, 0.75, 0.9])  # [left, bottom, width, height]
    
    subgraph = graph.subgraph(nodes)
    
    # Use Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(subgraph)
    
    # Calculate node degrees for color mapping
    degrees = dict(subgraph.degree())
    node_colors = [degrees[node] for node in subgraph.nodes()]
    
    # Draw the graph
    nodes = nx.draw_networkx_nodes(subgraph, pos, 
                           node_color=node_colors, 
                           node_size=500,
                           cmap=plt.cm.viridis,
                           alpha=0.8,
                           ax=ax)
    
    # Color-code edges
    edge_colors, edge_types, color_map = get_edge_colors(subgraph)
    edges = nx.draw_networkx_edges(subgraph, pos, 
                           edge_color=edge_colors,
                           width=0.7,
                           alpha=0.6,
                           ax=ax)
    
    # Add labels with a slight offset for better readability
    label_pos = {k: (v[0], v[1]+0.01) for k, v in pos.items()}
    nx.draw_networkx_labels(subgraph, label_pos, font_size=6, font_weight='bold', font_family='sans-serif', ax=ax)
    
    # Add a title
    ax.set_title(title, fontsize=20, fontweight='bold')
    
    # Remove axis
    ax.axis('off')
    
    # Add a colorbar for node degrees
    cax = fig.add_axes([0.82, 0.1, 0.02, 0.3])  # [left, bottom, width, height]
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=min(degrees.values()), vmax=max(degrees.values())))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label('Node Degree', rotation=270, labelpad=15)
    
    # Add a legend for edge types
    legend_elements = [plt.Line2D([0], [0], color=color_map.get(edge_type, 'gray'), 
                                  lw=2, label=edge_type) 
                       for edge_type in edge_types]
    
    # Create a separate axes for the legend
    legend_ax = fig.add_axes([0.82, 0.5, 0.15, 0.4])  # [left, bottom, width, height]
    legend_ax.axis('off')
    legend = legend_ax.legend(handles=legend_elements, title="Edge Types", 
                              loc="center left", fontsize='x-small')
    legend.get_title().set_fontsize('small')
    
    plt.show()

def create_color_map():
    return colors.ListedColormap([
        hex2color('#000000'), hex2color('#0074D9'), hex2color('#FF4136'), 
        hex2color('#2ECC40'), hex2color('#FFDC00'), hex2color('#AAAAAA'), 
        hex2color('#F012BE'), hex2color('#FF851B'), hex2color('#7FDBFF'), 
        hex2color('#870C25')
    ])

def draw_grid(grid, cell_width_px, cmap, draw_colors=False):
    canvas_height = int(len(grid) * cell_width_px)
    canvas_width = int(len(grid[0]) * cell_width_px)
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_DUPLEX
    font_thickness = 1

    # Light grey color for borders (RGB: 220, 220, 220)
    border_color = (220, 220, 220)

    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            top_left = (int(j * cell_width_px), int(i * cell_width_px))
            bottom_right = (int((j + 1) * cell_width_px), int((i + 1) * cell_width_px))
            
            # Draw cell borders in light grey
            cv2.rectangle(canvas, top_left, bottom_right, border_color, 1)

            # Get color for cell
            color_index = int(cell) % len(cmap.colors)
            rgb_color = tuple(int(c * 255) for c in cmap.colors[color_index])
            bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])  # Convert RGB to BGR

            if draw_colors:
                # Fill cell with color
                cv2.rectangle(canvas, (top_left[0]+1, top_left[1]+1), (bottom_right[0]-1, bottom_right[1]-1), bgr_color, -1)
            else:
                # Calculate font scale based on cell size
                font_scale = cell_width_px / 50  # Slightly reduced font size

                # Calculate text size and position
                text = str(cell)
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_x = int(top_left[0] + (cell_width_px - text_size[0]) / 2)
                text_y = int(top_left[1] + (cell_width_px + text_size[1]) / 2)

                # Create a subtle shadow effect instead of bold
                shadow_color = (220, 220, 220)  # Light gray
                cv2.putText(canvas, text, (text_x + 1, text_y + 1), font, font_scale, shadow_color, font_thickness)

                # Draw the main text
                cv2.putText(canvas, text, (text_x, text_y), font, font_scale, bgr_color, font_thickness)

    # Draw outer border in light grey
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, canvas.shape[0] - 1), border_color, 2)
    return canvas

def add_title(canvas, title, x, y):
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.6
    font_thickness = 1
    text_size, _ = cv2.getTextSize(title, font, font_scale, font_thickness)
    text_x = x - text_size[0] // 2  # Center the text horizontally
    text_y = y
    cv2.putText(canvas, title, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)


def create_task_canvas(train_inputs, train_outputs, grid_width, vertical_spacing, draw_colors=True):
    cmap = create_color_map()
    grid_pairs = []
    max_pair_width = 0
    max_pair_height = 0
    total_pairs = len(train_inputs)

    # Prepare grid pairs and calculate max dimensions
    for inp, out in zip(train_inputs, train_outputs):
        input_cell_width_px = grid_width / len(inp[0])
        output_cell_width_px = grid_width / len(out[0])
        
        input_canvas = draw_grid(inp, input_cell_width_px, cmap, draw_colors=draw_colors)
        output_canvas = draw_grid(out, output_cell_width_px, cmap, draw_colors=draw_colors)
        
        input_height, input_width = input_canvas.shape[:2]
        output_height, output_width = output_canvas.shape[:2]
        
        pair_height = max(input_height, output_height) + 30  # Extra 30 pixels for titles
        pair_width = input_width + output_width + 100  # 100 pixels for arrow and spacing
        
        max_pair_width = max(max_pair_width, pair_width)
        max_pair_height = max(max_pair_height, pair_height)
        
        grid_pairs.append((input_canvas, output_canvas, pair_height, pair_width))

    # Calculate optimal number of columns
    aspect_ratio = 4 / 3  # Wider aspect ratio for better space utilization
    num_columns = max(1, min(total_pairs, int(np.sqrt(total_pairs * aspect_ratio))))
    num_rows = (total_pairs + num_columns - 1) // num_columns

    # Calculate canvas dimensions
    canvas_width = num_columns * max_pair_width + (num_columns + 1) * vertical_spacing
    canvas_height = num_rows * (max_pair_height + vertical_spacing) + 2 * vertical_spacing

    canvas = np.ones((int(canvas_height), int(canvas_width), 3), dtype=np.uint8) * 255

    current_x, current_y = vertical_spacing, 2 * vertical_spacing

    for idx, (input_canvas, output_canvas, pair_height, pair_width) in enumerate(grid_pairs):
        if idx > 0 and idx % num_columns == 0:
            current_x = vertical_spacing
            current_y += max_pair_height + vertical_spacing

        input_height, input_width = input_canvas.shape[:2]
        output_height, output_width = output_canvas.shape[:2]

        # Add titles
        add_title(canvas, f"Input Grid {idx + 1} (Size: {len(train_inputs[idx])}x{len(train_inputs[idx][0])})", 
                  current_x + input_width // 2, current_y - 15)
        add_title(canvas, f"Output Grid {idx + 1} (Size: {len(train_outputs[idx])}x{len(train_outputs[idx][0])})", 
                  current_x + input_width + 100 + output_width // 2, current_y - 15)

        # Calculate vertical offsets to center grids
        input_offset = (max_pair_height - input_height) // 2
        output_offset = (max_pair_height - output_height) // 2

        # Place input_canvas
        canvas[current_y + input_offset:current_y + input_offset + input_height, 
               current_x:current_x + input_width] = input_canvas

        # Calculate the start position for output_canvas
        output_start_x = current_x + input_width + 100

        # Place output_canvas
        canvas[current_y + output_offset:current_y + output_offset + output_height, 
               output_start_x:output_start_x + output_width] = output_canvas

        # Draw arrow
        arrow_start = (current_x + input_width + 10, current_y + max_pair_height // 2)
        arrow_end = (output_start_x - 10, arrow_start[1])
        cv2.arrowedLine(canvas, arrow_start, arrow_end, (0, 0, 255), 3, tipLength=0.3)

        current_x += max_pair_width + vertical_spacing

    return canvas

def create_single_grid_canvas(grid, grid_width, title, cmap, draw_colors=True):
    cell_width_px = grid_width / len(grid[0])
    grid_canvas = draw_grid(grid, cell_width_px, cmap, draw_colors=draw_colors)
    grid_height, actual_width = grid_canvas.shape[:2]
    
    # Add extra space for the title
    canvas_height = grid_height + 40
    canvas = np.ones((canvas_height, actual_width, 3), dtype=np.uint8) * 255

    # Draw the grid
    canvas[40:, :] = grid_canvas

    # Add title
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.7  # Increased from 0.5
    font_thickness = 1
    text_size, _ = cv2.getTextSize(title, font, font_scale, font_thickness)
    text_x = int((actual_width - text_size[0]) / 2)
    text_y = 25  # Increased from 20
    cv2.putText(canvas, title, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness)


    return canvas

def render_task(canvas):
    # cv2_imshow(canvas)
    max_height = 800  # You can adjust this value
    scale = min(1.0, max_height / canvas.shape[0])
    resized_canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow('Task Canvas', resized_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_signal_single_grid(grid, grid_width, title, output_path, cmap):
    canvas = create_single_grid_canvas(grid, grid_width, title, cmap)
    
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    if not cv2.imwrite(output_path, canvas):
        print(f"Failed to save {output_path}")
    else:
        print(f"Successfully saved {output_path}")

if __name__ == '__main__':
    base_path = 'arc-prize-2024/'
    output_dir = 'images_train/'
