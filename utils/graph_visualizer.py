import os
import json
from pyvis.network import Network

def generate_interactive_network_html(nodes, edges, target_node_id=None, height="480px", is_light=True):
    """
    Renders an interactive network graph with luxury editorial styling.
    Clean, minimal design with refined color palette and elegant tooltips.
    """
    if is_light:
        bg_color = "#FAF8F5"
        font_color = "#1C1917"
        canvas_border = "#E5DCCE"
        tooltip_bg = "#FFFFFF"
        tooltip_text = "#1C1917"
        tooltip_border = "#DFD5C6"
        edge_label_bg = "#FAF8F5"
        edge_label_text = "#1C1917"
    else:
        bg_color = "#111827"
        font_color = "#F8FAFC"
        canvas_border = "#334155"
        tooltip_bg = "#111827"
        tooltip_text = "#F8FAFC"
        tooltip_border = "#475467"
        edge_label_bg = "#111827"
        edge_label_text = "#F8FAFC"

    net = Network(height=height, width="100%", bgcolor=bg_color, font_color=font_color, directed=True)

    # Refined physics for elegant, stable layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.06,
          "damping": 0.55
        },
        "maxVelocity": 20,
        "solver": "forceAtlas2Based",
        "timestep": 0.4,
        "stabilization": {"iterations": 150}
      },
      "edges": {
        "smooth": {"type": "continuous", "roundness": 0.3},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.7}},
        "color": {"inherit": false}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 60,
        "zoomView": true,
        "dragView": true,
        "dragNodes": false
      },
      "nodes": {
        "font": {"size": 11, "face": "Plus Jakarta Sans, sans-serif", "bold": true}
      }
    }
    """)

    # Refined color palette - Luxury Editorial
    color_map = {
        'Victim': {'color': '#2E3440', 'shape': 'diamond', 'size': 28, 'text': '#FFFFFF'},
        'Compromised': {'color': '#2E3440', 'shape': 'diamond', 'size': 28, 'text': '#FFFFFF'},
        'Cash-out': {'color': '#8A3324', 'shape': 'square', 'size': 26, 'text': '#FFFFFF'},
        'ATM': {'color': '#8A3324', 'shape': 'square', 'size': 26, 'text': '#FFFFFF'},
        'OFFSHORE': {'color': '#8A3324', 'shape': 'square', 'size': 26, 'text': '#FFFFFF'},
        'Laundering': {'color': '#98592B', 'shape': 'hexagon', 'size': 28, 'text': '#FFFFFF'},
        'Hub': {'color': '#98592B', 'shape': 'hexagon', 'size': 28, 'text': '#FFFFFF'},
        'Smurf': {'color': '#98592B', 'shape': 'hexagon', 'size': 28, 'text': '#FFFFFF'},
        'Mule': {'color': '#C06B3E', 'shape': 'circle', 'size': 24, 'text': '#1C1917'},
        'Merchant': {'color': '#4A7253', 'shape': 'box', 'size': 26, 'text': '#FFFFFF'},
        'Account': {'color': '#8B9582', 'shape': 'circle', 'size': 20, 'text': '#1C1917'}
    }

    # Add nodes with refined styling
    for node in nodes:
        node_id = node['id']
        node_type = node.get('type', 'Account')
        role = node.get('role', '')
        balance = node.get('balance', 0.0)

        is_target = (node_id == target_node_id)

        # Get color scheme based on type/role
        config = color_map.get(node_type, color_map.get(role, color_map['Account']))
        color = config['color']
        shape = config['shape']
        size = config['size']
        text_color = config['text']

        # Target node gets special treatment
        if is_target:
            border_color = '#8A3324'  # Deep rust for target
            border_width = 3
        else:
            border_color = '#4A3B32'
            border_width = 1

        # Clean tooltip
        tooltip_text = (
            f"Account: {node_id}\n"
            f"Role: {role if role else node_type}\n"
            f"Balance: ${balance:,.2f}"
        )

        net.add_node(
            node_id,
            label=node_id,
            title=tooltip_text,
            color={'background': color, 'border': border_color, 'highlight': {'background': color, 'border': border_color}},
            borderWidth=border_width,
            shape=shape,
            size=size,
            font={'size': 11, 'face': 'Plus Jakarta Sans, sans-serif', 'color': text_color, 'bold': True}
        )

    # Add edges with refined styling
    for edge in edges:
        source = edge['from']
        target = edge['to']
        amount = edge.get('amount', 0.0)
        attn = edge.get('attention_weight', 0.5)
        is_target_edge = edge.get('is_target', False)

        # Refined edge styling - no cluttered labels
        if is_target_edge:
            color = "#8A3324"  # Deep rust
            width = 3.5
            # No label on edges - too cluttered
        elif attn >= 0.70:
            color = "#98592B"  # Caramel
            width = 2.5
        else:
            color = "#A89F91"  # Neutral taupe
            width = 1.2

        edge_tooltip = (
            f"Amount: ${amount:,.2f}\n"
            f"Attention: {attn:.2f}"
        )

        net.add_edge(
            source,
            target,
            title=edge_tooltip,
            color=color,
            width=width,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.6}},
            smooth={'type': 'continuous', 'roundness': 0.3}
        )

    raw_html = net.generate_html()

    # Inject luxury styling
    if is_light:
        custom_style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

        #mynetwork {
            border: 1px solid #E5DCCE !important;
            border-radius: 12px !important;
            background-color: #FAF8F5 !important;
        }

        div.vis-tooltip {
            position: absolute;
            background-color: #FFFFFF !important;
            color: #1C1917 !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            padding: 12px 16px !important;
            border-radius: 10px !important;
            border: 1px solid #DFD5C6 !important;
            box-shadow: 0 8px 24px rgba(28, 25, 23, 0.10) !important;
            white-space: pre-line !important;
            line-height: 1.55 !important;
            max-width: 280px !important;
            z-index: 9999 !important;
            pointer-events: none !important;
        }

        .vis-network canvas {
            border-radius: 12px !important;
        }

        /* Hide edge labels - too cluttered */
        .vis-edge-label {
            display: none !important;
        }

        /* Refined node hover effect */
        .vis-node:hover {
            transform: scale(1.1) !important;
        }
        </style>
        """
    else:
        custom_style = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

        #mynetwork {
            border: 1px solid #334155 !important;
            border-radius: 12px !important;
            background-color: #111827 !important;
        }

        div.vis-tooltip {
            position: absolute;
            background-color: #111827 !important;
            color: #F8FAFC !important;
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            padding: 12px 16px !important;
            border-radius: 10px !important;
            border: 1px solid #475467 !important;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.30) !important;
            white-space: pre-line !important;
            line-height: 1.55 !important;
            max-width: 280px !important;
            z-index: 9999 !important;
            pointer-events: none !important;
        }

        .vis-network canvas {
            border-radius: 12px !important;
        }

        .vis-edge-label {
            display: none !important;
        }

        .vis-node:hover {
            transform: scale(1.1) !important;
        }
        </style>
        """

    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", f"{custom_style}</head>")
    else:
        raw_html = custom_style + raw_html

    return raw_html
