import os
import json
from pyvis.network import Network

def generate_interactive_network_html(nodes, edges, target_node_id=None, height="520px"):
    """
    Renders an interactive financial transaction network matching high-end SaaS dashboards:
    - Background: Clean white / light canvas #FFFFFF
    - Nodes: Color-coded with high visual contrast
    - Edges: Directed arrows with amounts and GAT attention weights
    - Tooltips: Clean modern card with crisp typography
    """
    net = Network(height=height, width="100%", bgcolor="#FFFFFF", font_color="#0F172A", directed=True)

    # Physics options for smooth organic layout with draggable nodes
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -70,
          "centralGravity": 0.015,
          "springLength": 140,
          "springConstant": 0.08,
          "damping": 0.5
        },
        "maxVelocity": 30,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": {"iterations": 130}
      },
      "edges": {
        "smooth": {"type": "curvedCW", "roundness": 0.15},
        "arrows": {"to": {"enabled": true, "scaleFactor": 0.85}}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 80,
        "zoomView": true,
        "dragView": true,
        "dragNodes": true
      }
    }
    """)

    # Add nodes
    for node in nodes:
        node_id = node['id']
        node_type = str(node.get('type', 'Account'))
        role = str(node.get('role', ''))
        balance = float(node.get('balance', 0.0))

        is_target = (node_id == target_node_id)
        combined_desc = (node_type + " " + role + " " + node_id).lower()

        # Rich, High-Contrast Node Colors
        if 'victim' in combined_desc or 'compromised' in combined_desc:
            color = "#1E293B"  # Obsidian Navy
            shape = "diamond"
            size = 32
            text_color = "#FFFFFF"
        elif 'atm' in combined_desc or 'cash-out' in combined_desc or 'cashout' in combined_desc or 'offshore' in combined_desc:
            color = "#DC2626"  # Crimson Red
            shape = "square"
            size = 30
            text_color = "#FFFFFF"
        elif 'hub' in combined_desc or 'laundering' in combined_desc or 'smurf' in combined_desc:
            color = "#B45309"  # Deep Caramel
            shape = "hexagon"
            size = 34
            text_color = "#FFFFFF"
        elif 'mule' in combined_desc:
            color = "#D97706"  # Vibrant Amber/Terracotta
            shape = "dot"
            size = 28
            text_color = "#0F172A"
        elif 'merchant' in combined_desc:
            color = "#059669"  # Emerald Green
            shape = "box"
            size = 32
            text_color = "#FFFFFF"
        else:
            color = "#64748B"  # Slate Gray
            shape = "dot"
            size = 24
            text_color = "#0F172A"

        if is_target:
            border_color = "#E11D48"  # Rose Red highlight ring
            border_width = 4
        else:
            border_color = "#0F172A" if shape == "dot" else color
            border_width = 2

        tooltip_text = (
            f"Account: {node_id}\n"
            f"Role: {role if role else node_type}\n"
            f"Classification: {node_type}\n"
            f"Current Balance: ${balance:,.2f}"
        )

        net.add_node(
            node_id,
            label=node_id,
            title=tooltip_text,
            color={
                'background': color,
                'border': border_color,
                'highlight': {'background': '#F59E0B', 'border': '#B45309'}
            },
            borderWidth=border_width,
            shape=shape,
            size=size,
            font={'size': 12, 'face': 'Plus Jakarta Sans, sans-serif', 'color': text_color, 'bold': True}
        )

    # Add edges
    for edge in edges:
        source = edge['from']
        target = edge['to']
        amount = float(edge.get('amount', 0.0))
        tx_type = str(edge.get('type', 'TRANSFER'))
        step = int(edge.get('step', 0))
        attn = float(edge.get('attention_weight', 0.5))
        is_target_edge = bool(edge.get('is_target', False))

        if is_target_edge:
            color = "#E11D48"  # Glowing Rose
            width = 4.5
            label = f"${amount:,.0f} [ATTN: {attn:.2f}]"
        elif attn >= 0.70:
            color = "#D97706"  # Amber Laundering Path
            width = 3.5
            label = f"${amount:,.0f} (α={attn:.2f})"
        else:
            color = "#94A3B8"  # Clean Neutral Slate
            width = 1.5
            label = f"${amount:,.0f}"

        edge_tooltip = (
            f"Transaction: {tx_type}\n"
            f"Amount: ${amount:,.2f}\n"
            f"Time Step: Hour {step}\n"
            f"GAT Attention Weight: {attn:.3f}\n"
            f"Risk Level: {'HIGH RISK LAUNDERING' if attn >= 0.7 else 'NORMAL'}"
        )

        net.add_edge(
            source,
            target,
            title=edge_tooltip,
            label=label,
            color=color,
            width=width,
            font={'size': 11, 'align': 'horizontal', 'background': '#FFFFFF', 'strokeWidth': 0, 'color': '#0F172A', 'bold': True}
        )

    raw_html = net.generate_html()
    # Strip relative script tag that causes console 404 in iframe
    raw_html = raw_html.replace('<script src="lib/bindings/utils.js"></script>', '')

    # High-end tooltip CSS injected into the HTML
    custom_tooltip_style = """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&display=swap');

    div.vis-tooltip {
        position: absolute;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        padding: 12px 16px !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 10px 25px rgba(15, 23, 42, 0.12) !important;
        white-space: pre-line !important;
        line-height: 1.55 !important;
        max-width: 280px !important;
        z-index: 9999 !important;
        pointer-events: none !important;
    }

    #mynetwork {
        border: 1px solid #E2E8F0 !important;
        border-radius: 16px !important;
        background-color: #FFFFFF !important;
    }
    </style>
    """

    if "</head>" in raw_html:
        raw_html = raw_html.replace("</head>", f"{custom_tooltip_style}</head>")
    else:
        raw_html = custom_tooltip_style + raw_html

    return raw_html
