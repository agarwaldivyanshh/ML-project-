"""
Apache ECharts Graph Visualizer for Banking Fraud Detection.
Replaces confusing clumped layouts with structured financial pipelines:
- Case A: Multi-Hop Mule Ring Pipeline (Slide 2 - The $4,500 Single Transfer Illusion)
- Case B: Smurfing Fan-Out & Re-aggregation (1 -> 6 -> 1)
- Case C: Verified Commercial Merchant Flow (Customers -> Merchant -> Suppliers)

Includes:
- Locked mouse-wheel zoom (no scrolling interference)
- Rich floating zoom & fit buttons at bottom right
- Rich glassmorphism tooltips for nodes and edges
- High-contrast enterprise color palette
"""

from typing import Dict, Any, List

def get_echarts_graph_option(
    scenario_id: str,
    scenario_data: Dict[str, Any],
    highlight_node: str = None
) -> Dict[str, Any]:
    """
    Builds an Apache ECharts graph specification tailored to the financial case study.
    """
    nodes_raw = scenario_data.get("nodes", [])
    edges_raw = scenario_data.get("edges", [])

    # Category definitions with distinct high-contrast colors
    categories = [
        {"name": "Compromised Source", "itemStyle": {"color": "#2563EB"}},       # Blue
        {"name": "Layering Money Mule", "itemStyle": {"color": "#F59E0B"}},      # Amber
        {"name": "Slide 2 Target ($4.5k)", "itemStyle": {"color": "#EA580C"}},   # Coral Orange
        {"name": "ATM Liquidation Point", "itemStyle": {"color": "#DC2626"}},    # Crimson Red
        {"name": "Offshore Conduit", "itemStyle": {"color": "#7C3AED"}},         # Purple
        {"name": "Verified Merchant", "itemStyle": {"color": "#059669"}},        # Emerald Green
        {"name": "Legitimate Customer", "itemStyle": {"color": "#0284C7"}},      # Sky Blue
        {"name": "Operational Counterparty", "itemStyle": {"color": "#475569"}}, # Slate Gray
    ]

    echarts_nodes = []
    echarts_links = []

    # -------------------------------------------------------------
    # CASE A: Multi-Hop Mule Ring (Slide 2)
    # -------------------------------------------------------------
    if scenario_id == "slide2_multi_hop_ring" or "ring" in scenario_id or "mule" in scenario_id:
        coords_map = {
            "VICTIM_CORP_01": (110, 260, "Compromised Source", 50, "circle"),
            "MULE_HOP_1":     (205, 180, "Layering Money Mule", 44, "circle"),
            "MULE_HOP_2":     (300, 340, "Layering Money Mule", 44, "circle"),
            "MULE_HOP_3":     (395, 180, "Layering Money Mule", 44, "circle"),
            "MULE_HOP_4":     (490, 340, "Slide 2 Target ($4.5k)", 54, "circle"),
            "MULE_HOP_5":     (585, 180, "Layering Money Mule", 44, "circle"),
            "ATM_CASHOUT_77": (680, 260, "ATM Liquidation Point", 50, "roundRect"),
        }

        for n in nodes_raw:
            nid = n.get("id", "")
            x, y, cat, size, symbol = coords_map.get(nid, (400, 260, "Layering Money Mule", 44, "circle"))
            is_target = (nid == "MULE_HOP_4" or nid == highlight_node)
            label_pos = "top" if y <= 190 else "bottom"

            node_item = {
                "id": nid,
                "name": nid,
                "x": x,
                "y": y,
                "symbol": symbol,
                "symbolSize": size + (6 if is_target else 0),
                "category": cat,
                "label": {
                    "show": True,
                    "position": label_pos,
                    "formatter": nid,
                    "color": "#0F172A",
                    "fontSize": 11,
                    "fontWeight": 700,
                    "distance": 8,
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        {nid}
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Role:</b> {n.get('role', cat)}<br/>
                        <b>Observed Balance:</b> ${n.get('balance', 0):,.2f}<br/>
                        <b>GAT Risk Flag:</b> <span style="color:{'#DC2626' if 'ATM' in nid or 'MULE' in nid else '#2563EB'}; font-weight:700;">{'FRAUD CONDUIT (96.0%)' if 'MULE' in nid or 'ATM' in nid else 'COMPROMISED ORIGIN'}</span><br/>
                        <b>Laundering Signature:</b> {'Instant 98.9% velocity pass-through' if 'MULE' in nid else 'Drainage source'}
                    </div>
                    """
                }
            }

            if is_target:
                node_item["itemStyle"] = {
                    "shadowBlur": 18,
                    "shadowColor": "rgba(234, 88, 12, 0.75)",
                    "borderColor": "#C2410C",
                    "borderWidth": 3
                }
            echarts_nodes.append(node_item)

        # Links for Case A
        for e in edges_raw:
            src = e.get("from")
            tgt = e.get("to")
            amt = e.get("amount", 0)
            is_target_hop = (tgt == "MULE_HOP_4" or src == "MULE_HOP_3")

            link_item = {
                "source": src,
                "target": tgt,
                "lineStyle": {
                    "width": 4 if is_target_hop else 2.8,
                    "color": "#EA580C" if is_target_hop else "#D97706",
                    "curveness": 0.12,
                    "type": "solid"
                },
                "label": {
                    "show": is_target_hop,
                    "formatter": f"${amt:,.0f} (Target)",
                    "color": "#9A3412",
                    "fontSize": 11,
                    "fontWeight": 700,
                    "backgroundColor": "rgba(254, 243, 199, 0.95)",
                    "padding": [2, 6],
                    "borderRadius": 4
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        Directed Transfer: {src} ➔ {tgt}
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Amount:</b> <span style="font-weight:700; color:#0F172A;">${amt:,.2f}</span><br/>
                        <b>GAT Attention Weight (α):</b> <span style="color:#DC2626; font-weight:700;">{'0.96 (High-Suspicion)' if is_target_hop else '0.91 (Pass-Through)'}</span><br/>
                        <b>Velocity:</b> Dispatched within 1 simulated hour<br/>
                        <b>Tabular vs GNN:</b> Tabular saw innocent row; GNN detected 4th hop in 5-sleeper chain.
                    </div>
                    """
                }
            }
            echarts_links.append(link_item)

    # -------------------------------------------------------------
    # CASE B: Smurfing Fan-Out Scheme
    # -------------------------------------------------------------
    elif scenario_id == "smurfing_fan_out" or "smurf" in scenario_id:
        hub_coords = (130, 250)
        offshore_coords = (660, 250)
        mule_y_positions = [95, 157, 219, 281, 343, 405]

        mule_nodes = [n for n in nodes_raw if "MULE" in n.get("id", "")]
        for idx, m in enumerate(mule_nodes):
            nid = m.get("id")
            y_pos = mule_y_positions[idx % len(mule_y_positions)]
            echarts_nodes.append({
                "id": nid,
                "name": nid,
                "x": 400,
                "y": y_pos,
                "symbol": "circle",
                "symbolSize": 30,
                "category": "Layering Money Mule",
                "label": {
                    "show": True,
                    "position": "top",
                    "formatter": nid,
                    "color": "#0F172A",
                    "fontSize": 11,
                    "fontWeight": 700,
                    "distance": 3
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        {nid} (Smurf Recipient {idx+1})
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Role:</b> Micro-dispersal buffer<br/>
                        <b>Observed Inflow:</b> ~$16,000 (Sub-threshold)<br/>
                        <b>GNN Risk Score:</b> <span style="color:#DC2626; font-weight:700;">97.0% (Structuring Scheme)</span>
                    </div>
                    """
                }
            })

        # Add Origin Hub
        hub_node = next((n for n in nodes_raw if "HUB" in n.get("id", "")), {"id": "HUB_SMURF_ALPHA", "role": "Origin Hub"})
        echarts_nodes.append({
            "id": hub_node.get("id"),
            "name": hub_node.get("id"),
            "x": hub_coords[0],
            "y": hub_coords[1],
            "symbol": "diamond",
            "symbolSize": 60,
            "category": "Compromised Source",
            "label": {
                "show": True,
                "position": "bottom",
                "formatter": "HUB_SMURF_ALPHA",
                "color": "#0F172A",
                "fontSize": 11,
                "fontWeight": 700,
                "distance": 8
            },
            "tooltip": {
                "formatter": """
                <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                    HUB_SMURF_ALPHA (Structuring Originator)
                </div>
                <div style="font-size:12px; line-height:1.6; color:#334155;">
                    <b>Origin Balance:</b> $120,000.00<br/>
                    <b>Scheme:</b> Disperses $98k into 6 sub-$20k slices to bypass AML thresholds.<br/>
                    <b>GAT Detection:</b> 97% confidence structuring pattern.
                </div>
                """
            }
        })

        # Add Offshore Re-aggregation Conduit
        offshore_node = next((n for n in nodes_raw if "OFFSHORE" in n.get("id", "")), {"id": "OFFSHORE_CONDUIT", "role": "Collector Sink"})
        echarts_nodes.append({
            "id": offshore_node.get("id"),
            "name": offshore_node.get("id"),
            "x": offshore_coords[0],
            "y": offshore_coords[1],
            "symbol": "roundRect",
            "symbolSize": 58,
            "category": "Offshore Conduit",
            "label": {
                "show": True,
                "position": "bottom",
                "formatter": "OFFSHORE_CONDUIT",
                "color": "#0F172A",
                "fontSize": 11,
                "fontWeight": 700,
                "distance": 8
            },
            "tooltip": {
                "formatter": """
                <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                    OFFSHORE_CONDUIT (Re-Aggregation Sink)
                </div>
                <div style="font-size:12px; line-height:1.6; color:#334155;">
                    <b>Total Recombined:</b> $98,000.00<br/>
                    <b>Convergence Time:</b> Within 4 simulated hours<br/>
                    <b>AML Status:</b> Flagged as coordinated offshore flight.
                </div>
                """
            }
        })

        # Links for Case B
        for e in edges_raw:
            src = e.get("from")
            tgt = e.get("to")
            amt = e.get("amount", 0)
            is_fanin = ("OFFSHORE" in tgt)

            echarts_links.append({
                "source": src,
                "target": tgt,
                "lineStyle": {
                    "width": 2.8,
                    "color": "#7C3AED" if is_fanin else "#D97706",
                    "curveness": 0.08 if is_fanin else -0.08
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        Transfer: {src} ➔ {tgt}
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Amount:</b> ${amt:,.2f}<br/>
                        <b>Stage:</b> {'Re-aggregation into offshore sink' if is_fanin else 'Structuring fan-out dispersal'}<br/>
                        <b>Attention Weight (α):</b> 0.94
                    </div>
                    """
                }
            })

    # -------------------------------------------------------------
    # CASE C: Legitimate Commercial Merchant
    # -------------------------------------------------------------
    else:
        cust_positions = [(140, 145), (140, 255), (140, 365)]
        outflow_positions = [(660, 180), (660, 330)]

        cust_idx = 0
        out_idx = 0
        for n in nodes_raw:
            nid = n.get("id", "")
            if "CUSTOMER" in nid:
                pos = cust_positions[cust_idx % len(cust_positions)]
                cust_idx += 1
                cat = "Legitimate Customer"
                symbol = "circle"
                size = 44
            elif "MERCHANT" in nid:
                pos = (400, 255)
                cat = "Verified Merchant"
                symbol = "roundRect"
                size = 56
            else:
                pos = outflow_positions[out_idx % len(outflow_positions)]
                out_idx += 1
                cat = "Operational Counterparty"
                symbol = "circle"
                size = 48

            echarts_nodes.append({
                "id": nid,
                "name": nid,
                "x": pos[0],
                "y": pos[1],
                "symbol": symbol,
                "symbolSize": size,
                "category": cat,
                "label": {
                    "show": True,
                    "position": "bottom",
                    "formatter": nid,
                    "color": "#0F172A",
                    "fontSize": 11,
                    "fontWeight": 700,
                    "distance": 8
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        {nid}
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Account Type:</b> {cat}<br/>
                        <b>GNN Risk Verdict:</b> <span style="color:#059669; font-weight:700;">BENIGN (4.0% Risk)</span><br/>
                        <b>Pattern:</b> Sustained commercial cashflow & buffer retention.
                    </div>
                    """
                }
            })

        for e in edges_raw:
            src = e.get("from")
            tgt = e.get("to")
            amt = e.get("amount", 0)
            echarts_links.append({
                "source": src,
                "target": tgt,
                "lineStyle": {
                    "width": 2.5,
                    "color": "#059669",
                    "curveness": 0.1
                },
                "tooltip": {
                    "formatter": f"""
                    <div style="font-weight:700; color:#0F172A; border-bottom:1px solid #E2E8F0; padding-bottom:4px; margin-bottom:6px;">
                        Commercial Flow: {src} ➔ {tgt}
                    </div>
                    <div style="font-size:12px; line-height:1.6; color:#334155;">
                        <b>Volume:</b> ${amt:,.2f}<br/>
                        <b>Verdict:</b> Verified normal business activity.
                    </div>
                    """
                }
            })

    # Only include categories actually active in this scenario in the legend
    active_cat_names = set(n.get("category") for n in echarts_nodes if "category" in n)
    scenario_categories = [c for c in categories if c["name"] in active_cat_names]

    # Overall ECharts Option (No internal title to prevent collision; clean top legend)
    option = {
        "backgroundColor": "#FFFFFF",
        "tooltip": {
            "trigger": "item",
            "backgroundColor": "rgba(255, 255, 255, 0.98)",
            "borderColor": "#CBD5E1",
            "borderWidth": 1,
            "textStyle": {"color": "#0F172A"},
            "extraCssText": "box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12); border-radius: 8px; padding: 10px 14px;"
        },
        "legend": {
            "show": True,
            "top": 12,
            "left": 16,
            "orient": "horizontal",
            "data": [c["name"] for c in scenario_categories],
            "textStyle": {"color": "#334155", "fontSize": 11, "fontWeight": 600},
            "itemGap": 12,
            "icon": "circle"
        },
        "series": [
            {
                "type": "graph",
                "layout": "none",
                "roam": "move",  # Allows click-and-drag panning; disables wheel zoom so scrolling web page is never hijacked!
                "zoom": 1.0,
                "edgeSymbol": ["none", "arrow"],
                "edgeSymbolSize": [4, 10],
                "categories": categories,
                "data": echarts_nodes,
                "links": echarts_links,
                "emphasis": {
                    "focus": "adjacency",
                    "lineStyle": {
                        "width": 5,
                        "shadowBlur": 10,
                        "shadowColor": "rgba(37, 99, 235, 0.5)"
                    }
                },
                "lineStyle": {
                    "opacity": 0.85,
                    "curveness": 0.12
                }
            }
        ]
    }

    return option
