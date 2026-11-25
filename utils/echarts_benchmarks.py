"""
ECharts chart configurations for model benchmarks, capability radar, and temporal volume.
"""

from typing import Dict, Any

def get_benchmark_bar_option() -> Dict[str, Any]:
    """
    Horizontal grouped bar chart comparing GAT, GraphSAGE, XGBoost, and Random Forest.
    """
    models = ["Random Forest\n(Control)", "XGBoost\n(Baseline)", "GraphSAGE\n(Neighbor)", "GAT\n(Attention)"]
    
    return {
        "backgroundColor": "#FFFFFF",
        "tooltip": {
            "trigger": "axis",
            "axisPointer": {"type": "shadow"},
            "backgroundColor": "rgba(255, 255, 255, 0.98)",
            "borderColor": "#E2E8F0",
            "textStyle": {"color": "#0F172A"}
        },
        "legend": {
            "top": 12,
            "left": 16,
            "data": ["PR-AUC (Primary)", "ROC-AUC", "Precision", "Recall"],
            "textStyle": {"color": "#334155", "fontSize": 11, "fontWeight": 600},
            "itemGap": 12,
            "icon": "circle"
        },
        "grid": {
            "left": "22%",
            "right": "8%",
            "bottom": "10%",
            "top": "16%",
            "containLabel": True
        },
        "xAxis": {
            "type": "value",
            "min": 0.65,
            "max": 1.0,
            "axisLabel": {
                "formatter": "{value}",
                "color": "#64748B",
                "fontSize": 11
            },
            "splitLine": {"lineStyle": {"type": "dashed", "color": "#F1F5F9"}}
        },
        "yAxis": {
            "type": "category",
            "data": models,
            "axisLabel": {"color": "#1E293B", "fontWeight": 700, "fontSize": 11}
        },
        "series": [
            {
                "name": "PR-AUC (Primary)",
                "type": "bar",
                "data": [0.762, 0.841, 0.918, 0.947],
                "itemStyle": {"color": "#2563EB", "borderRadius": [0, 4, 4, 0]},
                "label": {"show": True, "position": "right", "formatter": "{c}", "color": "#1E293B", "fontWeight": 700, "fontSize": 10}
            },
            {
                "name": "ROC-AUC",
                "type": "bar",
                "data": [0.912, 0.945, 0.976, 0.988],
                "itemStyle": {"color": "#0284C7", "borderRadius": [0, 4, 4, 0]},
                "label": {"show": False}
            },
            {
                "name": "Precision",
                "type": "bar",
                "data": [0.781, 0.852, 0.904, 0.932],
                "itemStyle": {"color": "#059669", "borderRadius": [0, 4, 4, 0]},
                "label": {"show": False}
            },
            {
                "name": "Recall",
                "type": "bar",
                "data": [0.745, 0.830, 0.928, 0.951],
                "itemStyle": {"color": "#D97706", "borderRadius": [0, 4, 4, 0]},
                "label": {"show": False}
            }
        ]
    }

def get_model_radar_option() -> Dict[str, Any]:
    """
    Radar chart comparing model architectures across 5 strategic AML capabilities.
    """
    return {
        "backgroundColor": "#FFFFFF",
        "tooltip": {
            "trigger": "item",
            "backgroundColor": "rgba(255, 255, 255, 0.98)",
            "borderColor": "#E2E8F0",
            "textStyle": {"color": "#0F172A"}
        },
        "legend": {
            "top": 12,
            "left": 16,
            "data": ["GAT (Attention)", "GraphSAGE", "XGBoost", "Random Forest"],
            "textStyle": {"color": "#334155", "fontSize": 11, "fontWeight": 600},
            "itemGap": 12,
            "icon": "circle"
        },
        "radar": {
            "indicator": [
                {"name": "Multi-Hop Vision", "max": 100},
                {"name": "Sleeper Mule Catch", "max": 100},
                {"name": "Imbalance Precision", "max": 100},
                {"name": "Explainability (XAI)", "max": 100},
                {"name": "Throughput Speed", "max": 100}
            ],
            "radius": "50%",
            "center": ["50%", "58%"],
            "splitNumber": 4,
            "axisName": {"color": "#334155", "fontWeight": 700, "fontSize": 11},
            "splitLine": {"lineStyle": {"color": "#E2E8F0"}},
            "splitArea": {
                "show": True,
                "areaStyle": {
                    "color": ["rgba(250, 250, 250, 0.4)", "rgba(240, 245, 255, 0.5)"]
                }
            }
        },
        "series": [
            {
                "type": "radar",
                "data": [
                    {
                        "value": [98, 96, 95, 94, 82],
                        "name": "GAT (Attention)",
                        "lineStyle": {"width": 2.5, "color": "#2563EB"},
                        "areaStyle": {"color": "rgba(37, 99, 235, 0.2)"}
                    },
                    {
                        "value": [92, 91, 91, 72, 88],
                        "name": "GraphSAGE",
                        "lineStyle": {"width": 2, "color": "#0284C7"},
                        "areaStyle": {"color": "rgba(2, 132, 199, 0.15)"}
                    },
                    {
                        "value": [22, 30, 84, 65, 96],
                        "name": "XGBoost",
                        "lineStyle": {"width": 1.5, "color": "#D97706", "type": "dashed"},
                        "areaStyle": {"color": "rgba(217, 119, 6, 0.08)"}
                    },
                    {
                        "value": [16, 22, 76, 58, 92],
                        "name": "Random Forest",
                        "lineStyle": {"width": 1.5, "color": "#64748B", "type": "dotted"}
                    }
                ]
            }
        ]
    }

def get_hourly_fraud_spike_option() -> Dict[str, Any]:
    """
    PaySim temporal transaction volume and synthetic fraud burst curve.
    """
    hours = [f"Hr {i}" for i in range(1, 30)]
    legit_volume = [1420, 1510, 1380, 1620, 1890, 2400, 3100, 3800, 4200, 4050, 3900, 3750, 3600, 3450, 3300, 3100, 2900, 2800, 2600, 2400, 2200, 1950, 1750, 1550, 1480, 1650, 1900, 2300, 2800]
    fraud_bursts = [0, 0, 0, 2, 0, 12, 45, 18, 0, 0, 0, 0, 0, 52, 68, 25, 0, 0, 0, 0, 0, 0, 0, 8, 34, 12, 0, 0, 0]

    return {
        "backgroundColor": "#FFFFFF",
        "tooltip": {
            "trigger": "axis",
            "backgroundColor": "rgba(255, 255, 255, 0.98)",
            "borderColor": "#E2E8F0"
        },
        "legend": {
            "top": 12,
            "left": 16,
            "data": ["Normal Transactions", "Synchronized Fraud Burst"],
            "textStyle": {"color": "#334155", "fontSize": 11, "fontWeight": 600},
            "icon": "circle"
        },
        "grid": {"left": "5%", "right": "5%", "bottom": "12%", "top": "18%", "containLabel": True},
        "xAxis": {
            "type": "category",
            "boundaryGap": False,
            "data": hours,
            "axisLabel": {"color": "#64748B", "fontSize": 10}
        },
        "yAxis": [
            {"type": "value", "name": "Legitimate Tx", "axisLabel": {"color": "#64748B"}, "splitLine": {"lineStyle": {"color": "#F1F5F9"}}},
            {"type": "value", "name": "Fraud Tx", "axisLabel": {"color": "#DC2626"}, "splitLine": {"show": False}}
        ],
        "series": [
            {
                "name": "Normal Transactions",
                "type": "line",
                "smooth": True,
                "data": legit_volume,
                "itemStyle": {"color": "#0284C7"},
                "areaStyle": {"color": "rgba(2, 132, 199, 0.12)"}
            },
            {
                "name": "Synchronized Fraud Burst",
                "type": "line",
                "yAxisIndex": 1,
                "smooth": True,
                "data": fraud_bursts,
                "itemStyle": {"color": "#DC2626"},
                "areaStyle": {"color": "rgba(220, 38, 38, 0.25)"}
            }
        ]
    }
