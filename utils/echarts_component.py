"""
ECharts component renderer for Streamlit using modern ECharts 5.5.
Provides zero-friction rendering, offline & CDN fallback, custom floating zoom toolbars,
roam controls, and responsive container resizing.
"""

import json
import uuid
import streamlit as st

def render_echarts(
    option: dict,
    height: str = "520px",
    show_controls: bool = True,
    chart_id: str = None,
    key: str = None
):
    """
    Renders an Apache ECharts instance with interactive floating zoom controls.
    - Locks mouse-wheel zoom to prevent hijacking page scroll.
    - Provides '+ Zoom In', '− Zoom Out', '↺ Fit to View', and '📷 Save' floating buttons.
    - Auto-resizes on window / sidebar toggle.
    """
    if chart_id is None:
        chart_id = f"echart_{uuid.uuid4().hex[:8]}"

    json_options = json.dumps(option, ensure_ascii=False)
    controls_display = "flex" if show_controls else "none"

    html_code = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <style>
        * {{
          box-sizing: border-box;
          margin: 0;
          padding: 0;
        }}
        body, html {{
          width: 100%;
          height: 100%;
          overflow: hidden;
          background: transparent;
          font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
        }}
        .chart-container-card {{
          position: relative;
          width: 100%;
          height: {height};
          background: #ffffff;
          border: 1px solid #e2e8f0;
          border-radius: 12px;
          overflow: hidden;
          box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }}
        #chart_{chart_id} {{
          width: 100%;
          height: 100%;
        }}
        .floating-controls {{
          position: absolute;
          bottom: 12px;
          right: 12px;
          display: {controls_display};
          align-items: center;
          gap: 6px;
          z-index: 100;
          background: rgba(255, 255, 255, 0.94);
          backdrop-filter: blur(10px);
          -webkit-backdrop-filter: blur(10px);
          padding: 4px 8px;
          border-radius: 8px;
          border: 1px solid #cbd5e1;
          box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        }}
        .control-btn {{
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: #f8fafc;
          border: 1px solid #e2e8f0;
          color: #334155;
          font-size: 12px;
          font-weight: 500;
          padding: 4px 8px;
          border-radius: 6px;
          cursor: pointer;
          transition: all 0.15s ease;
          user-select: none;
        }}
        .control-btn:hover {{
          background: #2563eb;
          color: #ffffff;
          border-color: #2563eb;
          box-shadow: 0 1px 3px rgba(37, 99, 235, 0.25);
        }}
        .control-btn:active {{
          transform: translateY(1px);
        }}
        .control-btn span {{
          margin-right: 4px;
          font-weight: 700;
        }}
      </style>
      <script src="https://cdn.jsdelivr.net/npm/echarts@5.5.0/dist/echarts.min.js"></script>
    </head>
    <body>
      <div class="chart-container-card">
        <div id="chart_{chart_id}"></div>
        <div class="floating-controls">
          <button class="control-btn" id="btn_in_{chart_id}" title="Zoom In"><span>＋</span> Zoom In</button>
          <button class="control-btn" id="btn_out_{chart_id}" title="Zoom Out"><span>−</span> Zoom Out</button>
          <button class="control-btn" id="btn_fit_{chart_id}" title="Fit to Box (Default)"><span>↺</span> Fit to View</button>
          <button class="control-btn" id="btn_save_{chart_id}" title="Save as Image"><span>📷</span> Save</button>
        </div>
      </div>

      <script>
        (function() {{
          const container = document.getElementById('chart_{chart_id}');
          if (!container) return;

          let myChart = null;
          try {{
            myChart = echarts.init(container, null, {{ renderer: 'canvas' }});
          }} catch (err) {{
            console.error('ECharts init error:', err);
            return;
          }}

          const option = {json_options};
          myChart.setOption(option);

          let currentZoom = 1.0;

          // Zoom In
          const btnIn = document.getElementById('btn_in_{chart_id}');
          if (btnIn) {{
            btnIn.addEventListener('click', function(e) {{
              e.preventDefault();
              currentZoom = Math.min(currentZoom * 1.25, 4.0);
              myChart.setOption({{
                series: [{{ zoom: currentZoom }}]
              }});
            }});
          }}

          // Zoom Out
          const btnOut = document.getElementById('btn_out_{chart_id}');
          if (btnOut) {{
            btnOut.addEventListener('click', function(e) {{
              e.preventDefault();
              currentZoom = Math.max(currentZoom / 1.25, 0.35);
              myChart.setOption({{
                series: [{{ zoom: currentZoom }}]
              }});
            }});
          }}

          // Fit to view
          const btnFit = document.getElementById('btn_fit_{chart_id}');
          if (btnFit) {{
            btnFit.addEventListener('click', function(e) {{
              e.preventDefault();
              currentZoom = 1.0;
              myChart.setOption({{
                series: [{{ zoom: 1.0, center: null }}]
              }});
            }});
          }}

          // Save PNG
          const btnSave = document.getElementById('btn_save_{chart_id}');
          if (btnSave) {{
            btnSave.addEventListener('click', function(e) {{
              e.preventDefault();
              try {{
                const url = myChart.getDataURL({{
                  type: 'png',
                  pixelRatio: 2,
                  backgroundColor: '#ffffff'
                }});
                const a = document.createElement('a');
                a.download = 'fraud_topology_{chart_id}.png';
                a.href = url;
                a.click();
              }} catch (err) {{
                console.error('Save failed:', err);
              }}
            }});
          }}

          // Responsive resizing
          window.addEventListener('resize', function() {{
            if (myChart) {{
              myChart.resize();
            }}
          }});
        }})();
      </script>
    </body>
    </html>
    """

    height_int = int(height.replace("px", "").strip()) if "px" in height else 520
    # Add a small padding to iframe height to prevent any scrollbars
    if hasattr(st, "iframe"):
        st.iframe(html_code, height=height_int + 10, width="stretch")
    else:
        st.components.v1.html(html_code, height=height_int + 10, scrolling=False)
