"""
Shared HTML utilities for v8 plots.
"""


def wrap_figures_with_cycle_dropdown(figure_htmls: dict, page_title: str) -> str:
    options_html = ""
    divs_html = ""

    for i, (label, fig_html) in enumerate(figure_htmls.items()):
        safe_id = f"cycle_view_{i}"
        selected = "selected" if i == 0 else ""
        display = "block" if i == 0 else "none"
        options_html += f'<option value="{safe_id}" {selected}>{label}</option>\n'
        divs_html += f'<div id="{safe_id}" style="display:{display};">\n{fig_html}\n</div>\n'

    return f"""<!DOCTYPE html>
<html><head>
<title>{page_title}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 20px; }}
    .controls {{ background: #f8f9fa; padding: 12px 20px; border-radius: 8px; margin-bottom: 16px;
                 display: flex; flex-wrap: wrap; align-items: center; gap: 12px; }}
    .controls h2 {{ margin: 0; font-size: 18px; width: 100%; }}
    .filter-group {{ display: flex; align-items: center; gap: 6px; }}
    .filter-group label {{ font-size: 13px; font-weight: 600; color: #555; white-space: nowrap; }}
    .filter-group select {{ font-size: 13px; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }}
</style>
</head><body>
<div class="controls">
    <h2>{page_title}</h2>
    <div class="filter-group">
        <label>Cycle:</label>
        <select id="cycleSelect" onchange="switchCycle()">
            {options_html}
        </select>
    </div>
</div>
{divs_html}
<script>
function switchCycle() {{
    var sel = document.getElementById('cycleSelect');
    var views = document.querySelectorAll('[id^="cycle_view_"]');
    views.forEach(function(v) {{ v.style.display = 'none'; }});
    document.getElementById(sel.value).style.display = 'block';
    window.dispatchEvent(new Event('resize'));
}}
</script>
</body></html>"""
