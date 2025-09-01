import os
import json
import re
from collections import defaultdict
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np

STEERING_N_NEURONS_SWEEP = [1, 3, 5, 10]
STEERING_MULTIPLIERS_SWEEP = [1.0, 10.0, 50.0]

def find_experiment_data():
    """
    Scans the 'data/' directory to find and parse all steering_summary.json files.
    """
    all_experiments_data = defaultdict(dict)
    data_root = 'data'
    if not os.path.isdir(data_root):
        print(f"Error: The '{data_root}' directory was not found.")

    for root, dirs, files in os.walk(data_root):
        if 'steering' in dirs and os.path.basename(root).startswith("attribution_"):
            exp_id = os.path.basename(root).split('_')[-1]
            steering_path = os.path.join(root, 'steering')
            analysis_modes = [d for d in os.listdir(steering_path) if os.path.isdir(os.path.join(steering_path, d))]

            for mode in analysis_modes:
                json_path = os.path.join(steering_path, mode, 'steering_summary.json')
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as f:
                            raw_data = json.load(f)
                        parsed_data = parse_and_normalize_data(raw_data)
                        all_experiments_data[exp_id][mode] = parsed_data
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read or parse {json_path}. Error: {e}")

    return dict(all_experiments_data)

def parse_and_normalize_data(raw_data):
    """
    Parses the raw JSON data, extracts parameters, and normalizes the results to percentages.
    """
    parsed_entries = []
    entry_pattern = re.compile(r'debias_(male|female)_sentences_n(\d+)_m([\d\.]+)')
    outcomes = ['bias_flipped', 'bias_neutralized', 'bias_persists', 'conflicting_bias_induced']

    for key, values in raw_data.items():
        match = entry_pattern.match(key)
        if not match:
            continue

        gender, n, m = match.groups()
        total = values.get('total_attempts', 1)
        if total == 0: total = 1

        entry = {
            'gender': gender,
            'n': int(n),
            'm': float(m)
        }
        for outcome in outcomes:
            entry[outcome] = values.get(outcome, 0)
            entry[f"{outcome}_perc"] = (values.get(outcome, 0) / total) * 100

        parsed_entries.append(entry)
    return parsed_entries

def analyze_and_generate_report(data):
    """
    Performs analysis and generates an HTML report with a tabbed interface.
    """
    flat_data = []
    for exp_id, modes in data.items():
        for mode, data_points in modes.items():
            for point in data_points:
                point['exp_id'] = exp_id
                point['mode'] = mode
                flat_data.append(point)

    df = pd.DataFrame(flat_data)
    outcomes = ['bias_flipped', 'bias_neutralized', 'bias_persists', 'conflicting_bias_induced']

    # Create 'both' gender data
    both_df = df.groupby(['exp_id', 'mode', 'n', 'm'])[outcomes].sum().reset_index()
    both_df['gender'] = 'both'
    total_attempts = both_df[outcomes].sum(axis=1)
    for outcome in outcomes:
        perc_col = f'{outcome}_perc'
        both_df[perc_col] = np.divide(both_df[outcome].to_numpy(dtype=float), total_attempts.to_numpy(dtype=float), 
                                      out=np.zeros_like(both_df[outcome].to_numpy(dtype=float)), 
                                      where=total_attempts.to_numpy(dtype=float)!=0) * 100
    df = pd.concat([df, both_df], ignore_index=True).fillna(0)

    # --- HTML Generation ---
    summary_p_values = defaultdict(lambda: defaultdict(dict))
    experiments = sorted(df['exp_id'].unique())
    all_multipliers = sorted(df['m'].unique())
    
    # Main Tab Buttons
    tab_buttons_html = '<div class="tab-nav">'
    for i, exp_id in enumerate(experiments):
        active_class = "active" if i == 0 else ""
        tab_buttons_html += f'<button class="tab-button {active_class}" onclick="openExperimentTab(event, \'{exp_id}\')">{exp_id.upper()}</button>'
    tab_buttons_html += '</div>'
    
    # Main Tab Content
    tab_content_html = ""
    for i, exp_id in enumerate(experiments):
        active_class = "active" if i == 0 else ""
        tab_content_html += f'<div id="{exp_id}" class="tab-content {active_class}">'
        
        sub_tab_buttons_html = '<div class="sub-tab-nav">'
        sub_tab_content_html = ""
        genders = ['male', 'female', 'both']

        for j, gender in enumerate(genders):
            sub_active_class = "active" if j == 0 else ""
            sub_tab_buttons_html += f'<button class="sub-tab-button {sub_active_class}" onclick="openGenderSubTab(event, \'{exp_id}\', \'{gender}\')">{gender.capitalize()}</button>'
            sub_tab_content_html += f'<div id="{exp_id}-{gender}" class="sub-tab-content {sub_active_class}">'
            
            exp_gender_df = df[(df['exp_id'] == exp_id) & (df['gender'] == gender)]
            
            for m in all_multipliers:
                sub_tab_content_html += f"<h2>Multiplier: {m}</h2>"
                chart_id = f"chart_{exp_id}_{gender}_{m}"
                sub_tab_content_html += f'<div id="{chart_id}" class="chart-container" data-exp-id="{exp_id}" data-gender="{gender}" data-multiplier="{m}"></div>'
                
                m_df = exp_gender_df[exp_gender_df['m'] == m]
                if m_df.empty:
                    sub_tab_content_html += "<p>No data for this multiplier.</p>"
                    continue
                
                pivot_table = m_df.pivot_table(index='mode', values=outcomes, aggfunc='sum')
                pivot_table_for_test = pivot_table.loc[:, (pivot_table.sum(axis=0) != 0)]
                
                p_value = None
                if not pivot_table_for_test.empty and pivot_table_for_test.shape[0] > 1 and pivot_table_for_test.shape[1] > 1:
                    try:
                        chi2, p, dof, ex = chi2_contingency(pivot_table_for_test)
                        p_value = p
                        summary_p_values[m][exp_id][gender] = p_value # Store p-value for summary
                        sub_tab_content_html += f"<p><b>Chi-squared test for significance:</b> p-value = {p:.4f}</p>"
                        
                        if p < 0.05:
                            sub_tab_content_html += "<p>✅ A statistically significant difference exists.</p>"
                            explanation_html = '<div class="explanation"><h4>Key Observations</h4><ul>'
                            for outcome_col in pivot_table_for_test.columns:
                                max_mode = pivot_table_for_test[outcome_col].idxmax()
                                max_val = pivot_table_for_test[outcome_col].max()
                                outcome_pretty = outcome_col.replace('_', ' ').capitalize()
                                mode_pretty = max_mode.replace('_', ' ').capitalize()
                                explanation_html += f"<li>For the '<b>{outcome_pretty}</b>' outcome, the <b>{mode_pretty}</b> mode had the highest count ({int(max_val)}).</li>"
                            explanation_html += '</ul></div>'
                            sub_tab_content_html += explanation_html
                        else:
                            sub_tab_content_html += "<p>&#10060; No statistically significant difference found.</p>"
                    except ValueError:
                         sub_tab_content_html += "<p><b>Chi-squared test not applicable.</b> Error during calculation.</p>"
                else:
                    sub_tab_content_html += "<p><b>Chi-squared test not applicable.</b> Not enough data or variability.</p>"

                sub_tab_content_html += "<h4>Raw Data (Counts)</h4>"
                sub_tab_content_html += pivot_table.to_html(classes='table')
            
            sub_tab_content_html += '</div>'
        
        sub_tab_buttons_html += '</div>'
        tab_content_html += sub_tab_buttons_html + sub_tab_content_html
        tab_content_html += '</div>'

    # --- NEW: Generate Summary Tab HTML ---
    summary_tab_html = '<div id="summary" class="tab-content">'
    summary_tab_html += "<h2>P-Value Summary</h2><p>This table summarizes the p-values for each test. Statistically significant results (p < 0.05) are highlighted in green. Click any cell to navigate to the detailed view.</p>"
    summary_tab_html += '<table class="summary-table"><thead>'
    # Header Row 1 (Experiments)
    summary_tab_html += '<tr><th>Multiplier</th>'
    for exp_id in experiments:
        summary_tab_html += f'<th colspan="3">{exp_id.upper()}</th>'
    summary_tab_html += '</tr>'
    # Header Row 2 (Genders)
    summary_tab_html += '<tr><th></th>'
    for exp_id in experiments:
        summary_tab_html += '<th>Male</th><th>Female</th><th>Both</th>'
    summary_tab_html += '</tr></thead><tbody>'

    # Table Body
    for m in all_multipliers:
        summary_tab_html += f'<tr><td><b>{m}</b></td>'
        for exp_id in experiments:
            for gender in ['male', 'female', 'both']:
                p_val = summary_p_values.get(m, {}).get(exp_id, {}).get(gender)
                if p_val is not None:
                    cell_class = "significant" if p_val < 0.05 else ""
                    summary_tab_html += f'<td class="{cell_class}" onclick="navigateTo(\'{exp_id}\', \'{gender}\')">{p_val:.4f}</td>'
                else:
                    summary_tab_html += f'<td onclick="navigateTo(\'{exp_id}\', \'{gender}\')">N/A</td>'
        summary_tab_html += '</tr>'
    summary_tab_html += '</tbody></table></div>'
    
    # Add summary button and content to the main HTML
    tab_buttons_html = tab_buttons_html.replace('</div>', '<button class="tab-button" onclick="openExperimentTab(event, \'summary\')">Summary</button></div>')
    final_html_body = tab_buttons_html + tab_content_html + summary_tab_html
    
    create_html_file(final_html_body, df.to_dict('records'))

def create_html_file(body_content, chart_data):
    """Generates the final HTML report file."""
    data_json = json.dumps(chart_data)

    html_template = f"""
<!DOCTYPE html><html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analysis Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }}
        .tab-nav, .sub-tab-nav {{ display: flex; border-bottom: 2px solid #dee2e6; margin-bottom: 20px; }}
        .sub-tab-nav {{ border-bottom: 1px solid #e9ecef; margin-top: 15px; }}
        .tab-button, .sub-tab-button {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; border-bottom: 3px solid transparent; margin-bottom: -2px; }}
        .sub-tab-button {{ font-size: 15px; padding: 8px 16px; margin-bottom: -1px; }}
        .tab-button.active, .sub-tab-button.active {{ color: #007bff; border-bottom-color: #007bff; font-weight: bold; }}
        .tab-content, .sub-tab-content {{ display: none; }}
        .tab-content.active, .sub-tab-content.active {{ display: block; }}
        h2 {{ border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 40px; }}
        .table, .summary-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; margin-bottom: 40px; }}
        .table th, .table td, .summary-table th, .summary-table td {{ padding: 8px; border: 1px solid #ddd; text-align: left; }}
        .table th, .summary-table th {{ background-color: #f2f2f2; }}
        .summary-table th, .summary-table td {{ text-align: center; }}
        .summary-table td.significant {{ background-color: #d4edda; }}
        .summary-table td:not(.significant) {{ background-color: #f8f9fa; }}
        .summary-table td {{ cursor: pointer; }}
        .summary-table td:hover {{ background-color: #cce5ff; }}
        .chart-container {{ min-height: 500px; }}
        .explanation {{ background-color: #eef7ff; border-left: 4px solid #007bff; padding: 10px 20px; margin: 15px 0; }}
        .explanation h4 {{ margin-top: 0; }}
    </style>
</head>
<body><div class="container">
    <h1>Steering Analysis Report</h1>
    {body_content}
</div>
<script>
    const fullData = {data_json};
    const nNeuronsSweep = {json.dumps(STEERING_N_NEURONS_SWEEP)};
    const outcomes = ['bias_flipped_perc', 'bias_neutralized_perc', 'bias_persists_perc', 'conflicting_bias_induced_perc'];
    const outcome_names = {{
        'bias_flipped_perc': 'Bias Flipped',
        'bias_neutralized_perc': 'Bias Neutralized',
        'bias_persists_perc': 'Bias Persists',
        'conflicting_bias_induced_perc': 'Conflicting Bias'
    }};

    function openExperimentTab(evt, expId) {{
        document.querySelectorAll(".tab-content").forEach(tc => tc.classList.remove("active"));
        document.querySelectorAll(".tab-button").forEach(tb => tb.classList.remove("active"));
        document.getElementById(expId).classList.add("active");
        evt.currentTarget.classList.add("active");
        
        if (expId !== 'summary') {{
            const firstSubTab = document.querySelector(`#${{expId}} .sub-tab-button:not(.active)`);
            if (firstSubTab) {{
                firstSubTab.click();
            }} else {{
                // If the first sub-tab is already active, we still need to ensure its charts are drawn
                 const activeSubTab = document.querySelector(`#${{expId}} .sub-tab-content.active`);
                 if(activeSubTab) {{
                    drawChartsForTab(activeSubTab.id);
                 }}
            }}
        }}
    }}

    function openGenderSubTab(evt, expId, gender) {{
        const parentTab = document.getElementById(expId);
        parentTab.querySelectorAll(".sub-tab-content").forEach(tc => tc.classList.remove("active"));
        parentTab.querySelectorAll(".sub-tab-button").forEach(tb => tb.classList.remove("active"));
        parentTab.querySelector(`#${{expId}}-${{gender}}`).classList.add("active");
        evt.currentTarget.classList.add("active");
        drawChartsForTab(`${{expId}}-${{gender}}`);
    }}
    
    function navigateTo(expId, gender) {{
        // Find and click the main experiment tab
        const expTabButton = Array.from(document.querySelectorAll('.tab-button')).find(btn => btn.textContent.toLowerCase() === expId.toLowerCase());
        if (expTabButton) {{
            expTabButton.click();
            // Use a short delay to ensure the sub-tab container is visible before clicking
            setTimeout(() => {{
                const genderSubTabButton = Array.from(document.querySelectorAll(`#${{expId}} .sub-tab-button`)).find(btn => btn.textContent.toLowerCase() === gender.toLowerCase());
                if (genderSubTabButton) {{
                    genderSubTabButton.click();
                }}
            }}, 50);
        }}
    }}

    function drawChartsForTab(tabContentId) {{
        const tabContent = document.getElementById(tabContentId);
        if (!tabContent) return;
        tabContent.querySelectorAll('.chart-container').forEach(chartDiv => {{
            if (chartDiv.dataset.drawn) return;
            const {{ expId, gender, multiplier }} = chartDiv.dataset;
            drawChart(chartDiv.id, expId, gender, parseFloat(multiplier));
            chartDiv.dataset.drawn = "true";
        }});
    }}

    function drawChart(divId, expId, gender, multiplier) {{
        const chartData = fullData.filter(d => d.exp_id === expId && d.gender === gender && d.m === multiplier);
        const modes = [...new Set(chartData.map(d => d.mode))].sort();

        if (modes.length === 0) {{
            document.getElementById(divId).innerHTML = "<p>No data available for this configuration.</p>";
            return;
        }}
        
        const traces = outcomes.map(outcome => ({{
            x: modes,
            y: modes.map(mode => {{
                const modeData = chartData.filter(d => d.mode === mode);
                if (modeData.length === 0) return 0;
                const totalPerc = modeData.reduce((acc, curr) => acc + curr[outcome], 0);
                return totalPerc / modeData.length;
            }}),
            name: outcome_names[outcome],
            type: 'bar'
        }}));

        const layout = {{
            title: `Avg. Outcome by Analysis Mode for n=[${{nNeuronsSweep.join(', ')}}]`,
            barmode: 'group',
            xaxis: {{ title: 'Analysis Mode' }},
            yaxis: {{ title: 'Average Percentage (%)', range: [0, 100] }}
        }};
        Plotly.newPlot(divId, traces, layout, {{responsive: true}});
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        const firstTab = document.querySelector('.tab-button');
        if (firstTab) {{
            firstTab.click();
        }}
    }});
</script>
</body></html>
    """
    try:
        with open("analysis_report.html", "w", encoding='utf-8') as f:
            f.write(html_template)
        print("Successfully generated HTML report: analysis_report.html")
    except IOError as e:
        print(f"Error: Could not write to file analysis_report.html. Error: {e}")

if __name__ == "__main__":
    experiment_data = find_experiment_data()
    if not experiment_data:
        print("No experiment data found. Please check your 'data' directory structure.")
    else:
        print(f"Found data for experiments: {', '.join(experiment_data.keys())}")
        analyze_and_generate_report(experiment_data)