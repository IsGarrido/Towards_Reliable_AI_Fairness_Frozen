import os
import json
import re
from collections import defaultdict, Counter

STEERING_N_NEURONS_SWEEP = [1, 3, 5, 10]

def find_experiment_data():
    """
    Scans the 'data/' directory to find and parse all steering summary and report files.
    """
    all_experiments_data = defaultdict(lambda: defaultdict(dict))
    data_root = 'data'
    if not os.path.isdir(data_root):
        print(f"Error: The '{data_root}' directory was not found.")

    for root, dirs, files in os.walk(data_root):
        if 'steering' in dirs and os.path.basename(root).startswith("attribution_"):
            exp_id = os.path.basename(root).split('_')[-1]
            steering_path = os.path.join(root, 'steering')
            analysis_modes = [d for d in os.listdir(steering_path) if os.path.isdir(os.path.join(steering_path, d))]

            for mode in analysis_modes:
                summary_path = os.path.join(steering_path, mode, 'steering_summary.json')
                report_path = os.path.join(steering_path, mode, 'steering_sentence_report.json')
                
                exp_mode_data = {}
                
                if os.path.exists(summary_path):
                    try:
                        with open(summary_path, 'r') as f:
                            raw_data = json.load(f)
                        exp_mode_data['summary'] = parse_and_normalize_data(raw_data)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read or parse {summary_path}. Error: {e}")

                if os.path.exists(report_path):
                    try:
                        with open(report_path, 'r') as f:
                            exp_mode_data['sentences'] = json.load(f)
                    except (json.JSONDecodeError, IOError) as e:
                        print(f"Warning: Could not read or parse {report_path}. Error: {e}")

                if exp_mode_data:
                    all_experiments_data[exp_id][mode] = exp_mode_data
                    
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
            entry[f"{outcome}_perc"] = (values.get(outcome, 0) / total) * 100

        parsed_entries.append(entry)
    return parsed_entries

def check_for_missing_data(all_data, expected_n_values):
    """
    Checks the parsed data for missing combinations and prints warnings.
    """
    print("\n--- Checking for missing data ---")
    found_missing = False
    for exp_id, modes in all_data.items():
        for mode, data in modes.items():
            data_points = data.get('summary', [])
            for n_val in expected_n_values:
                for gender in ['male', 'female']:
                    found = any(p['n'] == n_val and p['gender'] == gender for p in data_points)
                    if not found:
                        print(f"⚠️  WARNING: Missing data for Exp={exp_id}, Mode={mode}, n={n_val}, Gender={gender}")
                        found_missing = True
    if not found_missing:
        print("✅  No missing data combinations found.")
    print("---------------------------------\n")


def generate_statistical_analysis_html(exp_data):
    """Generates the HTML for the sentence consistency analysis section."""
    html = ['<div class="analysis-section"><h2>Sentence Consistency Analysis</h2>']
    
    for gender in ['male', 'female']:
        html.append(f'<div class="gender-analysis"><h3>{gender.capitalize()} Sentences</h3>')
        
        outcome_order = {
            'bias_persists': 'Bias Persists',
            'conflicting_bias_induced': 'Conflicting Bias',
            'bias_flipped': 'Bias Flipped',
            'bias_neutralized': 'Bias Neutralized'
        }

        for outcome_key, outcome_name in outcome_order.items():
            all_sentences = []
            for mode_data in exp_data.values():
                sentence_report = mode_data.get('sentences', {})
                if not sentence_report:
                    continue
                
                for key, sentences_by_outcome in sentence_report.items():
                    if f'debias_{gender}_sentences' in key:
                        all_sentences.extend(sentences_by_outcome.get(outcome_key, []))
            
            html.append(f'<h4>{outcome_name}</h4>')
            if not all_sentences:
                html.append('<p>No sentences found for this outcome.</p>')
                continue

            sentence_counts = Counter(all_sentences)
            sorted_sentences = sorted(sentence_counts.items(), key=lambda item: item[1], reverse=True)
            
            html.append('<table class="analysis-table"><tr><th>Frequency</th><th>Sentence</th></tr>')
            for sentence, count in sorted_sentences:
                html.append(f'<tr><td class="freq-cell">{count}</td><td>{sentence}</td></tr>')
            html.append('</table>')

        html.append('</div>')
    html.append('</div>')
    return ''.join(html)

def generate_html_body_content(data, n_neurons_sweep):
    """Generates the main HTML content for tabs, charts, and analysis."""
    tab_buttons = []
    tab_contents = []

    for i, exp_id in enumerate(data.keys()):
        active_class = "active" if i == 0 else ""
        tab_buttons.append(
            f'<button class="tab-button {active_class}" onclick="openTab(event, \'{exp_id}\')">{exp_id.upper()}</button>'
        )

        analysis_modes = list(data[exp_id].keys())
        global_controls = "".join(
            f'<button class="control-button global-mode-button" data-exp="{exp_id}" data-mode="{mode}" onclick="updateAllChartsInTab(\'{exp_id}\', \'{mode}\')">{mode}</button>'
            for mode in analysis_modes
        )

        male_charts = ""
        female_charts = ""
        for n_val in n_neurons_sweep:
            chart_buttons_male = "".join(
                f'<button class="control-button chart-mode-button" data-exp="{exp_id}" data-gender="male" data-n="{n_val}" data-mode="{mode}" onclick="updateChart(\'{exp_id}\', \'male\', {n_val}, \'{mode}\')">{mode}</button>'
                for mode in analysis_modes
            )
            male_charts += f'<div class="chart-container"><div id="chart_{exp_id}_male_{n_val}"></div><div class="controls">{chart_buttons_male}</div></div>'

            chart_buttons_female = "".join(
                f'<button class="control-button chart-mode-button" data-exp="{exp_id}" data-gender="female" data-n="{n_val}" data-mode="{mode}" onclick="updateChart(\'{exp_id}\', \'female\', {n_val}, \'{mode}\')">{mode}</button>'
                for mode in analysis_modes
            )
            female_charts += f'<div class="chart-container"><div id="chart_{exp_id}_female_{n_val}"></div><div class="controls">{chart_buttons_female}</div></div>'
        
        analysis_html = generate_statistical_analysis_html(data[exp_id])

        tab_contents.append(f'''
        <div id="{exp_id}" class="tab-content {active_class}">
            <div class="global-controls">
                <label>Change all charts in this tab to Analysis Mode:</label>
                {global_controls}
            </div>
            <div class="gender-section"><h2>Male Sentences</h2><div class="chart-grid">{male_charts}</div></div>
            <div class="gender-section"><h2>Female Sentences</h2><div class="chart-grid">{female_charts}</div></div>
            <hr class="section-divider">
            {analysis_html}
        </div>
        ''')

    return f'<div class="tab-nav">{"".join(tab_buttons)}</div>{"".join(tab_contents)}'


def create_html_report(data, n_neurons_sweep):
    """Generates the final HTML report file."""
    if not data:
        return

    data_json = json.dumps({exp: {mode: val.get('summary') for mode, val in modes.items()} for exp, modes in data.items()}, indent=4)
    body_content = generate_html_body_content(data, n_neurons_sweep)

    html_template = f"""
<!DOCTYPE html><html lang="en">
<head>
    <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Steering Experiments Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #333; background-color: #f8f9fa; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: auto; background: #fff; padding: 25px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .tab-nav {{ display: flex; border-bottom: 2px solid #dee2e6; margin-bottom: 20px; }}
        .tab-button {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; border-bottom: 2px solid transparent; }}
        .tab-button.active {{ color: #007bff; border-bottom-color: #007bff; font-weight: bold; }}
        .tab-content {{ display: none; }} .tab-content.active {{ display: block; }}
        .gender-section {{ margin-top: 30px; border-top: 1px solid #eee; padding-top: 20px; }}
        h1 {{ color: #343a40; }} h2 {{ color: #495057; border-bottom: 1px solid #eee; padding-bottom: 10px; }}
        h3 {{ color: #495057; }} h4 {{ color: #6c757d; }}
        .chart-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 30px; }}
        .chart-container {{ padding: 15px; border-radius: 5px; box-shadow: 0 1px 5px rgba(0,0,0,0.05); }}
        .controls {{ margin-top: 15px; text-align: center; }}
        .control-button {{ background-color: #6c757d; color: white; border: none; padding: 8px 15px; border-radius: 4px; cursor: pointer; font-size: 14px; margin: 0 5px; }}
        .control-button.active {{ background-color: #007bff; }}
        .global-controls {{ margin-bottom: 20px; padding: 15px; background-color: #e9ecef; border-radius: 5px; }}
        .global-controls label {{ font-weight: bold; margin-right: 10px; }}
        .section-divider {{ border: 0; height: 1px; background-color: #dee2e6; margin: 40px 0; }}
        .analysis-section h2 {{ border-bottom: 2px solid #343a40; }}
        .gender-analysis {{ margin-bottom: 30px; }}
        .analysis-table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 14px; }}
        .analysis-table th, .analysis-table td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        .analysis-table th {{ background-color: #f2f2f2; font-weight: bold; }}
        .analysis-table .freq-cell {{ font-weight: bold; text-align: center; width: 100px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>Steering Vector Analysis Report</h1>
    {body_content}
</div>
<script>
    const fullData = {data_json};
    const nNeuronsSweep = {n_neurons_sweep};
    const OUTCOME_ORDER = [
        {{ 'key': 'bias_flipped_perc', 'name': 'Bias Flipped', 'color': '#28a745' }},
        {{ 'key': 'bias_neutralized_perc', 'name': 'Bias Neutralized', 'color': '#82d99c' }},
        {{ 'key': 'bias_persists_perc', 'name': 'Bias Persists', 'color': '#dc3545' }},
        {{ 'key': 'conflicting_bias_induced_perc', 'name': 'Conflicting Bias', 'color': '#ffc107' }}
    ];

    function drawChart(experimentId, gender, nValue, analysisMode) {{
        const containerId = `chart_${{experimentId}}_${{gender}}_${{nValue}}`;
        const chartData = fullData[experimentId]?.[analysisMode];
        if (!chartData) {{
            document.getElementById(containerId).innerHTML = `<p>No data for Mode: <strong>${{analysisMode}}</strong></p>`;
            return;
        }}
        let filteredData = chartData.filter(d => d.gender === gender && d.n === nValue).sort((a, b) => a.m - b.m);
        
        if (filteredData.length === 0) {{
             document.getElementById(containerId).innerHTML = `<p style="text-align:center; padding: 20px;">No data found for n=${{nValue}}.</p>`;
             Plotly.purge(containerId);
             return;
        }}

        const xValues = filteredData.map(d => d.m);
        const traces = OUTCOME_ORDER.map(outcome => ({{
            x: xValues, y: filteredData.map(d => d[outcome.key]),
            name: outcome.name, type: 'bar', marker: {{ color: outcome.color }}
        }}));
        const layout = {{
            title: `Effect of Multiplier (m) for n=${{nValue}} Neurons`, barmode: 'stack',
            xaxis: {{ title: 'Multiplier (m)', type: 'category' }},
            yaxis: {{ title: 'Outcome Percentage (%)', ticksuffix: '%', range: [0, 100] }},
            legend: {{ orientation: 'h', y: -0.3, x: 0.5, xanchor: 'center' }},
            margin: {{ b: 120 }}
        }};
        Plotly.newPlot(containerId, traces, layout, {{responsive: true}});
    }}

    function setActiveButton(button) {{
        if (!button) return;
        button.parentElement.querySelectorAll('.control-button').forEach(btn => btn.classList.remove('active'));
        button.classList.add('active');
    }}

    function updateChart(experimentId, gender, nValue, analysisMode) {{
        drawChart(experimentId, gender, nValue, analysisMode);
        const selector = `.chart-mode-button[data-exp='${{experimentId}}'][data-gender='${{gender}}'][data-n='${{nValue}}'][data-mode='${{analysisMode}}']`;
        setActiveButton(document.querySelector(selector));
    }}

    function updateAllChartsInTab(experimentId, analysisMode) {{
        document.querySelectorAll(`#${{experimentId}} .chart-mode-button[data-mode='${{analysisMode}}']`).forEach(button => {{
            updateChart(button.dataset.exp, button.dataset.gender, parseInt(button.dataset.n), button.dataset.mode);
        }});
        const selector = `.global-mode-button[data-exp='${{experimentId}}'][data-mode='${{analysisMode}}']`;
        setActiveButton(document.querySelector(selector));
    }}
    
    function openTab(evt, experimentId) {{
        document.querySelectorAll(".tab-content").forEach(tc => tc.style.display = "none");
        document.querySelectorAll(".tab-button").forEach(tb => tb.classList.remove("active"));
        document.getElementById(experimentId).style.display = "block";
        evt.currentTarget.classList.add("active");
    }}

    document.addEventListener('DOMContentLoaded', () => {{
        const experiments = Object.keys(fullData);
        if (experiments.length === 0) return;
        document.querySelector('.tab-content.active').style.display = 'block';
        experiments.forEach(expId => {{
            const analysisModes = fullData[expId] ? Object.keys(fullData[expId]) : [];
            if (analysisModes.length === 0) return;
            const initialMode = analysisModes[0];
            
            document.querySelector(`.global-mode-button[data-exp='${{expId}}'][data-mode='${{initialMode}}']`)?.classList.add('active');
            ['male', 'female'].forEach(gender => {{
                nNeuronsSweep.forEach(nValue => {{
                    drawChart(expId, gender, nValue, initialMode);
                    document.querySelector(`.chart-mode-button[data-exp='${{expId}}'][data-gender='${{gender}}'][data-n='${{nValue}}'][data-mode='${{initialMode}}']`)?.classList.add('active');
                }});
            }});
        }});
    }});
</script>
</body></html>
    """
    
    try:
        with open("steering_report.html", "w") as f:
            f.write(html_template)
        print("Successfully generated HTML report: steering_report.html")
    except IOError as e:
        print(f"Error: Could not write to file steering_report.html. Error: {e}")

if __name__ == "__main__":
    experiment_data = find_experiment_data()
    if not experiment_data:
        print("No experiment data found. Please check your 'data' directory structure.")
    else:
        print(f"Found data for experiments: {', '.join(experiment_data.keys())}")
        check_for_missing_data(experiment_data, STEERING_N_NEURONS_SWEEP)
        create_html_report(experiment_data, STEERING_N_NEURONS_SWEEP)