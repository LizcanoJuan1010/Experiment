import json
import pandas as pd
from collections import defaultdict

file_path = '/home/juan/Documents/TESIS CODIGO/results/aphasia_test_results.json'

def analyze_results():
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    config = data['config']
    
    print(f"Total results: {len(results)}")
    
    # Metrics to track
    # Change rate by category
    category_changes = defaultdict(lambda: {'total': 0, 'changed': 0})
    
    # Change rate by method and technique
    method_technique_changes = defaultdict(lambda: {'total': 0, 'changed': 0})
    
    # Change rate by layer
    layer_changes = defaultdict(lambda: {'total': 0, 'changed': 0})
    
    # Change rate by alpha
    alpha_changes = defaultdict(lambda: {'total': 0, 'changed': 0})

    # Specific analysis for target vs control
    # Target: 'time', Control: 'place', 'tools'
    target_category = config['ablated_concept']
    
    detailed_stats = []

    for r in results:
        cat = r['category']
        changed = r['changed']
        method = r['method']
        technique = r['technique']
        layer = r['layer_label']
        alpha = r['alpha']
        
        # Category stats
        category_changes[cat]['total'] += 1
        if changed:
            category_changes[cat]['changed'] += 1
            
        # Method/Technique stats
        mt_key = f"{method}_{technique}"
        method_technique_changes[mt_key]['total'] += 1
        if changed:
            method_technique_changes[mt_key]['changed'] += 1

        # Layer stats
        layer_changes[layer]['total'] += 1
        if changed:
            layer_changes[layer]['changed'] += 1
            
        # Alpha stats
        alpha_changes[alpha]['total'] += 1
        if changed:
            alpha_changes[alpha]['changed'] += 1
            
        detailed_stats.append({
            'method': method,
            'technique': technique,
            'layer': layer,
            'alpha': alpha,
            'category': cat,
            'changed': changed,
            'role': r['role']
        })

    df = pd.DataFrame(detailed_stats)
    
    print("\n--- Summary by Category ---")
    for cat, stats in category_changes.items():
        percent = (stats['changed'] / stats['total']) * 100
        role = "Target" if cat == target_category else "Control"
        print(f"Category: {cat} ({role}) - Changed: {stats['changed']}/{stats['total']} ({percent:.2f}%)")

    print("\n--- Summary by Method & Technique ---")
    for mt, stats in method_technique_changes.items():
        percent = (stats['changed'] / stats['total']) * 100
        print(f"Method_Technique: {mt} - Changed: {stats['changed']}/{stats['total']} ({percent:.2f}%)")
        
    print("\n--- Summary by Layer ---")
    for layer, stats in layer_changes.items():
        percent = (stats['changed'] / stats['total']) * 100
        print(f"Layer: {layer} - Changed: {stats['changed']}/{stats['total']} ({percent:.2f}%)")

    print("\n--- Summary by Alpha ---")
    for alpha, stats in sorted(alpha_changes.items()):
        percent = (stats['changed'] / stats['total']) * 100
        print(f"Alpha: {alpha} - Changed: {stats['changed']}/{stats['total']} ({percent:.2f}%)")

    # Deeper dive: Subtraction on Target vs Control at different alphas
    print("\n--- Subtraction: Target vs Control Change Rate by Alpha ---")
    subtraction_df = df[df['technique'] == 'subtraction']
    for alpha in sorted(subtraction_df['alpha'].unique()):
        alpha_df = subtraction_df[subtraction_df['alpha'] == alpha]
        target_change = alpha_df[alpha_df['role'] == 'target']['changed'].mean() * 100
        control_change = alpha_df[alpha_df['role'] == 'control']['changed'].mean() * 100
        print(f"Alpha {alpha}: Target Changed {target_change:.2f}%, Control Changed {control_change:.2f}%")

    # Deeper dive: Projection on Target vs Control at different alphas
    print("\n--- Projection: Target vs Control Change Rate by Alpha ---")
    projection_df = df[df['technique'] == 'projection']
    for alpha in sorted(projection_df['alpha'].unique()):
        alpha_df = projection_df[projection_df['alpha'] == alpha]
        target_change = alpha_df[alpha_df['role'] == 'target']['changed'].mean() * 100
        control_change = alpha_df[alpha_df['role'] == 'control']['changed'].mean() * 100
        print(f"Alpha {alpha}: Target Changed {target_change:.2f}%, Control Changed {control_change:.2f}%")

if __name__ == "__main__":
    analyze_results()
