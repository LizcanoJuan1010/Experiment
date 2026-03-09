import json

file_path = '/home/juan/Documents/TESIS CODIGO/results/aphasia_test_results.json'

def check_baselines_for_collapse():
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = data['results']
    
    # Baselines are stored in a separate dict at the top level
    baselines = data.get('baselines', {})
    print("--- Baselines Defined in 'baselines' section ---")
    print(json.dumps(baselines, indent=2))

    print("\n--- Checking specific result entries for baseline collapse ---")
    # Let's look for known collapse patterns in the *baseline* field of the results
    # and compare with ablated
    
    collapse_suspects = []
    
    for i, r in enumerate(results):
        # Check for repetition in baseline
        baseline_text = r.get('baseline', '')
        ablated_text = r.get('ablated', '')
        
        # Simple heuristic for collapse: repeated substrings of length > 10
        # or just visual inspection of the first few
        
        # We will print out cases where baseline looks suspicious or just print a few random ones
        # specifically those where ablated failed hard.
        
        # Let's check the specific examples I cited before:
        # Alpha 5.0, Place, Subtraction (Example 166 in my previous view)
        # Alpha 5.0, Place, Projection (Example 391)
        
        if r['category'] == 'place' and r['alpha'] == 5.0 and r['technique'] == 'projection':
             print(f"\n[Index {i}] Technique: {r['technique']}, Alpha: {r['alpha']}, Category: {r['category']}")
             print(f"PROMPT: {r['prompt']}")
             print(f"BASELINE: {json.dumps(baseline_text)}")
             print(f"ABLATED:  {json.dumps(ablated_text)}")

        if r['category'] == 'time' and r['alpha'] == 8.0 and r['technique'] == 'projection':
             print(f"\n[Index {i}] Technique: {r['technique']}, Alpha: {r['alpha']}, Category: {r['category']}")
             print(f"PROMPT: {r['prompt']}")
             print(f"BASELINE: {json.dumps(baseline_text)}")
             print(f"ABLATED:  {json.dumps(ablated_text)}")

        # Check for repetition in baseline specifically
        # "The man was dead. The man was dead." seems to be in baseline for tools
        if "dead.\n\nThe man was dead" in baseline_text:
             if i < 50: # Just print one early warning
                 print(f"\n[Index {i}] FOUND REPETITION IN BASELINE (Category: {r['category']})")
                 print(f"BASELINE: {json.dumps(baseline_text)}")

if __name__ == "__main__":
    check_baselines_for_collapse()
