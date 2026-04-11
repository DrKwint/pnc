import os
import json
import re
import glob

def parse_filename(filename):
    # Example: pnc_s3b1_k10_n64_ps0.01-0.05-0.1-0.5-1.0-10.0-100.0_e100_subsetsize1024_seed0.json
    # or pnc_s3b1_k2_n32_ps0.01-0.05-0.1-0.5_e100_subsetsize256_seed0.json
    pattern = r"pnc_(?P<config>\w+)_k(?P<k>\d+)_n(?P<n>\d+)_ps(?P<ps_range>[\d\.\-]+)_e(?P<epochs>\d+)_subsetsize(?P<subsetsize>\d+)_seed(?P<seed>\d+)\.json"
    match = re.match(pattern, filename)
    if match:
        d = match.groupdict()
        # Convert numeric values
        d['k'] = int(d['k'])
        d['n'] = int(d['n'])
        d['epochs'] = int(d['epochs'])
        d['subsetsize'] = int(d['subsetsize'])
        d['seed'] = int(d['seed'])
        return d
    return None

def compile_results(results_dir, output_file):
    json_files = glob.glob(os.path.join(results_dir, "*.json"))
    all_records = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        params = parse_filename(filename)
        
        if params is None:
            print(f"Skipping file with unrecognized format: {filename}")
            continue
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            for ps_val, metrics in data.items():
                record = params.copy()
                record['perturb_scale'] = float(ps_val)
                record.update(metrics)
                all_records.append(record)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    with open(output_file, 'w') as f:
        json.dump(all_records, f, indent=2)
    
    print(f"Compiled {len(all_records)} records from {len(json_files)} files into {output_file}")

if __name__ == "__main__":
    results_dir = "/home/elean/pnc/results/cifar10"
    output_file = "/home/elean/pnc/results/cifar10_compiled.json"
    compile_results(results_dir, output_file)
