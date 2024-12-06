"""This module benchmarks a function with parameters defined in a json file metadata"""
import argparse
from pathlib import Path
from datetime import datetime
import sys
from typing import Dict 

from tqdm import tqdm
import json
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata_name', required = True)
    args = parser.parse_args()

    metadata_name = args.metadata_name

    ### unpack benchmark metadata
    try:
        with open(f'metadata/{metadata_name}.json', mode ='r', encoding="utf-8") as json_file:
            metadata_dict = json.load(json_file)
    except FileNotFoundError:
        sys.exit(f'No file at metadata/{metadata_name}.json')

    ### unpack variables
    benchmark_dict = {}
    for key in ['backend', 'script_name', 'parameters', 'n_calls']:
        try:
            benchmark_dict[key] = metadata_dict[key]
        except ValueError:
            sys.exit(f'No "{key}" key in the metadata_dict')

    ### load backend module
    if benchmark_dict['backend'] == 'odl':
        import scripts.odl_scripts as sc

    elif benchmark_dict['backend'] == 'torch':
        import scripts.torch_scripts as sc

    else:
        raise NotImplementedError(f'''Backend {benchmark_dict["backend"]} not supported, only
                                  "odl" and "torch"''')
    
    try:
        DEVICE = metadata_dict['parameters']['device_name']
    except ValueError:
        DEVICE = 'cpu'

    try:
        function = getattr(sc, benchmark_dict['script_name'])
    except AttributeError:
        sys.exit(f'''Script {benchmark_dict["script_name"]} not implemented for backend
              {benchmark_dict["backend"]}''')


    report_dict = {
    }

    for key, value in benchmark_dict["parameters"].items():
        print(key, value)
        
    for call in tqdm(range(benchmark_dict['n_calls'])):        
        return_dict:Dict = function(
            benchmark_dict["parameters"]
        )
        for k,v in return_dict.items():
            if k in report_dict:
                report_dict[k].append(v)
            else:
                report_dict[k] = [v]

    report_df = pd.DataFrame.from_dict(report_dict)
    for key, value in benchmark_dict["parameters"].items():
        report_df[key] = value
    report_df['backend'] = benchmark_dict['backend']
    report_df['timestamp'] = pd.Timestamp(datetime.now(), tz=None)
    result_file_path = f'results/{metadata_name}.csv'
    if Path(result_file_path).is_file():
        report_df = pd.concat([
            pd.read_csv(result_file_path), report_df
        ])
    report_df.to_csv(f'results/{metadata_name}.csv', index = False)
