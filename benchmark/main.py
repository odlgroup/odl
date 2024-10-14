"""This module benchmarks a function with parameters defined in a json file metadata"""
import argparse
from pathlib import Path
from datetime import datetime
import time
import sys

import json
import pandas as pd

N_CALLS = 1
MAX_ITERATIONS = 100

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
    for key in ['backend', 'script_name', 'parameters']:
        try:
            benchmark_dict[key] = metadata_dict[key]
        except ValueError:
            sys.exit(f'No "{key}" key in the metadata_dict')

    ### load backend module
    if benchmark_dict['backend'] == 'odl':
        import scripts.odl_scripts as sc
        DEVICE = 'cpu'

    elif benchmark_dict['backend'] == 'torch':
        import scripts.torch_scripts as sc
        try:
            DEVICE = metadata_dict['parameters']['device_name']
        except ValueError:
            DEVICE = 'cpu'

    else:
        raise NotImplementedError(f'''Backend {benchmark_dict["backend"]} not supported, only
                                  "odl" and "torch"''')

    try:
        function = getattr(sc, benchmark_dict['script_name'])
    except AttributeError:
        sys.exit(f'''Script {benchmark_dict["script_name"]} not implemented for backend
              {benchmark_dict["backend"]}''')


    report_dict = {
        "dimension" : [],
        "n_points"  : [],
        "time"      : [],
        "error"     : []
    }

    for dimension in benchmark_dict["parameters"]['dimensions']:
        for n_points in benchmark_dict["parameters"]['n_points']:
            print(
                f"""Benchmarking {benchmark_dict['script_name']}
                  for dimension {dimension} and {n_points} points"""
                )            
            for call in range(N_CALLS):
                start = time.time()
                error = function(
                    benchmark_dict["parameters"],
                    dimension, n_points, 
                    MAX_ITERATIONS
                )
                end = time.time()
                report_dict['dimension'].append(dimension)
                report_dict['n_points'].append(n_points)
                report_dict['time'].append(end - start) 
                report_dict['error'].append(error)

    report_df = pd.DataFrame.from_dict(report_dict)
    report_df['device']  = DEVICE
    report_df['backend'] = benchmark_dict['backend']
    report_df['max_iterations'] = MAX_ITERATIONS
    report_df['timestamp'] = pd.Timestamp(datetime.now(), tz=None)
    result_file_path = f'results/{metadata_name}.csv'
    if Path(result_file_path).is_file():
        report_df = pd.concat([
            pd.read_csv(result_file_path), report_df
        ])
    report_df.to_csv(f'results/{metadata_name}.csv', index = False)
