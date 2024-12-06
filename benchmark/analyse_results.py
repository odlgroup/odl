import argparse
import sys

import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file_name', required = True)
    args = parser.parse_args()

    result_file_name = args.result_file_name

    ### unpack benchmark metadata
    try:
        df = pd.read_csv(f'results/{result_file_name}.csv')
    except FileNotFoundError:
        sys.exit(f'No file at results/{result_file_name}.csv')

    if result_file_name in ['backward', 'forward']:
        average_df = df[df['device_name'] != 'cpu']
        average_df = average_df.groupby([
            'dimension',
            'n_points',
            'reco_space_impl', 
            'ray_trafo_impl', 
            'device_name'
            ])['time'].mean().reset_index()

        print(average_df)
    elif result_file_name in ['pytorch_wrapper_forward', 'pytorch_wrapper_backward']:
        average_df = df.groupby([
            'dimension',
            'n_points',
            'operator',
            'reco_space_impl', 
            'ray_trafo_impl', 
            'device_name'
            ])['time'].mean().reset_index()

        print(average_df)
    else:
        raise NotImplementedError
