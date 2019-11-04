import pandas as pd
import os
import io
import yaml


def save_csv(df: pd.DataFrame, ticker: str, path: str):
    file_path = os.path.join(path, ticker + '.csv').replace('\\', '/')
    df.to_csv(file_path, index=False)


def write_yaml(data, file_path, encoding='uft8'):
    with io.open(file_path, 'w', encoding=encoding) as outfile:
        yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)


def read_yaml(file_path, loader=yaml.SafeLoader):
    with open(file_path, 'r') as stream:
        output = yaml.load(stream, loader)
    return output
