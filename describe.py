"""
v.1

Yeap i use pandas and more "heavy" func
I am doing a project for more practice
And want to use lib
"""

import sys
import pandas as pd
import numpy as np

def functions(column):
    column = np.sort(column)

    count = len(column)

    mean = np.round(np.sum(column) / count, decimals=6)

    std = (column - mean)**2
    std = np.round(np.sqrt(np.sum(std) / count), decimals=6)

    min_ = np.min(column)

    pr25 = column[int(np.shape(column)[0] * 0.25)]
    pr50 = column[int(np.shape(column)[0] * 0.5)]
    pr75 = column[int(np.shape(column)[0] * 0.75)]

    max_ = np.max(column)
    return count, mean, std, min_, pr25, pr50, pr75, max_

def make_funx(data):
    desc = []
    desc.append(['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max'])
    for column in data:
        desc.append(functions(data[column]))
    labels = list(data.columns.values)
    labels.insert(0, 'Func')
    desc = np.array(desc)
    desc = desc.T
    new_df = pd.DataFrame.from_records(desc, columns=labels)
    print(new_df)

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit(1)
    file_name = sys.argv[1]
    try:
        df = pd.read_csv(file_name)
    except:
        print("Error (file name)")
    else:
        numeric = ['float16', 'float32', 'float64']
        clean_data = df.select_dtypes(numeric).dropna(axis=1, how='all').dropna()
        make_funx(clean_data)

