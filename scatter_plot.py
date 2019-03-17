import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import seaborn as sns; sns.set(style="ticks", color_codes=True)

# 2.2
# Astronomy and DADA
# 2.3
# Astronomy/Herbology
def plot_scatter(data):
    numeric = ['float16', 'float32', 'float64']
    houses = data['Hogwarts House']
    data =  data.select_dtypes(numeric).dropna(axis=1, how='all').dropna()
    sns.set(font_scale=0.5)
    data['Hogwarts House'] = houses

    g = sns.pairplot(data, hue='Hogwarts House', plot_kws={"s": 3}, height=1, markers='+')
    g.fig.subplots_adjust(wspace=.1, hspace=.1)
    plt.show()

def plot_scatter_test(data):
    data = data.dropna()
    g = sns.pairplot(data, hue='Hogwarts House',  markers='+')
    # plt.scatter(data.iloc[:, 0] , data.iloc[:,1], c=data.iloc[:, 2])
    plt.show()

if __name__ == '__main__':
    if len(sys.argv) == 1:
        exit(1)
    file_name = sys.argv[1]
    try:
        df = pd.read_csv(file_name)
    except:
        print("Error (file name)")
    else:
        plot_scatter(df)

        # For test
        # data = df[['Astronomy', 'Herbology', 'Hogwarts House']]
        # plot_scatter_test(data)
        