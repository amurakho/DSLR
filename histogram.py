import matplotlib.pyplot as plt
import pandas as pd
import sys
import seaborn as sns

# 2.1
# Ancient runes
def plot_hist(data):
    houses = data['Hogwarts House'].unique()
    numeric = ['float16', 'float32', 'float64']
    courses = df.select_dtypes(numeric).columns.values
    fig, axs = plt.subplots(ncols=4, nrows=4, figsize=(14, 12))
    plt.subplots_adjust(bottom=0.05, wspace=0.3)
    color = ['r', 'g', 'b', 'y']
    for i, course in enumerate(courses):
        row = (i // 4)
        col = i % 4
        for j, house in enumerate(houses):
            group_data = data.loc[data['Hogwarts House'] == house][course].dropna()
            sns.distplot(group_data, ax=axs[row, col], color=color[j])
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
        plot_hist(df)
