import matplotlib.pyplot as plt
import seaborn as sns


def plot_data_distribution(data, columns):
    data[columns].hist(bins=30, figsize=(15, 10))
    plt.show()


def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.show()
