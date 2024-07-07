import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import logging

class Visualizer:
    def __init__(self, data):
        self.data = data

    def scatter_plot(self, x, y, hue=None):
        """
        Create a scatter plot of two variables.
        
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        :param hue: Column name for color encoding.
        """
        try:
            sns.scatterplot(x=x, y=y, data=self.data, hue=hue)
            plt.title(f"{y} vs {x}")
            plt.show()
            logging.info(f"Scatter plot for {x} vs {y} created successfully.")
        except Exception as e:
            logging.error(f"Error creating scatter plot: {e}")

    def bar_plot(self, x, y):
        """
        Create a bar plot of two variables.
        
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        """
        try:
            sns.barplot(x=x, y=y, data=self.data)
            plt.title(f"{y} vs {x}")
            plt.show()
            logging.info(f"Bar plot for {x} vs {y} created successfully.")
        except Exception as e:
            logging.error(f"Error creating bar plot: {e}")

    def hist_plot(self, x):
        """
        Create a histogram of a variable.
        
        :param x: Column name for the histogram.
        """
        try:
            sns.histplot(x=x, data=self.data)
            plt.title(f"Distribution of {x}")
            plt.show()
            logging.info(f"Histogram for {x} created successfully.")
        except Exception as e:
            logging.error(f"Error creating histogram: {e}")

    def box_plot(self, x, y):
        """
        Create a box plot of two variables.
        
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        """
        try:
            sns.boxplot(x=x, y=y, data=self.data)
            plt.title(f"{y} vs {x}")
            plt.show()
            logging.info(f"Box plot for {x} vs {y} created successfully.")
        except Exception as e:
            logging.error(f"Error creating box plot: {e}")

    def heatmap(self, annot=True, cmap='viridis'):
        """
        Create a heatmap of the correlation matrix.
        
        :param annot: Boolean flag to annotate the heatmap.
        :param cmap: Colormap to use for the heatmap.
        """
        try:
            corr = self.data.corr()
            sns.heatmap(corr, annot=annot, cmap=cmap)
            plt.title("Correlation Heatmap")
            plt.show()
            logging.info("Correlation heatmap created successfully.")
        except Exception as e:
            logging.error(f"Error creating heatmap: {e}")

    def pair_plot(self, hue=None):
        """
        Create a pair plot of the dataset.
        
        :param hue: Column name for color encoding.
        """
        try:
            sns.pairplot(self.data, hue=hue)
            plt.title("Pair Plot")
            plt.show()
            logging.info("Pair plot created successfully.")
        except Exception as e:
            logging.error(f"Error creating pair plot: {e}")

    def violin_plot(self, x, y, hue=None):
        """
        Create a violin plot of two variables.
        
        :param x: Column name for x-axis.
        :param y: Column name for y-axis.
        :param hue: Column name for color encoding.
        """
        try:
            sns.violinplot(x=x, y=y, data=self.data, hue=hue, split=True)
            plt.title(f"{y} vs {x}")
            plt.show()
            logging.info(f"Violin plot for {x} vs {y} created successfully.")
        except Exception as e:
            logging.error(f"Error creating violin plot: {e}")

    def count_plot_categorical(self, cat_var):
        """
        Display count plots for all categorical variables.
        
        :param cat_var: List of categorical variable names.
        """
        try:
            for feature in cat_var:
                sns.set(style='whitegrid')
                plt.figure(figsize=(20, 5))
                total = len(self.data)
                ax = sns.countplot(x=self.data[feature], data=self.data)
                plt.title(f"Count Plot of {feature}")
                for p in ax.patches:
                    percentage = f'{100 * p.get_height() / total:.1f}%'
                    x = p.get_x() + p.get_width() / 2
                    y = p.get_height()
                    ax.annotate(percentage, (x, y), ha='center', va='center')
                plt.show()
                logging.info(f"Count plot for {feature} created successfully.")
        except Exception as e:
            logging.error(f"Error creating count plot for categorical variables: {e}")

    def density_plot_numerical(self, num_var):
        """
        Display density plots for all numerical features.
        
        :param num_var: List of numerical variable names.
        """
        try:
            for feature in num_var:
                sns.distplot(self.data[feature])
                plt.xlabel(feature)
                plt.ylabel('Density')
                plt.title(f"Density Plot of {feature}")
                plt.show()
                logging.info(f"Density plot for {feature} created successfully.")
        except Exception as e:
            logging.error(f"Error creating density plot for numerical features: {e}")

    def bivariate_analysis(self, num_var, target_var):
        """
        Perform bivariate analysis using FacetGrid and box plots.
        
        :param num_var: List of numerical variable names.
        :param target_var: Target variable name.
        """
        try:
            sns.FacetGrid(self.data, hue=target_var, height=7).map(sns.distplot, num_var[0]).add_legend()
            plt.title(f'{target_var} vs {num_var[0]}')
            plt.show()
            logging.info(f"Bivariate analysis plot for {target_var} vs {num_var[0]} created successfully.")
            
            for feature in num_var:
                if feature != target_var:
                    sns.boxplot(x=target_var, y=feature, data=self.data)
                    plt.xlabel(target_var)
                    plt.ylabel(feature)
                    plt.title(f"{feature} vs {target_var}")
                    plt.show()
                    logging.info(f"Box plot for {target_var} vs {feature} created successfully.")
        except Exception as e:
            logging.error(f"Error creating bivariate analysis plots: {e}")

    def class_imbalance_plot(self, target_var):
        """
        Show the class imbalance using a count plot.
        
        :param target_var: Target variable name.
        """
        try:
            sns.countplot(x=target_var, data=self.data)
            plt.title(f"Class Imbalance in {target_var}")
            plt.show()
            logging.info(f"Class imbalance plot for {target_var} created successfully.")
        except Exception as e:
            logging.error(f"Error creating class imbalance plot: {e}")

