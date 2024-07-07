import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder

class Preprocessor:
    def __init__(self, file_path):
        """
        Initialize the Preprocessor with a CSV file path.
        """
        try:
            self.data = pd.read_csv(file_path)
            logging.info("Data loaded successfully.")
        except FileNotFoundError as e:
            logging.error(f"File not found: {e}")
            raise
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            raise

    def get_basic_info(self):
        """
        Print basic information about the dataset.
        """
        try:
            logging.info("Displaying basic info of the dataset.")
            print(self.data.head())
            print(self.data.shape)
            print(self.data.info())
        except Exception as e:
            logging.error(f"Error displaying basic info: {e}")

    def get_unique_values(self, columns):
        """
        Print unique values and their counts for specified columns.
        
        :param columns: List of column names to analyze.
        """
        try:
            for col in columns:
                if col in self.data.columns:
                    unique_values = self.data[col].unique()
                    no_of_unique_values = self.data[col].nunique()
                    print(f"{col} unique values: {unique_values}")
                    print(f"{col} no of unique values: {no_of_unique_values}\n")
                else:
                    logging.warning(f"Column '{col}' does not exist in the dataset.")
        except Exception as e:
            logging.error(f"Error getting unique values: {e}")

    def remove_outliers(self, column, method='iqr', threshold=None):
        """
        Remove outliers from a specified column using different methods.
        
        :param column: Column name from which to remove outliers.
        :param method: Method to use for outlier detection ('iqr' or 'threshold').
        :param threshold: Threshold value for outlier removal when using 'threshold' method.
        """
        try:
            if method == 'iqr':
                Q1 = self.data[column].quantile(0.25)
                Q3 = self.data[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                self.data = self.data[(self.data[column] >= lower_bound) & (self.data[column] <= upper_bound)]
                logging.info(f"Outliers removed using IQR method for column {column}.")
            elif method == 'threshold' and threshold is not None:
                self.data = self.data[self.data[column] <= threshold]
                logging.info(f"Outliers removed using threshold method for column {column}.")
            else:
                logging.warning(f"Invalid method or threshold not provided for column {column}.")
        except Exception as e:
            logging.error(f"Error removing outliers: {e}")

    def dummify_columns(self, columns):
        """
        Convert specified columns to dummy/indicator variables.
        
        :param columns: List of column names to dummify.
        """
        try:
            self.data = pd.get_dummies(self.data, columns=columns, drop_first=True)
            logging.info(f"Columns {columns} converted to dummy variables.")
        except Exception as e:
            logging.error(f"Error dummifying columns: {e}")

    def handle_missing_values(self, strategy='mean', columns=None):
        """
        Handle missing values in the dataset.
        
        :param strategy: Strategy for handling missing values ('mean', 'median', 'mode', or 'drop').
        :param columns: List of columns to apply the strategy to. If None, applies to all columns.
        """
        try:
            if columns is None:
                columns = self.data.columns

            for col in columns:
                if strategy == 'mean':
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
                elif strategy == 'median':
                    self.data[col].fillna(self.data[col].median(), inplace=True)
                elif strategy == 'mode':
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                elif strategy == 'drop':
                    self.data.dropna(subset=[col], inplace=True)
                else:
                    logging.warning(f"Invalid strategy '{strategy}' for handling missing values.")
            logging.info(f"Missing values handled using {strategy} strategy.")
        except Exception as e:
            logging.error(f"Error handling missing values: {e}")

    def scale_features(self, columns, method='standard'):
        """
        Scale specified features using standardization or min-max scaling.
        
        :param columns: List of column names to scale.
        :param method: Scaling method ('standard' or 'minmax').
        """
        try:
            if method == 'standard':
                scaler = StandardScaler()
            elif method == 'minmax':
                scaler = MinMaxScaler()
            else:
                logging.warning(f"Invalid scaling method '{method}'.")
                return

            self.data[columns] = scaler.fit_transform(self.data[columns])
            logging.info(f"Features {columns} scaled using {method} method.")
        except Exception as e:
            logging.error(f"Error scaling features: {e}")

    def label_encode_columns(self, columns):
        """
        Label encode specified columns.
        
        :param columns: List of column names to label encode.
        """
        try:
            encoder = LabelEncoder()
            for col in columns:
                if col in self.data.columns:
                    self.data[col] = encoder.fit_transform(self.data[col])
                    logging.info(f"Column '{col}' label encoded.")
                else:
                    logging.warning(f"Column '{col}' does not exist in the dataset.")
        except Exception as e:
            logging.error(f"Error label encoding columns: {e}")

    def drop_columns(self, columns):
        """
        Drop specified columns from the dataset.
        
        :param columns: List of column names to drop.
        """
        try:
            self.data.drop(columns, axis=1, inplace=True)
            logging.info(f"Columns {columns} dropped from the dataset.")
        except Exception as e:
            logging.error(f"Error dropping columns: {e}")

    def save_processed_data(self, file_path):
        """
        Save the processed data to a CSV file.
        
        :param file_path: Path to save the processed data.
        """
        try:
            self.data.to_csv(file_path, index=False)
            logging.info(f"Processed data saved to {file_path}.")
        except Exception as e:
            logging.error(f"Error saving processed data: {e}")

    def get_numerical_columns(self):
        """
        Get a list of numerical columns in the dataset.
        
        :return: List of numerical column names.
        """
        try:
            numerical_columns = self.data.select_dtypes(include=['number']).columns.tolist()
            logging.info(f"Numerical columns identified: {numerical_columns}")
            return numerical_columns
        except Exception as e:
            logging.error(f"Error getting numerical columns: {e}")
            return []

    def get_categorical_columns(self):
        """
        Get a list of categorical columns in the dataset.
        
        :return: List of categorical column names.
        """
        try:
            categorical_columns = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
            logging.info(f"Categorical columns identified: {categorical_columns}")
            return categorical_columns
        except Exception as e:
            logging.error(f"Error getting categorical columns: {e}")
            return []

    def get_data(self):
        """
        Get the processed data.
        
        :return: Processed DataFrame.
        """
        return self.data
