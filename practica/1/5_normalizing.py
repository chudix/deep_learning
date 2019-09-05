import pandas as pd
import matplotlib.pyplot as plt

def min_max_normalization(a_dataframe):
    """
    Normalizes variables in dataset using min max method

    This function takes a normalizable dataframe, numeric data and
    calculates the normalization of each column.
    This method is calculated as:
    minmax = x - min(x) / (max(x)-min(x))

    Parameters
    ----------
    a_dataframe: DataFrame object of numeric data.

    Returns
    ----------
    The normalizated dataframe
    
    Raises
    ----------
    ValueError if dataframe has any non numeric column
    """
    
    normalized = (a_dataframe - a_dataframe.min())/(a_dataframe.max() - a_dataframe.min())

    return normalized

def z_score_normalization(a_dataframe):
    """
    Normalizes a dataframe using z-score normalization

    Takes a dataframe an calculates de mean normalization of each
    value.
    z-score normalization is defined as:
    z-score(x) = (x - mean(x)) / std(x)

    Parameters
    ----------
    a_dataframe: DataFrame object to normalize
    
    Returns
    ----------
    normalizated dataframe
    """

    normalized = (a_dataframe - a_dataframe.mean()) / a_dataframe.std()

    return normalized


# read data
dataframe = pd.read_csv('iris.csv')

# sanitize dataframe(get only numeric columns)
sanitized_dataframe = dataframe.select_dtypes(include=float)

# Normalize variables and make hist
min_max_dataframe = min_max_normalization(sanitized_dataframe)
min_max_dataframe.hist()

# Normalize using mean normalization and make hist
z_score_dataframe = z_score_normalization(sanitized_dataframe)
z_score_dataframe.hist()

# show plots
plt.show()
