import pandas
import pandas as pd


def make_list(s):
    if type(s) == list:
        return s
    else:
        return [s]
    
    
    
def from_df_to_list_of_dict(l1, l2=None):
    """
    Transforming dataframe into list of dict (if necessary)
    """
    if type(l1) == pandas.core.frame.DataFrame:
        l1 = l1.to_dict('records')
    if l2 is None:
        return l1
    if type(l2) == pandas.core.frame.DataFrame:
        l2 = l2.to_dict('records')
    return l1, l2

def from_df_to_numpy(df1, df2=None):
    """
    Transforming dataframe into list of dict (if necessary)
    """
    if type(df1) == pandas.core.frame.DataFrame:
        df1 = df1.to_numpy()
    if df2 is None:
        return df1
    if type(df2) == pandas.core.frame.DataFrame:
        df2 = df2.to_numpy()
    return df1, df2


def from_list_of_dict_to_df(l1, l2=None):
    """
    Transforming list of dict into dataframe (if necessary)
    """
    if type(l1) == list:
        l1 = pd.DataFrame(l1)
    if l2 is None:
        return l1
    if type(l2) == list:
        l2 = pd.DataFrame(l2)
    return l1, l2