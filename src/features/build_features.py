import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop = stopwords.words('english')

def clean_stopwords(data, input_columnm, output_column=None):
    """data: Pandas dataframe
       input_column: Column select to be cleaned
       output_column: Output column of cleaned column
    """

    # If output_column == None, then use the same column to put results
    if output_column is None:
        output_column = input_columnm

    # Remove rows with NA in any column
    data = data.dropna(0,how='any')

    # Remove numbers and punctuation
    data[output_column] = data[input_columnm].apply(lambda x: re.sub(r'[^\w\s]|\d+','',x))
    
    # Tokenization
    data[output_column] = data[output_column].apply(word_tokenize)

    # Remove stop words
    data[output_column] = data[output_column].apply(lambda x: [item.lower() for item in x if item not in stop])

    return data

def add_word_to_dict(token,word_dict):
    index = len(word_dict) + 1
    word_dict[token] = index
    return index


def indexing_words(data, word_dict, input_column, output_column=None):
    """data: Pandas dataframe
       input_column: Column select to be cleaned.
       output_column: Output column of cleaned column
       word_dict: A dictonary contaning word as key and index as value

       * Column needs to be a list of tokens!
    """

    # If output_column == None, then use the same column to put results
    if output_column is None:
        output_column = input_column


    data[input_column] = data[output_column].apply(lambda x: [word_dict[token] if token in word_dict else
                                                              add_word_to_dict(token, word_dict) for token in x])

    return data

def count_words(data, input_column, output_column=None):

    # If output_column == None, then use the same column to put results
    if output_column is None:
        output_column = input_column

    data[output_column] = data[input_column].apply(lambda x: len(x))

    return data

def padding(data,max_length, input_column, output_column=None):

    # If output_column == None, then use the same column to put results
    if output_column is None:
        output_column = input_column

    # Padding arrays with maximum size
    data[output_column] = data[input_column].apply(lambda x: np.pad(x,(0,max_length -
                                                                       len(x)),'constant',constant_values=0).tolist())

    # Important: Transform into list to save with pandas. Saving as numpy array turn hard read operation.

    return data


