from nltk.corpus import stopwords
stop = stopwords.words('english')

def clean_stopwords(data, input_columnm, output_column=None):
    """data: Pandas dataframe
       input_columnm: Column select to be cleaned
       output_column: Output column of cleaned column
    """

    # If output_column == None, then use the same column to put results
    if output_column is None:
        output_column = input_columnm

    # Remove rows with NA in any column
    data = data.dropna(0,how='any')

    # Remove numbers
    data[output_column] = data[input_columnm].str.replace('[^\w\s]','')

    # Remove numbers
    data[output_column] = data[output_column].str.replace('\d+', '')

    # Split and lower
    data[output_column] = data[output_column].str.lower().str.split()

    # Remove stop words
    data[output_column] = data[output_column].apply(lambda x: [item for item in x if item not in stop])

    return data
