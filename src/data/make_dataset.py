# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import pickle
import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.features.build_features import clean_stopwords, indexing_words, count_words, padding

pd.options.mode.chained_assignment = None  # default='warn'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    data = pd.read_csv(input_filepath)

    logger.info('Cleaning stopwords')
    data = clean_stopwords(data,"question1")
    data = clean_stopwords(data,"question2")
    logger.info('Cleaned stopwords')


    logger.info('Indexing words')
    word_dict = {}
    data = indexing_words(data, word_dict, "question1")
    data = indexing_words(data, word_dict, "question2")
    logger.info('Words indexed')


    # Saving dictionary using pickle
    with open('../../data/raw/word_dict.pickle', 'wb') as handle:
        pickle.dump(word_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info('Counting words')
    data = count_words(data,"question1","q1_len")
    data = count_words(data,"question2","q2_len")
    logger.info('Words counted')

    # Remove outliers question bigger than 20 and equal 0
    logger.info('Padding')
    max_length = 20
    data = data[data["q1_len"] <= max_length]
    data = data[data["q2_len"] <= max_length]
    data = data[data["q1_len"] != 0]
    data = data[data["q2_len"] != 0]

    # Padding
    logger.info("Making padding")
    data = padding(data,max_length,"question1")
    data = padding(data,max_length,"question2")
    logger.info('Padding completed')

    logger.info('Saving dataframe')
    data.to_csv(output_filepath,index=False,sep=";",encoding='utf-8')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
