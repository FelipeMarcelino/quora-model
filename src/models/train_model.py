import pandas as pd
import torch
import numpy as np
import sys
import click
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime, date
from tqdm import trange
from tqdm import tqdm
from src.models.create_dataset import QuoraDataset
from src.models.model import  QuoraModel
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

pd.options.mode.chained_assignment = None  # default='warn'


# Folders to save results and models
report_folder = "../../reports/"
report_figure_folder = "../../reports/figures/"
models_folder = "../../models/"
index_file_model = "../../reports/index.csv"


def write_to_index(dict_index):
    index_models = pd.read_csv(index_file_model,sep=";",header=0)

    # It is necessary to reoder columns before saving the index models
    cols = index_models.columns.values

    dict_index['id'] = [len(index_models)]

    # Verify if a model used cpu or gpu. Is importante to load this model later
    if 'cpu' in str(device):
        dict_index['gpu'] = 0
    else:
        dict_index['gpu'] = 1

    # Create a row with data
    idx = len(index_models)
    temp = pd.DataFrame.from_dict(dict_index)
    index_models = pd.concat([index_models, temp], axis=0,sort=True)

    # Reordeing columns
    index_models = index_models[cols]

    # Saving results
    index_models.to_csv(index_file_model,sep=";",index=False)

def test_model(test, model, criterion):
    model.eval()
    with torch.no_grad():
        correct = 0
        total  = 0
        loss_test = 0
        t = tqdm(iter(test),desc="Batchs",total=len(test))
        pred_test = []
        true_test = []
        for idx, sample in enumerate(t):
            output = model(sample[0].to(device),sample[1].to(device),sample[2].to(device),sample[3].to(device))
            loss = criterion(output[:,0],sample[4].to(device).float())
            loss_test += loss.item()
            pred = [1 if x >= 0.5 else 0 for x in  output[:,0].tolist()]
            true = sample[4].tolist()
            pred_test.extend(pred)
            true_test.extend(true)

        loss_test = loss_test / len(test)
        acc_test = accuracy_score(true_test,pred_test)
        f1 =  f1_score(true_test,pred_test)

        return  loss_test,acc_test, f1 ,pred_test,true_test

def save_model(model, optim, epoch, loss, learning_rate, model_name):

    PATH = models_folder + str(model_name) + ".pth"

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            'learning_rate': learning_rate
            }, PATH)

def train_model(train, test, model, epochs, learning_rate, model_name, date, batch_size, sample_size, test_frac):
    criterion = nn.BCELoss()
    optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    best_acc_test = 0
    best_acc_train = 0
    best_f1_score_test = 0


    for ep in trange(epochs,desc='Epochs'):
        loss_ep = 0
        model.train()
        t = trange(len(train),leave=True,desc='Batchs')
        pred_ep = []
        true_ep = []
        for idx, (sample, _) in enumerate(zip(train,t)):
            output = model(sample[0].to(device),sample[1].to(device),sample[2].to(device),sample[3].to(device))
            loss = criterion(output[:,0],sample[4].to(device).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_ep += loss.item()
            pred = [1 if x >= 0.5 else 0 for x in  output[:,0].tolist()]
            true = sample[4].tolist()
            pred_ep.extend(pred)
            true_ep.extend(true)


        acc = accuracy_score(true_ep,pred_ep)
        loss_ep  = loss_ep / len(train)

        # Valdiate model
        loss_test, acc_test, f1 ,pred_test,true_test = test_model(test, model, criterion)

        if acc > best_acc_train:
            best_acc_train = acc

        if acc_test > best_acc_test:
            best_acc_test = acc_test
            #Save model
            save_model(model, optim, ep, loss_ep, learning_rate, model_name)

        if f1 > best_f1_score_test:
            best_f1_score_test = f1

    dict_index = {}
    dict_index['model_name'] = [model_name]
    dict_index['date'] = [date]
    dict_index['best_acc_train'] = [best_acc_train]
    dict_index['best_acc_test'] = [best_acc_test]
    dict_index['loss_train(last)'] = loss_ep
    dict_index['loss_test(last)'] = [loss_test]
    dict_index['f1_score_test_best'] = [best_f1_score_test]
    dict_index['epochs'] = [epochs]
    dict_index['batch_size'] = [batch_size]
    dict_index['learning_rate'] = [learning_rate]
    dict_index['sample_size'] = [sample_size]
    dict_index['test_frac'] = [test_frac]
    write_to_index(dict_index)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
# Use dateime.now as default name for model
@click.option('--model_name',default=datetime.now().isoformat().split(".")[0], help="Model name to be saved")
@click.option('--frac_sample',default=1.0, help="Fraction of sample train used to train the model")
@click.option('--test_frac',default=0.2, help="Fraction of sample train used to test the model")
@click.option('--batch_size',default=128, help="Batch size")
@click.option('--epochs',default=10, help="Numbers of epochs to train the model")
@click.option('--learning_rate', default=0.001, help="Learning rate")
def main(input_filepath,model_name,frac_sample,test_frac,batch_size, epochs, learning_rate):


    model_name = model_name.replace(":","-")
    create_model_date = date.today().isoformat()

    data = pd.read_csv(input_filepath,header=0,sep=";")

    # Only for testing
    data = data.sample(frac=frac_sample, random_state=1)
    infile = open("../../data/raw/word_dict.pickle","rb")
    word_dict = pickle.load(infile)


    # Separating train and test
    mask = np.random.rand(len(data)) < (1 - test_frac)
    train_df = data[mask]
    test_df = data[~mask]

    # Creting dataset 
    train_dataset = QuoraDataset(train_df)
    test_dataset = QuoraDataset(test_df)

    # Create dataloader
    train = DataLoader(train_dataset,batch_size, shuffle=True)
    test = DataLoader(test_dataset,len(test_dataset), shuffle=False)
    """
        These dataloaders returns question1, q1_len, question2, q2_len and label
    """

    # Creating model
    kernel_sizes = [3,5,7,9]
    out_channels = 100
    stride = 2
    hidden_size = 300
    embedding_dim = 300
    layers = 2
    output_dim_fc1 = 150
    vocab_size = len(word_dict)

    model = QuoraModel(kernel_sizes,out_channels,stride,hidden_size,layers,embedding_dim,output_dim_fc1,
                       vocab_size).to(device)

    train_model(train, test, model, epochs, learning_rate, model_name, create_model_date, batch_size, len(data), test_frac)

if __name__ == "__main__":
    main()
