import pandas as pd
import torch
import numpy as np
import sys
import click
import pickle
import matplotlib.pyplot as plt
import torch.nn as nn
import math
import os
from sklearn.metrics import accuracy_score, f1_score, auc, roc_curve
from datetime import datetime, date
from tqdm import trange
from tqdm import tqdm
from src.models.create_dataset import QuoraDataset
from src.models.model import  QuoraModel
from torch.utils.data import DataLoader

" Code copied from: https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click"
class OptionEatAll(click.Option):

    def __init__(self, *args, **kwargs):
        self.save_other_options = kwargs.pop('save_other_options', True)
        nargs = kwargs.pop('nargs', -1)
        assert nargs == -1, 'nargs, if set, must be -1 not {}'.format(nargs)
        super(OptionEatAll, self).__init__(*args, **kwargs)
        self._previous_parser_process = None
        self._eat_all_parser = None

    def add_to_parser(self, parser, ctx):

        def parser_process(value, state):
            # method to hook to the parser.process
            done = False
            value = [value]
            if self.save_other_options:
                # grab everything up to the next option
                while state.rargs and not done:
                    for prefix in self._eat_all_parser.prefixes:
                        if state.rargs[0].startswith(prefix):
                            done = True
                    if not done:
                        value.append(state.rargs.pop(0))
            else:
                # grab everything remaining
                value += state.rargs
                state.rargs[:] = []
            value = tuple(value)

            # call the actual process
            self._previous_parser_process(value, state)

        retval = super(OptionEatAll, self).add_to_parser(parser, ctx)
        for name in self.opts:
            our_parser = parser._long_opt.get(name) or parser._short_opt.get(name)
            if our_parser:
                self._eat_all_parser = our_parser
                self._previous_parser_process = our_parser.process
                our_parser.process = parser_process
                break
        return retval

" Code copied from: https://stackoverflow.com/questions/48391777/nargs-equivalent-for-options-in-click"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda:0')

pd.options.mode.chained_assignment = None  # default='warn'


# Folders to save results and models
report_folder = "../../reports/"
report_figure_folder = "../../reports/figures/"
models_folder = "../../models/"
temp_index_models = "../../reports/index.csv"

# Columns for reports files
report_columns = ['epoch','acc','f1_score','loss','auc']
index_columns = \
['id','model_name','date','loss_train(last)','loss_test(last)','acc_train','acc_test','auc_train','auc_test','f1_score_train','f1_score_test',\
  'gpu','sample_size','test_frac','epochs','batch_size','learning_rate','optimizer','momentum', 'embedding_dim','stride_cnn','kernel_sizes',\
 'output_channel(filters)','hidden_size(lstm)','layers(lstm)','dense_linear_dim','dropout']

# Truncate decimal
fix_decimal = 4

# Truncate decimals of float 
def truncate(number, digits) -> float:
    stepper = 10.0 ** digits
    return math.trunc(stepper * number) / stepper

def write_to_index(dict_index,index_file):
    index_models = None

    if os.path.exists(index_file):
        index_models = pd.read_csv(index_file,sep=";",header=0)
    else:
        temp_file = open(index_file,"w+")
        for i,col in enumerate(index_columns):

            temp_file.write(str(col))

            if i + 1 != len(index_columns):
                temp_file.write(";")
            else:
                temp_file.write("\n")

        temp_file.close()
        index_models = pd.read_csv(index_file,sep=";",header=0)


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
    index_models = index_models[index_columns]

    # Saving results
    index_models.to_csv(index_file,sep=";",index=False)

def test_model(test, model, criterion):

    # Reactive dropout batch_norm layer
    model.eval()

    # Reducing usage of memory deactivating autograd engine
    with torch.no_grad():
        loss_test = 0
        t = tqdm(iter(test),desc="Batchs Test",total=len(test), leave=False)
        pred_test = []
        true_test = []
        probs_test = []
        for idx, sample in enumerate(t):
            output = model(sample[0].to(device),sample[1].to(device),sample[2].to(device),sample[3].to(device))
            loss = criterion(output[:,0],sample[4].to(device).float())
            loss_test += loss.item()
            pred = [1 if x >= 0.5 else 0 for x in  output[:,0].tolist()]
            probs = output[:,0].tolist()
            true = sample[4].tolist()
            pred_test.extend(pred)
            true_test.extend(true)
            probs_test.extend(probs)


        fpr, tpr, thresholds = roc_curve(true_test,probs_test)

        loss_test = truncate(loss_test / len(test), fix_decimal)
        acc_test = truncate(accuracy_score(true_test,pred_test), fix_decimal)
        f1_test =  truncate(f1_score(true_test,pred_test), fix_decimal)
        auc_test = truncate(auc(fpr,tpr), fix_decimal)

        return  loss_test,acc_test, f1_test, auc_test ,pred_test,true_test

def save_model(model, optim, epoch, loss, learning_rate, model_name):

    PATH = models_folder + str(model_name) + ".pth"

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'loss': loss,
            'learning_rate': learning_rate
            }, PATH)

def train_model(train, test, model, epochs, learning_rate, model_name, date, batch_size, sample_size, test_frac,
                index_file, optimizer,momentum, dict_index):

    # Loss function
    criterion = nn.BCELoss()

    # Optimizer
    if optimizer.lower() == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer.lower() == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer.lower() == "rmsprop":
        optim = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        print("Optimizer is not in list avaliable")
        sys.exit(1)

    # Metrics important for binary classification
    best_acc_test = 0
    best_acc_train = 0
    best_f1_score_test = 0
    best_f1_score_train = 0
    best_auc_test = 0
    best_auc_train = 0

    # Create files for report each epoch of training process
    train_report = open(report_folder + model_name + "_train.csv","w+")
    test_report = open(report_folder + model_name + "_test.csv","w+")

    train_report.write("epoch;loss;acc;f1_score;auc\n")
    test_report.write("epoch;loss;acc;f1_score;auc\n")

    t1 = trange(epochs,leave=True,desc='Epochs')
    t2 = trange(len(train),leave=True,desc='Batchs')
    for ep in t1:
        loss_ep = 0
        model.train()
        pred_ep = []
        true_ep = []
        probs_ep = []
        for idx, (sample, _) in enumerate(zip(train,t2)):
            output = model(sample[0].to(device),sample[1].to(device),sample[2].to(device),sample[3].to(device))
            loss = criterion(output[:,0],sample[4].to(device).float())
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_ep += loss.item()
            pred = [1 if x >= 0.5 else 0 for x in  output[:,0].tolist()]
            probs = output[:,0].tolist()
            true = sample[4].tolist()
            pred_ep.extend(pred)
            true_ep.extend(true)
            probs_ep.extend(probs)

        fpr, tpr, thresholds = roc_curve(true_ep,probs_ep)

        acc_train = truncate(accuracy_score(true_ep,pred_ep), fix_decimal)
        f1_train = truncate(f1_score(true_ep, pred_ep), fix_decimal)
        auc_train = truncate(auc(fpr,tpr), fix_decimal)
        loss_ep  = truncate(loss_ep / len(train), fix_decimal)


        # Valdiate model
        loss_test, acc_test, f1_test ,auc_test, pred_test,true_test = test_model(test, model, criterion)

        # Writing reports 
        train_report.write(str(ep) + ";" + str(loss_ep) + ";" + str(acc_train) + ";" + str(f1_train) + ";" +
                           str(auc_train) + "\n")

        test_report.write(str(ep) + ";" + str(loss_test) + ";" + str(acc_test) + ";" + str(f1_test) + ";" +
                           str(auc_test) + "\n")

        if acc_train > best_acc_train:
            best_acc_train = acc_train

        if f1_train > best_f1_score_train:
            best_f1_score_train = f1_train

        if auc_train > best_auc_train:
            best_auc_train  = auc_train

        if acc_test > best_acc_test:
            best_acc_test = acc_test
            save_model(model, optim, ep, loss_ep, learning_rate, model_name)

        if f1_test > best_f1_score_test:
            best_f1_score_test = f1_test

        if auc_test > best_auc_test:
            best_auc_test  = auc_test


    dict_index['model_name'] = [model_name]
    dict_index['date'] = [date]
    dict_index['acc_train'] = [best_acc_train]
    dict_index['acc_test'] = [best_acc_test]
    dict_index['loss_train(last)'] = [loss_ep]
    dict_index['loss_test(last)'] = [loss_test]
    dict_index['f1_score_train'] = [best_f1_score_train]
    dict_index['f1_score_test'] = [best_f1_score_test]
    dict_index['auc_train'] = [best_auc_train]
    dict_index['auc_test'] = [best_auc_test]
    dict_index['epochs'] = [epochs]
    dict_index['batch_size'] = [batch_size]
    dict_index['learning_rate'] = [learning_rate]
    dict_index['sample_size'] = [sample_size]
    dict_index['test_frac'] = [test_frac]
    write_to_index(dict_index,index_file)

    train_report.close()
    test_report.close()


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
# Use dateime.now as default name for model
@click.option('--model_name',default=datetime.now().isoformat().split(".")[0], help="Model name to be saved")
@click.option('--index_file',type=click.Path(exists=False),default="../../reports/index.csv")
@click.option('--frac_sample',default=1.0, help="Fraction of sample train used to train the model")
@click.option('--test_frac',default=0.2, help="Fraction of sample train used to test the model")
@click.option('--batch_size',default=128, help="Batch size")
@click.option('--epochs',default=10, help="Numbers of epochs to train the model")
@click.option('--learning_rate', default=0.001, help="Learning rate")
@click.option('--embedding_dim', default=100, help="Embedding dim for words")
@click.option('--kernel_sizes', cls=OptionEatAll, default=[2,3,5,7], help="Kernel sizes")
@click.option('--out_channel', default=50, help="The number of filters(channels)")
@click.option('--stride', default=50, help="Stride of cnn")
@click.option('--layers_lstm', default=2, help="Number of layers of lstm")
@click.option('--hidden_size', default=100, help="Hidden neurons in lstm layers")
@click.option('--dense_size', default=100, help="Number of neurons in dense linear layer")
@click.option('--dropout',default=0.5, help="Probability of dropout")
@click.option('--optimizer',default='adam', help="Optmizer")
@click.option('--momentum', default=0.0, help="Nesterov momentum sgd")
def main(input_filepath,index_file,model_name,frac_sample,test_frac,batch_size, epochs, learning_rate,
         embedding_dim,kernel_sizes, out_channel, stride, layers_lstm,hidden_size,dense_size,dropout, optimizer,
         momentum):

    model_name = model_name.replace(":","-")
    create_model_date = date.today().isoformat()

    kernel_sizes = [int(x) for x in kernel_sizes] # Transforming into ints numbers


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
    test = DataLoader(test_dataset,batch_size, shuffle=False)
    """
        These dataloaders returns question1, q1_len, question2, q2_len and label
    """

    padding = []
    for p in range(len(kernel_sizes)):
        padding.append(int((embedding_dim  - 1) * stride - embedding_dim - kernel_sizes[p]))

    # Some hiperparameters that will be deducted by the above 
    input_lstm = int((((embedding_dim + 2 * padding[0] - 1 * (kernel_sizes[0] - 1) - 1)/stride) + 1)) * len(kernel_sizes)
    vocab_size = len(word_dict) + 1

    model = QuoraModel(kernel_sizes,out_channel,stride,hidden_size,layers_lstm,embedding_dim, dense_size,
                       vocab_size, input_lstm, dropout, padding).to(device)

    dict_index = {}
    dict_index['dropout'] = [dropout]
    dict_index['dense_linear_dim'] = [dense_size]
    dict_index['layers(lstm)'] = [layers_lstm]
    dict_index['hidden_size(lstm)'] = [hidden_size]
    dict_index['output_channel(filters)'] = [out_channel]
    dict_index['kernel_sizes'] = [kernel_sizes]
    dict_index['stride_cnn'] = [stride]
    dict_index['embedding_dim'] = [embedding_dim]
    dict_index['optimizer'] = [optimizer]
    dict_index['momentum'] = [momentum]

    train_model(train, test, model, epochs, learning_rate, model_name, create_model_date, batch_size, len(data),
                test_frac, index_file, optimizer, momentum, dict_index)

if __name__ == "__main__":
    main()
