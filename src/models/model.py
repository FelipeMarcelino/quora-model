import torch 
import torch.nn as nn
import torch.nn.functional as F

class QuoraModel(nn.Module):
    def __init__(self,kernel_sizes, out_channels, stride, hidden_size,layers, embedding_dim,output_dim_fc1, vocab_size,
                 input_lstm, dropout):
        super(QuoraModel, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.stride = stride
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0,)
        self.cnn_1 = nn.Conv1d(20,out_channels,kernel_sizes[0],stride,padding=1)
        self.cnn_2 = nn.Conv1d(20,out_channels,kernel_sizes[1],stride,padding=2)
        self.cnn_3 = nn.Conv1d(20,out_channels,kernel_sizes[2],stride,padding=3)
        self.cnn_4 = nn.Conv1d(20,out_channels,kernel_sizes[3],stride,padding=3)
        self.gru = nn.GRU(input_lstm,hidden_size,layers,dropout=dropout,batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, output_dim_fc1) # 2 * because of bidirectional lstm
        self.fc2 = nn.Linear(output_dim_fc1 * 2,1)  # Concat representation between q1 and q2
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_channels)


    def forward(self, questions1, q1_lens, questions2,q2_lens, hidden=None):
        embedding_output_q1 = self.embedding(questions1)
        embedding_output_q2 = self.embedding(questions2)

        cnn_1_output_q1 = self.cnn_1(embedding_output_q1)
        cnn_2_output_q1 = self.cnn_2(embedding_output_q1)
        cnn_3_output_q1 = self.cnn_3(embedding_output_q1)
        cnn_4_output_q1 = self.cnn_3(embedding_output_q1)
        q1_cnn_cat = torch.tanh(torch.cat((cnn_1_output_q1,cnn_2_output_q1,cnn_3_output_q1,cnn_4_output_q1),dim=-1))

        cnn_1_output_q2 = self.cnn_1(embedding_output_q2)
        cnn_2_output_q2 = self.cnn_2(embedding_output_q2)
        cnn_3_output_q2 = self.cnn_3(embedding_output_q2)
        cnn_4_output_q2 = self.cnn_3(embedding_output_q2)
        q2_cnn_cat = torch.tanh(torch.cat((cnn_1_output_q2,cnn_2_output_q2,cnn_3_output_q2,cnn_4_output_q2),dim=-1))

        # Batch normalization
        q1_cnn_cat = self.batch_norm(q1_cnn_cat)
        q2_cnn_cat = self.batch_norm(q2_cnn_cat)

        lstm_output_q1, hn = self.gru(q1_cnn_cat)
        lstm_output_q2, hn = self.gru(q2_cnn_cat)

        linear_output_q1 = self.fc1(lstm_output_q1[:,-1,:])
        linear_output_q2 = self.fc1(lstm_output_q2[:,-1,:])

        # Dropout to avoid overfitting
        linear_output_q1 = self.dropout(linear_output_q1)
        linear_output_q2 = self.dropout(linear_output_q2)

        concat_q1_q2 = torch.cat((linear_output_q1,linear_output_q2),dim=1)
        output = self.fc2(concat_q1_q2)
        output = torch.sigmoid(output)

        return output
