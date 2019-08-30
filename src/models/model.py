import torch 
import torch.nn as nn
import torch.nn.functional as F

class QuoraModel(nn.Module):
    def __init__(self,kernel_sizes, out_channels, stride, hidden_size,layers, embedding_dim,output_dim_fc1, vocab_size,
                 input_lstm, dropout, padding):
        super(QuoraModel, self).__init__()

        self.kernel_sizes = kernel_sizes
        self.out_channels = out_channels
        self.stride = stride
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.layers = layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0,)
        self.cnns = nn.ModuleList([nn.Conv1d(20,out_channels,kernel_sizes[i],stride,padding=padding[i]) for i in
                                   range(len(kernel_sizes))])
        self.gru = nn.GRU(input_lstm,hidden_size,layers,dropout=dropout,batch_first=True,bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, output_dim_fc1) # 2 * because of bidirectional lstm
        self.fc2 = nn.Linear(output_dim_fc1 * 2,1)  # Concat representation between q1 and q2
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(out_channels)


    def forward(self, questions1, q1_lens, questions2,q2_lens, hidden=None):
        embedding_output_q1 = self.embedding(questions1)
        embedding_output_q2 = self.embedding(questions2)

        cnn_output_q1 = [torch.tanh(conv(embedding_output_q1)) for conv in self.cnns]
        cnn_output_q2 = [torch.tanh(conv(embedding_output_q2)) for conv in self.cnns]

        q1_cnn_cat = torch.cat(cnn_output_q1,dim=-1)
        q2_cnn_cat = torch.cat(cnn_output_q2,dim=-1)

        q1_cnn_cat = torch.relu(q1_cnn_cat)
        q2_cnn_cat = torch.relu(q2_cnn_cat)

        # Batch normalization
        q1_cnn_cat = self.batch_norm(q1_cnn_cat)
        q2_cnn_cat = self.batch_norm(q2_cnn_cat)

        lstm_output_q1, hn = self.gru(q1_cnn_cat)
        lstm_output_q2, hn = self.gru(q2_cnn_cat)

        linear_output_q1 = self.fc1(lstm_output_q1[:,-1,:])
        linear_output_q2 = self.fc1(lstm_output_q2[:,-1,:])

        linear_output_q1 = torch.relu(linear_output_q1)
        linear_output_q2 = torch.relu(linear_output_q2)

        # Dropout to avoid overfitting
        linear_output_q1 = self.dropout(linear_output_q1)
        linear_output_q2 = self.dropout(linear_output_q2)

        concat_q1_q2 = torch.cat((linear_output_q1,linear_output_q2),dim=1)
        output = self.fc2(concat_q1_q2)
        output = torch.sigmoid(output)

        return output
