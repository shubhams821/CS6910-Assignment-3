# CS6910-Assignment-3

1. Use Kaggle to run the each file.
2. Before running the code make sure to add the dataset 'Aksharantar dataset' in the kaggle and GPU is turned on.
wandb report: [https://wandb.ai/shubham821/CS6910_Assn3_RNN/reports/CS6910-Assignment-3--Vmlldzo3OTQwMDk1](url)
## Question 1:
RNN Network code block - 
```py
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, n_layers, dropout, cell_type):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.cell_type = cell_type
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)




    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        outputs, (hidden, cell) = self.rnn(embedded)
        # outputs = [src length, batch size, hidden dim * n directions]
        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # outputs are always from the top hidden layer
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout,cell_type):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.cell_type = cell_type
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        if self.cell_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif self.cell_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        else:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)


    def forward(self, input, hidden, cell):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden, cell
```


## Question 2:
For wandb sweep configuration:

```py

sweep_configuration = {
    "method": "bayes",
    "metric": {
        "name": "validation_accuracy",
        "goal": "maximize"
    },
    "parameters": {
        "embed_size": {
            "values": [32,64,128]
        },
        "hidden_size": {
            "values": [128, 256, 512]
        },
        "cell_type": {
            "values": ["GRU", "LSTM", "RNN"]
        },
        "num_layers": {
            "values": [1, 2, 3]
        },
        "dropout": {
            "values": [0, 0.1, 0.2]
        },
        "learning_rate": {
            "values": [0.0005, 0.001, 0.005]
        },
        "optimizer": {
            "values": ["Sgd", "Adam","Nadam"]
        },
        "teacher_forcing_ratio": {
            "values": [0.5, 0.75, 0.25]
        }
    }
}
```

## Question 3:
Best model configuration and stats:

| dropout | embed_size | hidden_size | learning_rate | optimizer | num_layers | val_accuracy  | test_accuracy |
|---------|------------|-------------|---------------|-----------|------------|-------|----------------------|
| 0       | 128        | 512         | 0.005          | Adam      | 3          | 0.3662      | 0.3386        |



