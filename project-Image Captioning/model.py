import torch
import torch.nn as nn
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.batch = nn.BatchNorm1d(embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        features = self.batch(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_dim = hidden_size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, bias=True,
                            dropout=0.3, batch_first= True)
        
        # Fully Connected layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def init_hidden(self, batch_size):
        return (torch.zeros((1, batch_size, self.hidden_dim), device=device),
                torch.zeros((1, batch_size, self.hidden_dim), device=device))
    
    def forward(self, features, captions):
        # initialize hidden state
        self.hidden = self.init_hidden(features.shape[0])
        
        # create embedded word vectors for each word in a sentence
        print("A:",captions.shape)
        embeds = self.word_embeddings(captions[:, :-1])
        print("B:",embeds.shape)
        
        embed_in = torch.cat((features.unsqueeze(1), embeds), 1)
        
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        lstm_out, self.hidden = self.lstm(embed_in, self.hidden)
        
        result = self.fc(lstm_out)
        
        return result

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output = []
        for i in range(max_len):
            l_out, states = self.lstm(inputs, states)
            l_out = self.fc(l_out.squeeze(1))
            _, predicted = l_out.max(1) 
            output.append(predicted.item())
            inputs = self.word_embeddings(predicted.long()) 
            inputs = inputs.unsqueeze(1)
        return output