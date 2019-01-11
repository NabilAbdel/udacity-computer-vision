import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):

        super(DecoderRNN, self).__init__()
        ## create the word embedding
        self.embed = nn.Embedding(vocab_size, embed_size)
        ## define the LSTM model
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        ## since we want the output to be different size we use the linear on the last cell output
        ## final output fully connected layer
        self.linear = nn.Linear(hidden_size, vocab_size)

        ## initialize the initial state
        self.init_weights()

    def forward(self, features, captions):

        ## here we need to remove end token which is <END> 
        captions = captions[:,:-1]
        words_embeds = self.embed(captions)
        inputs = torch.cat((features.unsqueeze(1),words_embeds),1)
        output, hidden = self.lstm(inputs)
        ## add out to the fully connected layer
        output = self.linear(output)

        return output

    def init_weights(self):
        ''' Initialize weights for fully connected layer and lstm forget gate bias'''
        
        # Set initial value 
        self.linear.bias.data.fill_(0.01)
        # FC weights as xavier normal
        torch.nn.init.xavier_normal_(self.linear.weight)
        # initialize forget gate bias to 1, this is according to the paper
        # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745/4
        # adding bias to 1 improved performance
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n,  names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n//4, n//2
                bias.data[start:end].fill_(1.)
    
    def sample(self, features, states=None, max_seg_length=20):

       
        print("sample 19")
        
        sampled_ids = []

        for i in range(max_seg_length):
            # hiddens: (batch_size, 1, hidden_size)
            hiddens, states = self.lstm(features, states)
            # outputs:  (batch_size, vocab_size)
            outputs = self.linear(hiddens.squeeze(1))
            # predicted: (batch_size)
            _, predicted = outputs.max(1)
            #predicted = outputs.max(1)[1]
            ##predicted = nn.LogSoftmax(predicted)
            #print(predicted.data[0])
            #sampled_ids.append(predicted.data[0])
            #print(predicted)
            sampled_ids.append(predicted.item())
            # inputs: (batch_size, embed_size)
            features = self.embed(predicted)                       
            # inputs: (batch_size, 1, embed_size)
            features = features.unsqueeze(1)   
           # print(features)
        #print(type(sampled_ids))    
        return sampled_ids