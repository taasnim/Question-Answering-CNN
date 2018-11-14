import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import load_data, build_dict, gen_embeddings, gen_embeddings_fasttext, vectorize, mini_batches

import numpy as np
import math
from sklearn import metrics


class Unidirectional_Attention_Model(nn.Module):

    def __init__(self, embeddings, hidden_size, output_size, num_layers=1):
        super(Unidirectional_Attention_Model, self).__init__()
        
        self.vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        weight = torch.Tensor(embeddings)
        self.embeddings = nn.Embedding.from_pretrained(weight, freeze=False)
            
        self.context_gru = nn.GRU(embedding_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.25)
        
        self.ws = nn.Linear(hidden_size*2, hidden_size*2) # mult by 2 for bidirectional
        self.linear = nn.Linear(hidden_size*2, output_size)

        
    def forward(self, doc_x, ques_x, doc_seq_lengths, ques_seq_lengths):   
        '''
        doc_x: torch tensor. size - (batch_size, doc_seq_len)
        ques_x: torch tensor. size - (batch_size, ques_seq_len)
        doc_seq_lengths: 1d numpy array containing lengths of each document in doc_x
        ques_seq_lengths: 1d numpy array containing lengths of each question in ques_x
        
        '''
        
        def contextual_embedding(data, seq_lengths):
            # Sort by length (keep idx)
            seq_lengths, idx_sort = np.sort(seq_lengths)[::-1], np.argsort(-seq_lengths)
            idx_original = np.argsort(idx_sort)
            idx_sort = torch.from_numpy(idx_sort).to(device)
            data = data.index_select(0, idx_sort)

            packed_input = pack_padded_sequence(data, seq_lengths, batch_first=True)
            packed_output, hidden = self.context_gru(packed_input)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
            #print("out: ", output.size(), " hid: ", hidden.size())

            # Un-sort by length
            idx_original = torch.from_numpy(idx_original).to(device)
            output = output.index_select(0, idx_original)
            hidden = hidden.index_select(1, idx_original)
            
            return output, hidden

        doc_data = self.embeddings(doc_x) # doc_data shape: (batch_size, doc_seq_len, embedding_dim)
        ques_data = self.embeddings(ques_x) # ques_data shape: (batch_size, ques_seq_len, embedding_dim)
         
        ##For Documents/passages
        doc_output, doc_hidden = contextual_embedding(doc_data, doc_seq_lengths)
        ques_output, ques_hidden = contextual_embedding(ques_data, ques_seq_lengths)
        # output shape: (batch_size, seq_len, hidden_size * num_directions)
        # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
        
        ques_fwd_h = ques_hidden[0:ques_hidden.size(0):2]
        ques_bwd_h = ques_hidden[1:ques_hidden.size(0):2]
        ques_hidden = torch.cat([ques_fwd_h, ques_bwd_h], dim=2)   
        #print("After hid: ", ques_hidden.size())
        
        q_ws = self.ws(ques_hidden) #q_ws shape:  torch.Size([1, bs, 256])
        #print("q_ws shape: ", q_ws.size()) 
        q_ws = q_ws.squeeze().unsqueeze(1) #q_ws shape:  torch.Size([bs, 1, 256])
        q_ws.transpose_(1,2) #q_ws shape:  torch.Size([bs, 256, 1])
        q_ws_p = torch.bmm(doc_output, q_ws).squeeze() # q_ws_p shape: torch.Size([bs, timesteps])
        #print("q_ws_p shape: ", q_ws_p.size())
        alpha = F.softmax(q_ws_p, dim=1) #alpha shape:  torch.Size([bs, 1808])
        #print("alpha shape: ", alpha.size())
        attention = torch.mul(alpha.unsqueeze(2), doc_output) #attention shape:  torch.Size([bs, 1808, 256])
        #print("attention shape: ", attention.size())
        attention = torch.sum(attention, dim=1) #After summing attention shape:  torch.Size([bs, 256])
        #print("After summing attention shape: ", attention.size())
        
        logits = self.linear(attention) #logits shape:  torch.Size([bs, numClasses])
        #print("logits shape: ", logits.size())
        
        return logits


# Read data
fin_train = 'data/cnn/train.txt'
fin_dev = 'data/cnn/dev.txt'
print("Train Data: ")
train_d, train_q, train_a = load_data(fin_train, relabeling=True)
print("Dev Data: ")
dev_d, dev_q, dev_a = load_data(fin_dev, relabeling=True)

# Build dictionary
print('Build dictionary..')
word_dict = build_dict(train_d + train_q, max_words=50000)
entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_a))
entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
print('Entity markers: %d' % len(entity_dict))
num_labels = len(entity_dict)

# Glove embeddings
embedding_size = 300
#embeddings = gen_embeddings(word_dict, embedding_size, '../pretrained_embeddings/Glove6B/glove.6B.{}d.txt'.format(embedding_size))
# Fasttext embeddings
#embeddings = gen_embeddings_fasttext(word_dict, embedding_size, '/home/tasnim/projects/word_translation/data/wiki.en.vec')
# Random embeddings
print("Using random embeddings")
num_words = len(word_dict) + 2
embeddings = np.random.uniform(size=(num_words, embedding_size))
print('Embeddings shape: ', embeddings.shape)


# vectorize the data
doc, query, l, answer, doc_len, query_len = vectorize(train_d, train_q, train_a, word_dict, entity_dict)
dev_doc, dev_query, dev_l, dev_answer, dev_doc_len, dev_query_len = vectorize(dev_d, dev_q, dev_a, word_dict, entity_dict)



model = Unidirectional_Attention_Model(embeddings=embeddings, hidden_size=128, output_size=len(entity_dict))
print(model)

device = torch.device("cuda")
#device= torch.device("cpu")
print("Using Device: ", device)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters() , lr=0.1)
optimizer = torch.optim.SGD(model.parameters() , lr=0.1)

def eval_on_dev_set(model):
    model.eval()
    
    #with torch.no_grad():
    dev_minibatches = mini_batches(dev_doc, dev_query, dev_answer, dev_doc_len, dev_query_len)
    num_minibatches = len(dev_minibatches)
    
    predicted_labels = torch.LongTensor([]).to(device)
    for (i, dev_minibatch) in enumerate(dev_minibatches):
        dev_minibatch_doc, dev_minibatch_query, dev_minibatch_answer, dev_minibatch_doc_len, dev_minibatch_query_len= dev_minibatch
        doc_tensor = torch.LongTensor(dev_minibatch_doc)
        query_tensor = torch.LongTensor(dev_minibatch_query)   

        doc_tensor = doc_tensor.to(device)
        query_tensor = query_tensor.to(device)

        scores = model(doc_tensor, query_tensor, np.array(dev_minibatch_doc_len), np.array(dev_minibatch_query_len))
        
        _, batch_predicted_labels = torch.max(scores, dim=1) 
        predicted_labels = torch.cat((predicted_labels, batch_predicted_labels))
        
    acc_dev = metrics.accuracy_score(np.array(dev_answer), predicted_labels.cpu().numpy())
    #mic_p, mic_r, mic_f, sup = metrics.precision_recall_fscore_support(answer_tensor.numpy(), predicted_label.numpy(), average='micro')
    #mac_p, mac_r, mac_f, sup = metrics.precision_recall_fscore_support(answer_tensor.numpy(), predicted_label.numpy(), average='macro')

    #print("Epoch: ", epoch, "  Iteration: ", iter, "  Dev Accuracy-- ", acc_dev)
    return acc_dev


best_acc=-1
best_epoch=-1
best_iter=-1
for epoch in range(30):
    
    running_loss = 0
    
    train_minibatches = mini_batches(doc, query, answer, doc_len, query_len, epoch)
    num_minibatches = len(train_minibatches)
    #print("number of minibatches: ", num_minibatches)
    for (i, train_minibatch) in enumerate(train_minibatches):
        
        model.train()
        # Set the gradients to zeros
        optimizer.zero_grad()
        
        train_minibatch_doc, train_minibatch_query, train_minibatch_answer, train_minibatch_doc_len, train_minibatch_query_len= train_minibatch
        
        doc_tensor = torch.LongTensor(train_minibatch_doc)
        query_tensor = torch.LongTensor(train_minibatch_query)
        answer_tensor = torch.LongTensor(train_minibatch_answer)
        
        doc_tensor = doc_tensor.to(device)
        query_tensor = query_tensor.to(device)
        answer_tensor = answer_tensor.to(device)
        
        #seq_tensor.requires_grad_()
        
        scores = model(doc_tensor, query_tensor, np.array(train_minibatch_doc_len), np.array(train_minibatch_query_len))
        #print(scores.size())
        #scores = scores.view(-1, 5)
        
        loss = criterion(scores, answer_tensor)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.detach().item()
        #print(i, ":  ", loss.item())
        
        if ((i + 1) % 100 == 0 or i == num_minibatches - 1):
            acc_dev = eval_on_dev_set(model)
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_epoch = epoch
                best_iter = i
        
            print("Epoch: ", epoch," iter: ", i, " cur acc: ", acc_dev, "best acc: ", best_acc, "   loss = ", running_loss/num_minibatches)


