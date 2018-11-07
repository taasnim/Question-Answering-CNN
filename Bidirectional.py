import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import load_data, build_dict, gen_embeddings, vectorize, mini_batches

import numpy as np
import math
from sklearn import metrics


class Bidirectional_Attention_Model(nn.Module):

    def __init__(self, embeddings, hidden_size, output_size, maxlen, num_layers=1):
        super(Bidirectional_Attention_Model, self).__init__()
        
        self.vocab_size = embeddings.shape[0]
        self.embedding_size = embeddings.shape[1]
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.maxlen = maxlen
        self.num_layers = num_layers
        
        weight = torch.Tensor(embeddings)
        self.embeddings = nn.Embedding.from_pretrained(weight, freeze=False)
            
        self.context_gru = nn.GRU(embedding_size, hidden_size, num_layers=num_layers, 
                          batch_first=True, bidirectional=True, dropout=0.25)
        
        self.sim_W = nn.Linear(hidden_size*6, 1)
        
        self.modeling_layer = nn.GRU(hidden_size*8, hidden_size, num_layers=2, 
                                     batch_first=True, bidirectional=True, dropout=0.25)
        
        self.output_layer_1 = nn.Linear(hidden_size*10, 1)
        self.output_layer_2 = nn.Linear(maxlen, 1000)
        self.output_layer_3 = nn.Linear(1000, output_size)
        
        
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
            # output shape: (batch_size, seq_len, hidden_size * num_directions) --seq_len is the largest lengths in the minibatch
            # hidden shape: (num_layers * num_directions, batch_size, hidden_size)
            
            return output, hidden

        doc_data = self.embeddings(doc_x) # doc_data shape: (batch_size, doc_seq_len, embedding_dim)
        ques_data = self.embeddings(ques_x) # ques_data shape: (batch_size, ques_seq_len, embedding_dim)
         
        ## contextual embedding for documents and ques
        doc_output, doc_hidden = contextual_embedding(doc_data, doc_seq_lengths)
        ques_output, ques_hidden = contextual_embedding(ques_data, ques_seq_lengths)
        
        ## Attention Flow
        # Similarity Matrix calcuation
        doc_seq_len = doc_output.size(1) # T
        ques_seq_len = ques_output.size(1) # J
        shape = (doc_output.size(0), doc_seq_len, ques_seq_len, 2*self.hidden_size) # (N, T, J, 2d)
        #print("T: ", doc_seq_len, " J: ", ques_seq_len, " Shape: ", shape)
        doc_output_extra = doc_output.unsqueeze(2)  # (N, T, 1, 2d)
        doc_output_extra = doc_output_extra.expand(shape) # (N, T, J, 2d)
        ques_output_extra = ques_output.unsqueeze(1)  # (N, 1, J, 2d)
        ques_output_extra = ques_output_extra.expand(shape) # (N, T, J, 2d)
        elmwise_mul = torch.mul(doc_output_extra, ques_output_extra) # (N, T, J, 2d)
        #print("doc: ", doc_output_extra.size(), " ques: ", ques_output_extra.size(), " elem: ", elmwise_mul.size())
        cat_data = torch.cat((doc_output_extra, ques_output_extra, elmwise_mul), 3) # (N, T, J, 6d), [h;u;h◦u]
        similarity_matrix = self.sim_W(cat_data).squeeze() # (N, T, J)
        #print("cat: ", cat_data.size(), " similarity_matrix: ", similarity_matrix.size())
        
        
        a = F.softmax(similarity_matrix, dim=2) # (bs, T, J)
        doc2ques_attention = torch.bmm(a, ques_output) # (bs, T, 2*d)
        #print("a: ", a.size(), " doc2ques: ", doc2ques_attention.size())
        b = F.softmax(torch.max(similarity_matrix, dim=2)[0], dim=1).unsqueeze(1) # (bs, 1, T)
        ques2doc_attention = torch.bmm(b, doc_output).squeeze()  # (bs, 2d)
        ques2doc_attention = ques2doc_attention.unsqueeze(1).expand(-1, doc_seq_len, -1) # (bs, T, 2*d)
        # q2c_att = torch.stack([q2c_att] * c_len, dim=1)

        G = torch.cat((doc_output, doc2ques_attention, doc_output.mul(doc2ques_attention), 
                       doc_output.mul(ques2doc_attention)), 2) # (bs, T, 8d)
        #print("b: ", b.size(), " ques2doc: ", ques2doc_attention.size(), " G: ", G.size())
        
        ## Modeling Layer
        M, _h = self.modeling_layer(G) # M: (bs, T, 2d)
        
        ## Output Layer
        G_M = torch.cat((G, M), 2) # (bs, T, 10d)
        G_M = self.output_layer_1(G_M).squeeze() # (bs, T)
        G_M = F.pad(G_M, pad=(0,self.maxlen-G_M.size(1),0,0), mode='constant', value=0) # (bs, self.maxlen)

        logits = F.relu(self.output_layer_2(G_M)) # (N, 1000)
        logits = self.output_layer_3(logits) # (N, output_size)
        #print("M: ", M.size(), " G_M: ", G_M.size(), " logits: ", logits.size())
        
        #return F.softmax(logits, dim=-1) 
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
word_dict = build_dict(train_d + train_q)
entity_markers = list(set([w for w in word_dict.keys() if w.startswith('@entity')] + train_a))
entity_markers = ['<unk_entity>'] + entity_markers
entity_dict = {w: index for (index, w) in enumerate(entity_markers)}
print('Entity markers: %d' % len(entity_dict))
num_labels = len(entity_dict)

# Glove embeddings
embedding_size = 300
embeddings = gen_embeddings(word_dict, embedding_size, '../pretrained_embeddings/Glove6B/glove.6B.{}d.txt'.format(embedding_size))
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



model = Bidirectional_Attention_Model(embeddings=embeddings, hidden_size=128, output_size=len(entity_dict), maxlen=doc.shape[1])
print(model)

device = torch.device("cuda")
#device= torch.device("cpu")
print("Using Device: ", device)

model = model.to(device)
criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.Adam(model.parameters() , lr=0.1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def eval_on_dev_set(model, epoch, iter):
    model.eval()
    
    #with torch.no_grad():
    dev_minibatches = mini_batches(dev_doc, dev_query, dev_answer, dev_doc_len, dev_query_len, mini_batch_size=8)
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
    
    train_minibatches = mini_batches(doc, query, answer, doc_len, query_len, epoch, mini_batch_size=8)
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
        #for preventing exploding gradients
        clip_grad_norm_(model.parameters(), max_norm=0.75)
        optimizer.step()
        
        running_loss += loss.detach().item()
        #print(i, ":  ", loss.item())
        
        if ((i + 1) % 100 == 0 or i == num_minibatches - 1):
            acc_dev = eval_on_dev_set(model, epoch, i)
            if acc_dev > best_acc:
                best_acc = acc_dev
                best_epoch = epoch
                best_iter = i
            print("Epoch: ", epoch," iter: ", i, " cur acc: ", acc_dev, "best acc: ", best_acc, "   loss = ", running_loss/num_minibatches)

    
