import numpy as np
from collections import Counter
import math
import io


def load_data(in_file, max_example=None, relabeling=True):
    """
        load CNN / Daily Mail data from {train | dev | test}.txt
        relabeling: relabel the entities by their first occurence if it is True.
    """

    documents = []
    questions = []
    answers = []
    num_examples = 0
    with open(in_file, 'r', encoding='utf-8') as f:
        while True:
            line = f.readline()
            if not line:
                break
            question = line.strip().lower()
            answer = f.readline().strip()
            document = f.readline().strip().lower()

            if relabeling:
                q_words = question.split(' ')
                d_words = document.split(' ')
                assert answer in d_words

                entity_dict = {}
                entity_id = 0
                for word in d_words + q_words:
                    if (word.startswith('@entity')) and (word not in entity_dict):
                        entity_dict[word] = '@entity' + str(entity_id)
                        entity_id += 1

                q_words = [entity_dict[w] if w in entity_dict else w for w in q_words]
                d_words = [entity_dict[w] if w in entity_dict else w for w in d_words]
                answer = entity_dict[answer]

                question = ' '.join(q_words)
                document = ' '.join(d_words)

            questions.append(question)
            answers.append(answer)
            documents.append(document)
            num_examples += 1

            f.readline()
            if (max_example is not None) and (num_examples >= max_example):
                break
                
    print('#Examples: %d' % len(documents))
    return (documents, questions, answers)


def build_dict(sentences, max_words=50000):
    """
        Build a dictionary for the words in `sentences`.
        Only the max_words ones are kept and the remaining will be mapped to <UNK>.
    """
    word_count = Counter()
    for sent in sentences:
        for w in sent.split(' '):
            word_count[w] += 1

    ls = word_count.most_common(max_words)
    print('#Words: %d -> %d' % (len(word_count), len(ls)))
    for key in ls[:5]:
        print(key)
    print('...')
    for key in ls[-5:]:
        print(key)

    # leave 0 to UNK
    # leave 1 to delimiter |||
    vocab_dict = {w[0]: index + 2 for (index, w) in enumerate(ls)}
    vocab_dict['<UNK>'] = 0
    vocab_dict['<PAD>'] = 1

    return vocab_dict


def gen_embeddings(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = len(word_dict) + 2
    embeddings = np.random.uniform(size=(num_words, dim))
    print('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        print('Loading embedding file: %s' % in_file)
        pre_trained = 0
        for line in open(in_file, encoding='utf-8').readlines():
            sp = line.split()
            assert len(sp) == dim + 1 # word + embeddings ..
            if sp[0] in word_dict:
                pre_trained += 1
                embeddings[word_dict[sp[0]]] = [float(x) for x in sp[1:]]
        print('Pre-trained: %d (%.2f%%)' %
              (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def gen_embeddings_fasttext(word_dict, dim, in_file=None):
    """
        Generate an initial embedding matrix for `word_dict`.
        If an embedding file is not given or a word is not in the embedding file,
        a randomly initialized vector will be used.
    """

    num_words = len(word_dict) + 2
    embeddings = np.random.uniform(size=(num_words, dim))
    print('Embeddings: %d x %d' % (num_words, dim))

    if in_file is not None:
        print('Loading fasttext embedding file: %s' % in_file)
        pre_trained = 0
        with io.open(in_file, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            for i, line in enumerate(f):
                if i == 0:
                    split = line.split()
                    assert len(split) == 2
                    #assert _emb_dim_file == int(split[1])
                else:
                    word, vect = line.rstrip().split(' ', 1)        
                    if word in word_dict:
                        pre_trained += 1
                        vect = np.fromstring(vect, sep=' ')
                        #print(vect.shape)
                        embeddings[word_dict[word]] = vect
        print('Pre-trained: %d (%.2f%%)' % (pre_trained, pre_trained * 100.0 / num_words))
    return embeddings


def vectorize(doc, query, ans, word_dict, entity_dict):
    """
        Vectorize `examples`.
        in_x1, in_x2: sequences for document and question respecitvely.
        in_y: label
        in_l: whether the entity label occurs in the document.
    """
    in_x1 = []
    in_x2 = []
    in_l = np.zeros((len(doc), len(entity_dict)))#.astype(config._floatX)
    in_y = []
    
    doc_len = [len(w.split(' ')) for i, w in enumerate(doc)]
    query_len = [len(w.split(' ')) for i, w in enumerate(query)]
    doc_maxlen = max(doc_len)
    q_maxlen = max(query_len)
    print("max doc len: ", doc_maxlen)
    print("max query len: ", q_maxlen)
    
    for idx, (d, q, a) in enumerate(zip(doc, query, ans)):
        d_words = d.split(' ')
        q_words = q.split(' ')
        #print(d_words)
        #print(q_words)
        #print(a)
        assert (a in d)
        seq1 = [word_dict[w] if w in word_dict else 0 for w in d_words]
        seq1 = seq1[:doc_maxlen]
        pad_1 = max(0, doc_maxlen - len(seq1))
        seq1 += [0] * pad_1
        seq2 = [word_dict[w] if w in word_dict else 0 for w in q_words]
        seq2 = seq2[:q_maxlen]
        pad_2 = max(0, q_maxlen - len(seq2))
        seq2 += [0] * pad_2
        
        if (len(seq1) > 0) and (len(seq2) > 0):
            in_x1.append(seq1)
            in_x2.append(seq2)
            in_l[idx, [entity_dict[w] for w in d if w in entity_dict]] = 1.0
            in_y.append(entity_dict[a])
        '''    
        if idx % 1000 == 0:
            print('vectorize: Vectorization: processed %d / %d' % (idx, len(doc)))
        '''
    return np.array(in_x1), np.array(in_x2), np.array(in_l), in_y, doc_len, query_len


def mini_batches(doc, query, answer, doc_len, query_len, epoch=-1, mini_batch_size=32):

    if epoch>=0:
        np.random.seed(2018+epoch)
        np.random.shuffle(doc)    
        np.random.seed(2018+epoch)
        np.random.shuffle(query)
        np.random.seed(2018+epoch)
        np.random.shuffle(answer)
        np.random.seed(2018+epoch)
        np.random.shuffle(doc_len)
        np.random.seed(2018+epoch)
        np.random.shuffle(query_len)    
    
    m = doc.shape[0] #total number of examples in the minibatch
    mini_batches = []

    num_complete_minibatches = int(math.floor(m / mini_batch_size))

    for k in range(0, num_complete_minibatches):
        start_index = k * mini_batch_size
        end_index = start_index + mini_batch_size
        
        mini_batch_doc = doc[start_index : end_index]
        mini_batch_query = query[start_index : end_index]
        mini_batch_answer = answer[start_index : end_index]
        mini_batch_doc_len = doc_len[start_index : end_index]
        mini_batch_query_len = query_len[start_index : end_index]
       
        mini_batch = (mini_batch_doc, mini_batch_query, mini_batch_answer, mini_batch_doc_len, mini_batch_query_len)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        start_index = num_complete_minibatches * mini_batch_size
        
        mini_batch_doc = doc[start_index : m]
        mini_batch_query = query[start_index : m]
        mini_batch_answer = answer[start_index : m]
        mini_batch_doc_len = doc_len[start_index : m]
        mini_batch_query_len = query_len[start_index : m]
    
        mini_batch = (mini_batch_doc, mini_batch_query, mini_batch_answer, mini_batch_doc_len, mini_batch_query_len)
        mini_batches.append(mini_batch)

    return mini_batches
