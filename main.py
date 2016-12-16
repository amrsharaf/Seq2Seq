import time
from ordering_dataset import get_ordering_dataset
# from chainer.iterators import MultiprocessIterator
from chainer.iterators import SerialIterator
from chainer import Chain
import chainer.links as L
from chainer import optimizers
from chainer import training
from chainer.dataset import convert
from functools import partial
from chainer.functions.loss import softmax_cross_entropy
from chainer.training import extensions
from iterator import Seq2SeqIterator
import numpy as np
import cupy as cp
from chainer import reporter
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from operator import truth
from argparse import ArgumentParser
import h5py
import chainer.functions as F
import chainer.links as L

# 
# Manages encoder/decoder data matrices.
#
class Data():
# function features_on_gpu(features)
#   local clone = {}
#   for i = 1,#features do
#     table.insert(clone, {})
#     for j = 1,#features[i] do
#       table.insert(clone[i], features[i][j]:cuda())
#     end
#   end
#   return clone
# end
# 
# -- using the sentences id, build the alignment tensor
# function generate_aligns(batch_sent_idx, alignment_cc_colidx, alignment_cc_val, source_l, target_l, opt_start_symbol)
#   if batch_sent_idx == nil then
#     return nil
#   end
#   local batch_size = batch_sent_idx:size(1)
# 
#   local src_offset = 0
#   if opt_start_symbol == 0 then
#     src_offset = 1
#   end
# 
#   t = torch.Tensor(batch_size, source_l, target_l)
#   for k = 1, batch_size do
#     local sent_idx=batch_sent_idx[k]
#     for i = 0, source_l-1 do
#       t[k][i+1]:copy(alignment_cc_val:narrow(1, alignment_cc_colidx[sent_idx+1+i+src_offset]+1, target_l))
#     end
#   end
# 
#   return t
# end
# 
# local data = torch.class("data")
# 
# 
    def size(self):
        return self.length
 
    def __getitem__(self, idx):
        # TODO str
        target_input = self.batches[idx][0]
        target_output = self.batches[idx][1]
        nonzeros = self.batches[idx][2]
        source_input = self.batches[idx][3]
        batch_l = self.batches[idx][4]
        target_l = self.batches[idx][5]
        source_l = self.batches[idx][6]
        target_l_all = self.batches[idx][7]
        source_features = self.batches[idx][8]
        # TODO generate_aligns
        alignment = None 
         # TODO: GPU
        return [target_input, target_output, nonzeros, source_input,
            batch_l, target_l, source_l, target_l_all, source_features, alignment]

    def __init__(self, opt, data_file):
        f = h5py.File(data_file, 'r')
        self.source = f['source'][:]
        self.target = f['target'][:]
        self.target_output = f['target_output'][:]
        self.target_l = f['target_l'][:] # max target length each batch
        self.target_l_all = f['target_l_all'][:]
        self.target_l_all = self.target_l_all - 1
        self.batch_l = f['batch_l'][:]
        self.source_l = f['batch_w'][:] # max source length each batch
        
        self.num_source_features = f['num_source_features'][:][0]
        self.source_features = {}
        self.source_features_size = {}
        self.source_features_vec_size = {}
        self.total_source_features_size = 0

        # TODO: implement source features
        if opt.start_symbol == 0:
            self.source_l = self.source_l - 2
            # TODO: why do we truncate the last column?
            self.source = self.source[: , 1:self.source.shape[1]-1]
            # TODO: implement source features
        
        self.batch_idx = f['batch_idx'][:]
 
        self.target_size = f['target_size'][:][0]
        self.source_size = f['source_size'][:][0]
        self.target_nonzeros = f['target_nonzeros'][:]
 
        # TODO: Implement guided_alignment    
        # TODO: Implement opt.use_chars_enc
        # TODO: Implement opt.use_chars_dec
 
        self.length = self.batch_l.shape[0]
        self.seq_length = self.target.shape[1]
        self.batches = []
        
        max_source_l = max(self.source_l)
        source_l_rev = np.ones((max_source_l,1), dtype=np.int32)
        # One based to zero based
        self.batch_idx = self.batch_idx - 1
        # TODO: Should this be zero based or one based?
        for i in range(max_source_l):
          source_l_rev[i] = max_source_l - i

        for i in range(self.length):
            target_output_i = self.target_output[self.batch_idx[i]:self.batch_idx[i] + self.batch_l[i], 0:self.target_l[i]]
            target_l_i = self.target_l_all[self.batch_idx[i]:self.batch_idx[i]+self.batch_l[i]]
            # TODO: implement opt.use_chars_enc
            source_i = self.source[self.batch_idx[i]:self.batch_idx[i]+self.batch_l[i], 0:self.source_l[i]].T

            # TODO: Implement opt.reverse_src
            # TODO: Implement opt.use_chars_dec
 
            target_i = self.target[self.batch_idx[i]:self.batch_idx[i]+self.batch_l[i], 0:self.target_l[i]].T

            source_feats = {}
            target_feats = {}
            target_feats_output = {}
 
            # TODO: Implement source features

            # convert table of timesteps per feature to a table of features per timestep
            source_features_i = self.features_per_timestep(source_feats)
 
            alignment_i = None
            # TODO: Implement guided alignment
 
            self.batches.append([target_i, target_output_i.T, self.target_nonzeros[i], source_i, self.batch_l[i], self.target_l[i], self.source_l[i], target_l_i, source_features_i, alignment_i])
        return
    
    def features_per_timestep(self, features):
        # TODO: Implement this
      return None

    
def parse_arguments():
    ap = ArgumentParser()
    # data files
    ap.add_argument('--data_file', default='data/demo-train.hdf5', help='Path to the training *.hdf5 file from preprocess.py')
    ap.add_argument('--val_data_file', default='data/demo-val.hdf5', help='Path to validation *.hdf5 file from preprocess.py')
    ap.add_argument('--savefile', default='seq2seq_lstm_attn', help='Savefile name (model will be saved as savefile_epochX_PPL.t7 where X is the X-th epoch and PPL is the validation perplexity')
    ap.add_argument('--num_shards', type=int, default=0, help='If the training data has been broken up into different shards, then training files are in this many partitions')
    ap.add_argument('--train_from', default='', help='If training from a checkpoint then this is the path to the pretrained model.')
    # rnn model specs
    ap.add_argument('--num_layers', type=int, default=2, help='Number of layers in the LSTM encoder/decoder')
    ap.add_argument('--rnn_size', type=int, default=500, help='Size of LSTM hidden states')
    ap.add_argument('--word_vec_size', type=int, default=500, help='Word embedding sizes')
    # TODO: Should this be the last hidden state of the encoder?!
    ap.add_argument('--attn', type=int, default=1, help='If = 1, use attention on the decoder side. If = 0, it uses the last hidden state of the decoder as context at each time step.')
    ap.add_argument('--brnn', type=int, default=0, help='If = 1, use a bidirectional RNN. Hidden states of the fwd/bwd RNNs are summed.')
    ap.add_argument('--use_chars_enc', type=int, default=0, help='use character on the encoder side (instead of word embeddings')
    ap.add_argument('--use_chars_dec', type=int, default=0, help='If = 1, use character on the decoder side (instead of word embeddings')
    ap.add_argument('--reverse_src', type=int, default=0, help='reverse the source sequence. The original sequence-to-sequence paper found that this was crucial to achieving good performance, but with attention models this does not seem necessary. Recommend leaving it to 0')
    ap.add_argument('--init_dec', type=int, default=1, help='Initialize the hidden/cell state of the decoder at time 0 to be the last hidden/cell state of the encoder. If 0, the initial states of the decoder are set to zero vectors')
    ap.add_argument('--input_feed', type=int, default=1, help='If = 1, feed the context vector at each time step as additional input (via concatenation with the word embeddings) to the decoder')
    ap.add_argument('--multi_attn', type=int, default=0, help='If > 0, then use a another attention layer on this layer of the decoder. For example, if num_layers = 3 and `multi_attn = 2`, then the model will do an attention over the source sequence on the second layer (and use that as input to the third layer) and the penultimate layer')
    ap.add_argument('--res_net', type=int, default=0, help='Use residual connections between LSTM stacks whereby the input to the l-th LSTM layer if the hidden state of the l-1-th LSTM layer added with the l-2th LSTM layer. We did not find this to help in our experiments')
    ap.add_argument('--guided_alignment', type=int, default=0, help='If 1, use external alignments to guide the attention weights as in (Chen et al., Guided Alignment Training for Topic-Aware Neural Machine Translation, arXiv 2016.). Alignments should have been provided during preprocess')
    ap.add_argument('--guided_alignment_weight', type=float, default=0.5, help='default weights for external alignments')
    ap.add_argument('--guided_alignment_decay', type=float, default=1, help='decay rate per epoch for alignment weight - typical with 0.9, weight will end up at ~30% of its initial value')
    # char-cnn model specs (if use_chars == 1)
    ap.add_argument('--char_vec_size', type=int, default=25, help='Size of the character embeddings')
    ap.add_argument('--kernel_width', type=int, default=6, help='Size (i.e. width) of the convolutional filter')
    ap.add_argument('--num_kernels', type=int, default=1000, help='Number of convolutional filters (feature maps). So the representation from characters will have this many dimensions')
    ap.add_argument('--num_highway_layers', type=int, default=2, help='Number of highway layers in the character model')
    # optimization
    ap.add_argument('--epochs', type=int, default=13, help='Number of training epochs')
    ap.add_argument('--start_epoch', type=int, default=0, help='If loading from a checkpoint, the epoch from which to start')
    ap.add_argument('--param_init', type=float, default=0.1, help='Parameters are initialized over uniform distribution with support (-param_init, param_init)')
    ap.add_argument('--optim', default='sgd', help='Optimization method. Possible options are: sgd (vanilla SGD), adagrad, adadelta, adam')
    ap.add_argument('--learning_rate', type=float, default=1.0, help='Starting learning rate. If adagrad/adadelta/adam is used, then this is the global learning rate. Recommended settings: sgd =1, adagrad = 0.1, adadelta = 1, adam = 0.1')
    ap.add_argument('--max_grad_norm', type=float, default=5.0, help='If the norm of the gradient vector exceeds this re-normalize it to have the norm equal to max_grad_norm')
    ap.add_argument('--dropout', type=float, default=0.3, help='Dropout probability. Dropout is applied between vertical LSTM stacks.')
    ap.add_argument('--lr_decay', type=float, default=0.5, help='Decay learning rate by this much if (i) perplexity does not decrease on the validation set or (ii) epoch has gone past the start_decay_at_limit')
    ap.add_argument('--start_decay_at', type=int, default=9, help='Start decay after this epoch')
    ap.add_argument('--curriculum', type=int, default=0, help='For this many epochs, order the minibatches based on source sequence length. Sometimes setting this to 1 will increase convergence speed.')
    ap.add_argument('--feature_embeddings_dim_exponent', type=float, default=0.7, help='If the feature takes N values, then the embbeding dimension will be set to N^exponent')
    ap.add_argument('--pre_word_vecs_enc', default='', help='If a valid path is specified, then this will load pretrained word embeddings (hdf5 file) on the encoder side. See README for specific formatting instructions.')
    ap.add_argument('--pre_word_vecs_dec', default='', help='If a valid path is specified, then this will load pretrained word embeddings (hdf5 file) on the decoder side. See README for specific formatting instructions.')
    ap.add_argument('--fix_word_vecs_enc', type=int, default=0, help='If = 1, fix word embeddings on the encoder side')
    ap.add_argument('--fix_word_vecs_dec', type=int, default=0, help='If = 1, fix word embeddings on the decoder side')
    ap.add_argument('--max_batch_l', default='', help='If blank, then it will infer the max batch size from validation data. You should only use this if your validation set uses a different batch size in the preprocessing step')
    # Other options
    ap.add_argument('--start_symbol', type=int, default=0, help='Use special start-of-sentence and end-of-sentence tokens on the source side. We have found this to make minimal difference')
    # GPU
    ap.add_argument('--gpuid', type=int, default=-1, help='Which gpu to use. -1 = use CPU')
    ap.add_argument('--gpuid2', type=int, default=-1, help='If this is >= 0, then the model will use two GPUs whereby the encoder is on the first GPU and the decoder is on the second GPU. This will allow you to train with bigger batches/models.')
    ap.add_argument('--cudnn', type=int, default=0, help='Whether to use cudnn or not for convolutions (for the character model). cudnn has much faster convolutions so this is highly recommended if using the character model')
    # bookkeeping
    ap.add_argument('--save_every', type=int, default=1, help='Save every this many epochs')
    ap.add_argument('--print_every', type=int, default=50, help='Print stats after this many batches')
    ap.add_argument('--seed', type=int, default=3435, help='Seed for random initialization')
    ap.add_argument('--prealloc', type=int, default=1, help='Use memory preallocation and sharing between cloned encoder/decoders')
    args = ap.parse_args()
    return args

class LstmEncoder(Chain):
  def __init__(self, in_size, n_out):
    super(LstmEncoder, self).__init__(
      lstm = L.LSTM(in_size, n_out))

  def __call__(self, x):
    batch_size = x.shape[0]
    n_words = x.shape[1]
    embed_dim = x.shape[2]
    self.lstm.reset_state()
    for w_id in range(n_words):
      word_batch = x[:, w_id, :]
      encoding = self.lstm(word_batch)
    return encoding

class LstmDecoder(Chain):
  def __init__(self, in_size, n_vocab):
    super(LstmDecoder, self).__init__(
      lstm = L.LSTM(in_size, in_size),
      lin = L.Linear(in_size, n_vocab))

  def get_prediction(self, word_id, valid_ids, word_id_batch, predictions):
    batch_size = valid_ids.shape[0]
    for word in range(batch_size):
      valids = valid_ids[word, :]
      n_words = sum(valids != 0)
      if word_id >= n_words:
        continue
      scores = word_id_batch[word, :]
      valid_scores = np.take(scores, valids[word_id:n_words])
      max_id = np.argmax(valid_scores)
      real_valids = valids[word_id:n_words]
      selection = np.take(real_valids, max_id)
      predictions[word, word_id] = selection
      # swap
      tmp = valid_ids[word, word_id]
      valid_ids[word, word_id] = selection
      new_index = max_id + word_id
      valid_ids[word, new_index] = tmp

  def get_bleu(self, truth, prediction):
    assert(truth.shape == prediction.shape)
    batch_size = truth.shape[0]
    bleu = 0
    ref_list = []
    hyp_list = []
    for sentence in range(batch_size):
      n_words = sum(truth[sentence, :] != 0)
      reference =  truth[sentence, 0:n_words]
      hypothesis = prediction[sentence, 0:n_words]
      ref_list.append([reference])
      hyp_list.append(hypothesis)
    chencherry = SmoothingFunction()
    bleu = corpus_bleu(ref_list, hyp_list)
#       bleu += sentence_bleu([[str(x) for x in reference.tolist()]], [str(x) for x in hypothesis.tolist()], smoothing_function=chencherry.method2)
#     return bleu * (1.0 / batch_size)
    return bleu

  def __call__(self, x, y, embeder, lstm):
    self.lstm = lstm
#     self.lstm.reset_state()
    loss = 0
    decode_length = y.shape[1]
    # TODO: convert to cupy
    valid_ids = np.array(cp.asnumpy(y.data))
    predictions = np.zeros_like(valid_ids)
    decoder_input = x
    for word_id in range(decode_length):
      # TODO handle test time behavior
      # TODO This is wrong, we should use truth embedding at train time, and
      # and our own predictions at test time, also the decoder should be
      # initialized properly
      word_batch = self.lstm(decoder_input)
      word_id_batch = self.lin(word_batch)
      truth = y[:, word_id]
      loss += softmax_cross_entropy.softmax_cross_entropy(word_id_batch, truth)
      self.get_prediction(word_id, valid_ids, cp.asnumpy(word_id_batch.data), predictions)
      # Now the decoder input should be the embedding for the true word
      decoder_input = embeder(truth)

    bleu = self.get_bleu(y.data, predictions)
    return (loss * (1.0 / decode_length), bleu)

class Seq2SeqModel(Chain):
  def __init__(self, n_vocab, n_embed):
    super(Seq2SeqModel, self).__init__(
      embed   = L.EmbedID(n_vocab, n_embed),
      encoder = LstmEncoder(n_embed, n_embed),
      decoder = LstmDecoder(n_embed, n_vocab))

  def __call__(self, x, t):
    y = self.embed(x)
    z = self.encoder(y)
    n_words = x.shape[1]
    loss, bleu = self.decoder(z, t, self.embed, self.encoder.lstm)
    reporter.report({'loss': loss}, self)
    reporter.report({'bleu': bleu}, self)
    return loss

class Encoder(Chain):
    def __init__(self, data, opt, use_chars):
        assert(model == 'enc')
        self.data = data
        self.opt = opt
        self.use_chars = use_chars

    def forward(self, inputs):
        model = 'enc'
        name = '_' + model
        # TODO: or 0?!
        dropout = self.opt.dropout
        n = self.opt.num_layers
        rnn_size = self.opt.rnn_size
        RnnD = [self.opt.rnn_size, self.opt.rnn_size]
        input_size = None
        if self.use_chars == 0:
            input_size = self.opt.word_vec_size
        else:
            input_size = self.opt.num_kernels
        offset = 0
        # TODO decoder
        # TODO: num_source_features
         
        x = None
        input_size_L = None
        outputs = []
        print 'offset: ', offset
        for l in range(n):
            nameL = model + '_L' + str(l) + '_'
            # c,h from previous timesteps
            prev_c = inputs[l*2+1+offset]
            prev_h = inputs[l*2+2+offset]
            # the input to this layer
            if l == 0:
                if self.use_chars == 0:
                    word_vecs = None
                    if model == 'enc':
                      word_vecs = L.EmbedID(self.data.source_size, input_size)
                    else:
                      word_vecs = L.EmbedID(self.data.target_size, input_size)
                    word_vecs.name = 'word_vecs' + name
                    x = word_vecs(inputs[0]) # batch_size x word_vec_size
                # TODO: use_chars
                # TODO source num features
                input_size_L = input_size
                # TODO: decoder
                input_size_L = input_size_L + self.data.total_source_features_size
            else:
                x = outputs[(l-1)*2]
                if self.opt.res_net == 1 and l > 2:
                    x = x + outputs[(l-2)*2]
                input_size_L = rnn_size
                # TODO: decoder
    #             if dropout > 0:
    #                 x = nn.Dropout(dropout, nil, false):usePrealloc(nameL.."dropout", {{self.opt.max_batch_l, input_size_L}})(x)
            # evaluate the input sums at once for efficiency
            i2h = L.Linear(input_size_L, 4 * rnn_size)(x)
            # TODO Why don't we use bias here?
            h2h = L.Linear(rnn_size, 4 * rnn_size, nobias=True)(prev_h)
            all_input_sums = i2h + h2h
               
            reshaped = F.reshape(all_input_sums, (4, rnn_size))
            n1, n2, n3, n4 = F.split_axis(reshaped, 4, axis=2) 
            # decode the gates
            in_gate = F.sigmoid(n1)
            forget_gate = F.sigmoid(n2)
            out_gate = F.sigmoid(n3)
            # decode the write inputs
            in_transform = F.tanh(n4)
            # perform the LSTM update
            next_c = (forget_gate * prev_c) + (in_gate * in_transform)
            # gated cells form the output
            next_h = out_gate * F.tanh(next_c)
              
            outputs.append(next_c)
            outputs.append(next_h)
        # TODO: guided_allignment
        print 'inputs: ', inputs, ' outputs: ', outputs
        return outputs
    
def make_lstm(data, opt, model, use_chars):
    assert(model == 'enc' or model == 'dec')
    if model == 'enc':
        return Encoder(data, opt, use_chars)
    name = '_' + model
    # TODO: or 0?!
    dropout = opt.dropout
    n = opt.num_layers
    rnn_size = opt.rnn_size
    RnnD = [opt.rnn_size, opt.rnn_size]
    input_size = None
    if use_chars == 0:
        input_size = opt.word_vec_size
    else:
        input_size = opt.num_kernels
    offset = 0
    # there will be 2*n+1 inputs for encoder
    inputs = []
    inputs.append(F.Identity) # x (batch_size x max_word_l)
    if model == 'dec':
        inputs.append(F.Identity) # all context (batch_size x source_l x rnn_size)
        offset = offset + 1
        if opt.input_feed == 1:
            inputs.append(F.Identity) # prev context_attn (batch_size x rnn_size)
            offset = offset + 1
    # TODO: num_source_features
    
    for l in range(n):
        inputs.append(F.Identity) # prev_c[l]
        inputs.append(F.Identity) # prev_h[l]
     
    x = None
    input_size_L = None
    outputs = []
    print 'offset: ', offset
    for l in range(n):
        nameL = model + '_L' + str(l) + '_'
        # c,h from previous timesteps
        prev_c = inputs[l*2+1+offset]
        prev_h = inputs[l*2+2+offset]
        # the input to this layer
        if l == 0:
            if use_chars == 0:
                word_vecs = None
                if model == 'enc':
                  word_vecs = L.EmbedID(data.source_size, input_size)
                else:
                  word_vecs = L.EmbedID(data.target_size, input_size)
                word_vecs.name = 'word_vecs' + name
                x = lambda b: word_vecs(inputs[0](b)) # batch_size x word_vec_size
            # TODO: use_chars
            # TODO source num features
            input_size_L = input_size
            if model == 'dec':
                if opt.input_feed == 1:
                    # TODO: Can we use memory pre-allocation with chainer?
                    # TODO: offset has to be fixed
                    x = F.Concat(2)((x, inputs[offset])) # batch_size x (word_vec_size + rnn_size)
                    input_size_L = input_size_L + rnn_size
            else:
                input_size_L = input_size_L + data.total_source_features_size
        else:
            x = outputs[(l-1)*2]
            if opt.res_net == 1 and l > 2:
                x = x + outputs[(l-2)*2]
            input_size_L = rnn_size
            if opt.multi_attn == l and model == 'dec':
                multi_attn = make_decoder_attn(data, opt, 1)
                multi_attn.name = 'multi_attn' + l
                x = multi_attn({x, inputs[1]})
#             if dropout > 0:
#                 x = nn.Dropout(dropout, nil, false):usePrealloc(nameL.."dropout", {{opt.max_batch_l, input_size_L}})(x)
        # evaluate the input sums at once for efficiency
        i2h = L.Linear(input_size_L, 4 * rnn_size)(x)
        # TODO Why don't we use bias here?
        h2h = L.Linear(rnn_size, 4 * rnn_size, nobias=True)(prev_h)
        all_input_sums = i2h + h2h
           
        reshaped = F.reshape(all_input_sums, (4, rnn_size))
        n1, n2, n3, n4 = F.split_axis(reshaped, 4, axis=2) 
        # decode the gates
        in_gate = F.sigmoid(n1)
        forget_gate = F.sigmoid(n2)
        out_gate = F.sigmoid(n3)
        # decode the write inputs
        in_transform = F.tanh(n4)
        # perform the LSTM update
        next_c = (forget_gate * prev_c) + (in_gate * in_transform)
        # gated cells form the output
        next_h = out_gate * F.tanh(next_c)
          
        outputs.append(next_c)
        outputs.append(next_h)
    if model == 'dec':
        top_h = outputs[-1]
        decoder_out = None
        attn_output = None
        if opt.attn == 1:
            decoder_attn = make_decoder_attn(data, opt)
            decoder_attn.name = 'decoder_attn'
            if opt.guided_alignment == 1:
                # TODO Implement the attention function
                decoder_out, attn_output = F.split_axis(decoder_attn({top_h, inputs[2]}), 2)
            else:
                decoder_out = decoder_attn({top_h, inputs[2]})
        else:
            # TODO: Fix indices
            decoder_out = F.Concat(2)((top_h, inputs[1]))
            decoder_out = F.Tanh(F.Linear(opt.rnn_size*2, opt.rnn_size, nobiad=True)(decoder_out))
        if dropout > 0:
            # TODO Fix dropout input
            decoder_out = L.Dropout(dropout, nil, false)
            outputs.append(decoder_out)
    # TODO: guided_allignment
    print 'inputs: ', inputs, ' outputs: ', outputs
    return None
    
def train(train_data, valid_data, opt, layers):
    timer = None
    num_params = 0
    num_prunedparams = 0
    start_decay = 0
    params = []
    grad_params = [] 
    opt.train_perf = []
    opt.val_perf = {}

#    for i in range(len(layers)):
#        # TODO: Implement gpu
#        p, gp = layers[i]:getParameters()
#        if len(opt.train_from) == 0:
#            p:uniform(-opt.param_init, opt.param_init)
#        num_params = num_params + p:size(1)
#        params[i] = p
#        grad_params[i] = gp
#        layers[i]:apply(function (m) if m.nPruned then num_prunedparams=num_prunedparams+m:nPruned() end end)

    # TODO: opt.pre_word_vecs_enc
    # TODO: opt.pre_word_vecs_dec
    # TODO: opt.brnn 
    print 'Number of parameters: ', num_params, ' (active: ', (num_params - num_prunedparams), ')'

    # TODO: GPU
#    word_vec_layers[1].weight[1]:zero()
#    word_vec_layers[2].weight[1]:zero()
    # TODO: opt.brnn

    # prototypes for gradients so there is no need to clone
    encoder_grad_proto = np.zeros((opt.max_batch_l, opt.max_sent_l, opt.rnn_size))
    encoder_bwd_grad_proto = np.zeros((opt.max_batch_l, opt.max_sent_l, opt.rnn_size))
    context_proto = np.zeros((opt.max_batch_l, opt.max_sent_l, opt.rnn_size))
    # TODO: opt.gpuid2

    # clone encoder/decoder up to max source/target length
#  decoder_clones = clone_many_times(decoder, opt.max_sent_l_targ)
#  encoder_clones = clone_many_times(encoder, opt.max_sent_l_src)
    # TODO: opt.brnn
#  for i = 1, opt.max_sent_l_src do
#    if encoder_clones[i].apply then
#      encoder_clones[i]:apply(function(m) m:setReuse() end)
#    end
    # TODO: opt.brnn 
#  end
#  for i = 1, opt.max_sent_l_targ do
#    if decoder_clones[i].apply then
#      decoder_clones[i]:apply(function(m) m:setReuse() end)
#    end
#  end

    # TODO xp    
    h_init = np.zeros((opt.max_batch_l, opt.rnn_size), dtype = np.float32)
    attn_init = np.zeros((opt.max_batch_l, opt.max_sent_l))
    # TODO GPU

    # these are initial states of encoder/decoder for fwd/bwd steps
    init_fwd_enc = []
    init_bwd_enc = []
    init_fwd_dec = []
    init_bwd_dec = []

    for L in range(opt.num_layers):
        init_fwd_enc.append(h_init.copy())
        init_fwd_enc.append(h_init.copy())
        init_bwd_enc.append(h_init.copy())
        init_bwd_enc.append(h_init.copy())

    # TODO: opt.gpuid2
    if opt.input_feed == 1:
        init_fwd_dec.append(h_init.copy())
    init_bwd_dec.append(h_init.copy())
    # TODO Move this to a separate function
    for L in range(opt.num_layers):
        init_fwd_dec.append(h_init.copy())
        init_fwd_dec.append(h_init.copy())
        init_bwd_dec.append(h_init.copy())
        init_bwd_dec.append(h_init.copy())

    dec_offset = 3 # offset depends on input feeding
    if opt.input_feed == 1:
        dec_offset = dec_offset + 1

    def reset_state(state, batch_l, t):
        if t == None:
            u = []
            for i in range(len(state)): 
                state[i].fill(0)
                u.append(state[i][0:batch_l])
            return u
        else:
            u = {t: []}
            for i in range(len(state)):
                state[i].fill(0)
                u[t].append(state[i][0:batch_l])
            return u

#  # clean layer before saving to make the model smaller
#  function clean_layer(layer)
    # TODO: opt.gpuid
#    layer.output = torch.DoubleTensor()
#    layer.gradInput = torch.DoubleTensor()
#    if layer.modules then
#      for i, mod in ipairs(layer.modules) do
#        clean_layer(mod)
#      end
#    elseif torch.type(self) == "nn.gModule" then
#      layer:apply(clean_layer)
#    end
#  end

#  # decay learning rate if val perf does not improve or we hit the opt.start_decay_at limit
#  function decay_lr(epoch)
#    print(opt.val_perf)
#    if epoch >= opt.start_decay_at then
#      start_decay = 1
#    end
#
#    if opt.val_perf[#opt.val_perf] ~= nil and opt.val_perf[#opt.val_perf-1] ~= nil then
#      local curr_ppl = opt.val_perf[#opt.val_perf]
#      local prev_ppl = opt.val_perf[#opt.val_perf-1]
#      if curr_ppl > prev_ppl then
#        start_decay = 1
#      end
#    end
#    if start_decay == 1 then
#      opt.learning_rate = opt.learning_rate * opt.lr_decay
#    end
#  end
#
    def train_batch(data, epoch):
        opt.num_source_features = data.num_source_features

        train_nonzeros = 0
        train_loss = 0
        train_loss_cll = 0
        batch_order = np.random.permutation(data.length) # shuffle mini batch order
        start_time = time.time()
        num_words_target = 0
        num_words_source = 0

        for i in range(data.length):
            print i 
            # TODO zero grads?
            # zero_table(grad_params, 'zero')
            # TODO: opt.curriculum
            d = data[batch_order[i]]
            print d
            target, target_out, nonzeros, source = d[0], d[1], d[2], d[3]
            print 'target: ', target
            print 'target_out: ', target_out
            print 'nonzeros: ', nonzeros
            print 'source: ', source
            batch_l, target_l, source_l = d[4], d[5], d[6]
            print 'batch_l: ', batch_l
            print 'target_l: ', target_l
            print 'source_l: ', source_l
            source_features = d[8]
            print 'source_features: ', source_features
            alignment = d[9]
            print 'alignment: ', alignment
            norm_alignment = None
            # TODO: opt.guided_alignment
            encoder_grads = encoder_grad_proto[0:batch_l, 0:source_l]
            encoder_bwd_grads = None
            # TODO: opt.brnn
            # TODO: opt.gpuid 
            rnn_state_enc = reset_state(init_fwd_enc, batch_l, -1)
            context = context_proto[0:batch_l, 0:source_l]
            # forward prop encoder
            for t in range(source_l):
                print 'source_l'
                # TODO: set traiining to True, this is important for dropout
#                encoder_clones[t]:training()
                encoder_input = [source[t]]
                # TODO: data.num_source_features
                # TODO: Is the index here correct?
                encoder_input += rnn_state_enc[t-1]
                print 'here'
#                out = encoder_clones[t]:forward(encoder_input)
#                rnn_state_enc[t] = out
#                context[:,t]:copy(out[len(out)])
            print 'hello train_batch'

            rnn_state_enc_bwd = None
            # TODO opt.brnn
            # TODO: opt.gpuid opt.gpuid2
            # copy encoder last hidden state to decoder initial state
            rnn_state_dec = reset_state(init_fwd_dec, batch_l, 0)
#      if opt.init_dec == 1 then
#        for L = 1, opt.num_layers do
#          rnn_state_dec[0][L*2-1+opt.input_feed]:copy(rnn_state_enc[source_l][L*2-1])
#          rnn_state_dec[0][L*2+opt.input_feed]:copy(rnn_state_enc[source_l][L*2])
#        end
        # TODO: opt.brnn
#      end
#      # forward prop decoder
#      local preds = {}
#      local attn_outputs = {}
#      local decoder_input
#      for t = 1, target_l do
#        decoder_clones[t]:training()
#        local decoder_input
#        if opt.attn == 1 then
#          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
#        else
#          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
#        end
#        local out = decoder_clones[t]:forward(decoder_input)
#        local out_pred_idx = #out
        # TODO: opt.guided_alignment 
#        local next_state = {}
#        table.insert(preds, out[out_pred_idx])
#        if opt.input_feed == 1 then
#          table.insert(next_state, out[out_pred_idx])
#        end
#        for j = 1, out_pred_idx-1 do
#          table.insert(next_state, out[j])
#        end
#        rnn_state_dec[t] = next_state
#      end

#      # backward prop decoder
#      encoder_grads:zero()
    # TODO: opt.brnn
#      local drnn_state_dec = reset_state(init_bwd_dec, batch_l)
    # TODO: opt.guided_alignment
#      local loss = 0
#      local loss_cll = 0
#      for t = target_l, 1, -1 do
#        local pred = generator:forward(preds[t])
#
#        local input = pred
#        local output = target_out[t]
        # TODO: opt.guided_alignment
#
#        loss = loss + criterion:forward(input, output)/batch_l
#
#        local drnn_state_attn
#        local dl_dpred
        # TODO: opt.guided_alignment
#        dl_dpred = criterion:backward(input, output)
#
#        dl_dpred:div(batch_l)
#        local dl_dtarget = generator:backward(preds[t], dl_dpred)
#
#        local rnn_state_dec_pred_idx = #drnn_state_dec
        # TODO: opt.guided_alignment
#        drnn_state_dec[rnn_state_dec_pred_idx]:add(dl_dtarget)
#
#        local decoder_input
#        if opt.attn == 1 then
#          decoder_input = {target[t], context, table.unpack(rnn_state_dec[t-1])}
#        else
#          decoder_input = {target[t], context[{{}, source_l}], table.unpack(rnn_state_dec[t-1])}
#        end
#        local dlst = decoder_clones[t]:backward(decoder_input, drnn_state_dec)
#        # accumulate encoder/decoder grads
#        if opt.attn == 1 then
#          encoder_grads:add(dlst[2])
        # TODO: opt.brnn
#        else
#          encoder_grads[{{}, source_l}]:add(dlst[2])
        # TODO: opt.brnn 
#        end

#        drnn_state_dec[rnn_state_dec_pred_idx]:zero()
        # TODO opt.guided_alignment
#        if opt.input_feed == 1 then
#          drnn_state_dec[rnn_state_dec_pred_idx]:add(dlst[3])
#        end
#        for j = dec_offset, #dlst do
#          drnn_state_dec[j-dec_offset+1]:copy(dlst[j])
#        end
#      end
#      word_vec_layers[2].gradWeight[1]:zero()
    # TODO: opt.fix_word_vecs_dec 
#      grad_norm = 0
#      grad_norm = grad_norm + grad_params[2]:norm()^2 + grad_params[3]:norm()^2
#
#      # backward prop encoder
    # TODO: opt.gpuid  opt.gpuid2 
#      local drnn_state_enc = reset_state(init_bwd_enc, batch_l)
#      if opt.init_dec == 1 then
#        for L = 1, opt.num_layers do
#          drnn_state_enc[L*2-1]:copy(drnn_state_dec[L*2-1])
#          drnn_state_enc[L*2]:copy(drnn_state_dec[L*2])
#        end
#      end
#
#      for t = source_l, 1, -1 do
#        local encoder_input = {source[t]}
        # TODO: num_source_features
#        append_table(encoder_input, rnn_state_enc[t-1])
#        if opt.attn == 1 then
#          drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
#        else
#          if t == source_l then
#            drnn_state_enc[#drnn_state_enc]:add(encoder_grads[{{},t}])
#          end
#        end
#        local dlst = encoder_clones[t]:backward(encoder_input, drnn_state_enc)
#        for j = 1, #drnn_state_enc do
#          drnn_state_enc[j]:copy(dlst[j+1+data.num_source_features])
#        end
#      end
#
        # TODO: opt.brnn 
#
#      word_vec_layers[1].gradWeight[1]:zero()
    # TODO: opt.fix_word_vecs_enc
#      
#      grad_norm = grad_norm + grad_params[1]:norm()^2
    # TODO: opt.brnn
#      grad_norm = grad_norm^0.5
#      -- Shrink norm and update params
#      local param_norm = 0
#      local shrinkage = opt.max_grad_norm / grad_norm
#      for j = 1, #grad_params do
#        if opt.gpuid >= 0 and opt.gpuid2 >= 0 then
#          if j == 1 then
#            cutorch.setDevice(opt.gpuid)
#          else
#            cutorch.setDevice(opt.gpuid2)
#          end
#        end
#        if shrinkage < 1 then
#          grad_params[j]:mul(shrinkage)
#        end
#        if opt.optim == 'adagrad' then
#          adagrad_step(params[j], grad_params[j], layer_etas[j], optStates[j])
#        elseif opt.optim == 'adadelta' then
#          adadelta_step(params[j], grad_params[j], layer_etas[j], optStates[j])
#        elseif opt.optim == 'adam' then
#          adam_step(params[j], grad_params[j], layer_etas[j], optStates[j])
#        else
#          params[j]:add(grad_params[j]:mul(-opt.learning_rate))
#        end
#        param_norm = param_norm + params[j]:norm()^2
#      end
#      param_norm = param_norm^0.5
    # TODO: opt.brnn
#      # Bookkeeping
#      num_words_target = num_words_target + batch_l*target_l
#      num_words_source = num_words_source + batch_l*source_l
#      train_nonzeros = train_nonzeros + nonzeros
#      train_loss = train_loss + loss*batch_l
    # TODO: opt.guided_alignment
#      local time_taken = timer:time().real - start_time
#      if i % opt.print_every == 0 then
#        local stats = string.format('Epoch: %d, Batch: %d/%d, Batch size: %d, LR: %.4f, ',
#          epoch, i, data:size(), batch_l, opt.learning_rate)
#        if opt.guided_alignment == 1 then
#          stats = stats .. string.format('PPL: %.2f, PPL_CLL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
#            math.exp(train_loss/train_nonzeros), math.exp(train_loss_cll/train_nonzeros), param_norm, grad_norm)
#        else
#          stats = stats .. string.format('PPL: %.2f, |Param|: %.2f, |GParam|: %.2f, ',
#            math.exp(train_loss/train_nonzeros), param_norm, grad_norm)
#        end
#        stats = stats .. string.format('Training: %d/%d/%d total/source/target tokens/sec',
#          (num_words_target+num_words_source) / time_taken,
#          num_words_source / time_taken,
#          num_words_target / time_taken)
#        print(stats)
#      end
#      if i % 200 == 0 then
#        collectgarbage()
#      end
#    end
############################## end of for
    # TODO: opt.guided_alignment
#    return train_loss, train_nonzeros
#
    print 'here'
    for epoch in range(opt.start_epoch, opt.epochs): 
        #generator:training()
        # TODO: opt.num_shards
        # TODO: opt.guided_alignment
        total_loss, total_nonzeros = train_batch(train_data, epoch)
#        local train_score = math.exp(total_loss/total_nonzeros)
#        print('Train', train_score)
#        opt.train_perf[#opt.train_perf + 1] = train_score
#        local score = eval(valid_data)
#        opt.val_perf[#opt.val_perf + 1] = score
#        if opt.optim == 'sgd' then --only decay with SGD
#          decay_lr(epoch)
#        end
#       # TODO: opt.guided_alignment 
#        -- clean and save models
#        local savefile = string.format('%s_epoch%.2f_%.2f.t7', opt.savefile, epoch, score)
#        if epoch % opt.save_every == 0 then
#          print('saving checkpoint to ' .. savefile)
#          clean_layer(generator)
#          if opt.brnn == 0 then
#            torch.save(savefile, {{encoder, decoder, generator}, opt})
#          else
#            torch.save(savefile, {{encoder, decoder, generator, encoder_bwd}, opt})
#          end
#        end

    # save final model
#  local savefile = string.format('%s_final.t7', opt.savefile)
#  clean_layer(generator)
#  print('saving final model to ' .. savefile)
    # TODO opt.brnn
#  torch.save(savefile, {{encoder:double(), decoder:double(), generator:double(),
#          encoder_bwd:double()}, opt})

def main():
    # parse input params
    opt = parse_arguments()
    np.random.seed(opt.seed)
    cp.random.seed(opt.seed)
    
    if opt.gpuid >= 0:    
        print 'using CUDA on GPU ' + str(opt.gpuid) + '...'
        if opt.gpuid2 >= 0:
            print 'using CUDA on second GPU ' + str(opt.gpuid2) + '...'
        # TODO: Do we need to do something special for cudnn?!
        # TODO: Do we need to do something special to set the GPU device?

    # Create the data loader class.
    print 'loading data...'
    if opt.num_shards == 0 :
        train_data = Data(opt, opt.data_file)
    else:
        # TODO: Implement shards
        exit(0)

    valid_data = Data(opt, opt.val_data_file)
    print 'done!'
#   print(string.format('Source vocab size: %d, Target vocab size: %d',
#       valid_data.source_size, valid_data.target_size))
    opt.max_sent_l_src = valid_data.source.shape[1]
    opt.max_sent_l_targ = valid_data.target.shape[1]
    opt.max_sent_l = max(opt.max_sent_l_src, opt.max_sent_l_targ)
    if opt.max_batch_l == '':
        opt.max_batch_l = max(valid_data.batch_l)

    # TODO: Implement use_chars_enc and use_chars_dec
    
    print 'Source max sent len: ', valid_data.source.shape[1], ' Target max sent len: ', valid_data.target.shape[1]
    print 'Number of additional features on source side: ', valid_data.num_source_features
 
    # TODO: is there memory preallocation in chainer?
 
    # Build model
    if len(opt.train_from) == 0:
        encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
        decoder = None
#        decoder = make_lstm(valid_data, opt, 'dec', opt.use_chars_dec)
        generator = None
#        generator, criterion = make_generator(valid_data, opt)
        # TODO: Implement opt.brnn
    # TODO: Implement loading a pre-trained model

    # TODO: implement opt.guided_alignment
 
    layers = [encoder, decoder, generator]
    # TODO: Implement opt.brnn
    # TODO: Implement other optimization algorithms
    # TODO: Implement GPU support
    
    # these layers will be manipulated during training
    word_vec_layers = []
    # TODO: Implement opt.use_chars_enc
    
#   encoder:apply(get_layer)
#   decoder:apply(get_layer)
    # TODO: implement opt.brnn
    train(train_data, valid_data, opt, layers)
    
    
    train_data, valid_data, test_data = get_ordering_dataset()
    batch_size = 100
    
    PAD_ID = 0
    BOS_ID = 1
    EOS_ID = 2
    
    #   train_iter = MultiprocessIterator(train_data, batch_size, repeat=True,  shuffle=True)
    #   valid_iter = MultiprocessIterator(valid_data, batch_size, repeat=False, shuffle= False)
    #   test_iter  = MultiprocessIterator(test_data,  batch_size, repeat=False, shuffle=False)
    
    train_iter = SerialIterator(train_data, batch_size, repeat=True,  shuffle=True)
    #   valid_iter = SerialIterator(valid_data, batch_size, repeat=False, shuffle= False)
    valid_iter = SerialIterator(valid_data, len(valid_data), repeat=False, shuffle= False)
    valid_train_iter = SerialIterator(valid_data, batch_size, repeat=True, shuffle=True)
    test_iter  = SerialIterator(test_data,  batch_size, repeat=False, shuffle=False)
    
    max_length = 40
    
    
    #   train_iter = Seq2SeqIterator(np, train_data, batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle=True  )
    #   valid_iter = Seq2SeqIterator(np, valid_data, batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle= False)
    #   test_iter  = Seq2SeqIterator(np, test_data,  batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle=False )
    
    n_vocab    = 15947
    #   n_embed    = 500
    n_embed = 200
    
    model = Seq2SeqModel(n_vocab, n_embed)
    # TODO: convert to cupy
    #   model.to_gpu()
    optimizer = optimizers.SGD()
    optimizer.setup(model)
    converter = partial(convert.concat_examples, padding=PAD_ID)
    
    updater = training.StandardUpdater(valid_train_iter, optimizer, converter=converter)
    #   updater = training.StandardUpdater(train_iter, optimizer, converter=converter)
    
    #   updater = training.StandardUpdater(valid_iter, optimizer)
    
    trainer = training.Trainer(updater, (20, 'epoch'), out='result')
    
    trainer.extend(extensions.Evaluator(valid_iter, model, converter=converter))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'main/bleu', 'validation/main/loss', 'validation/main/bleu']))
    trainer.extend(extensions.ProgressBar())
    trainer.run()
    print 'done'

if __name__ == '__main__':
  main()
