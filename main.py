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
# function data:size()
#   return self.length
# end
# 
# function data.__index(self, idx)
#   if type(idx) == "string" then
#     return data[idx]
#   else
#     local target_input = self.batches[idx][1]
#     local target_output = self.batches[idx][2]
#     local nonzeros = self.batches[idx][3]
#     local source_input = self.batches[idx][4]
#     local batch_l = self.batches[idx][5]
#     local target_l = self.batches[idx][6]
#     local source_l = self.batches[idx][7]
#     local target_l_all = self.batches[idx][8]
#     local source_features = self.batches[idx][9]
#     local alignment = generate_aligns(self.batches[idx][10],
#                                       self.alignment_cc_colidx,
#                                       self.alignment_cc_val,
#                                       source_l,
#                                       target_l,
#                                       opt.start_symbol)
# 
#     if opt.gpuid >= 0 then --if multi-gpu, source lives in gpuid1, rest on gpuid2
#       cutorch.setDevice(opt.gpuid)
#       source_input = source_input:cuda()
#       source_features = features_on_gpu(source_features)
#       if opt.gpuid2 >= 0 then
#         cutorch.setDevice(opt.gpuid2)
#       end
#       target_input = target_input:cuda()
#       target_output = target_output:cuda()
#       target_l_all = target_l_all:cuda()
#       if opt.guided_alignment == 1 then
#         alignment = alignment:cuda()
#       end
#     end
#     return {target_input, target_output, nonzeros, source_input,
#       batch_l, target_l, source_l, target_l_all, source_features, alignment}
#   end
# end
# 
# return data

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
    ap.add_argument('--start_epoch', type=int, default=1, help='If loading from a checkpoint, the epoch from which to start')
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
#   if opt.train_from:len() == 0 then
#     encoder = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
#     decoder = make_lstm(valid_data, opt, 'dec', opt.use_chars_dec)
#     generator, criterion = make_generator(valid_data, opt)
#     if opt.brnn == 1 then
#       encoder_bwd = make_lstm(valid_data, opt, 'enc', opt.use_chars_enc)
#     end
#   else
#     assert(path.exists(opt.train_from), 'checkpoint path invalid')
#     print('loading ' .. opt.train_from .. '...')
#     local checkpoint = torch.load(opt.train_from)
#     local model, model_opt = checkpoint[1], checkpoint[2]
#     opt.num_layers = model_opt.num_layers
#     opt.rnn_size = model_opt.rnn_size
#     opt.input_feed = model_opt.input_feed or 1
#     opt.attn = model_opt.attn or 1
#     opt.brnn = model_opt.brnn or 0
#     encoder = model[1]
#     decoder = model[2]
#     generator = model[3]
#     if model_opt.brnn == 1 then
#       encoder_bwd = model[4]
#     end
#     _, criterion = make_generator(valid_data, opt)
#   end

    # TODO: implement opt.guided_alignment
 
#   layers = {encoder, decoder, generator}
#   if opt.brnn == 1 then
#     table.insert(layers, encoder_bwd)
#   end
# 
#   if opt.optim ~= 'sgd' then
#     layer_etas = {}
#     optStates = {}
#     for i = 1, #layers do
#       layer_etas[i] = opt.learning_rate -- can have layer-specific lr, if desired
#       optStates[i] = {}
#     end
#   end
# 
#   if opt.gpuid >= 0 then
#     for i = 1, #layers do
#       if opt.gpuid2 >= 0 then
#         if i == 1 or i == 4 then
#           cutorch.setDevice(opt.gpuid) --encoder on gpu1
#         else
#           cutorch.setDevice(opt.gpuid2) --decoder/generator on gpu2
#         end
#       end
#       layers[i]:cuda()
#     end
#     if opt.gpuid2 >= 0 then
#       cutorch.setDevice(opt.gpuid2) --criterion on gpu2
#     end
#     criterion:cuda()
#   end
# 
#   -- these layers will be manipulated during training
#   word_vec_layers = {}
#   if opt.use_chars_enc == 1 then
#     charcnn_layers = {}
#     charcnn_grad_layers = {}
#   end
#   encoder:apply(get_layer)
#   decoder:apply(get_layer)
#   if opt.brnn == 1 then
#     if opt.use_chars_enc == 1 then
#       charcnn_offset = #charcnn_layers
#     end
#     encoder_bwd:apply(get_layer)
#   end
#   train(train_data, valid_data)
    
    
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
