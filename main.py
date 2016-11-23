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
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from operator import truth

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
  def __init__(self, in_size, n_vocab, decode_length):
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
    for sentence in range(batch_size):
      n_words = sum(truth[sentence, :] != 0)
      reference =  truth[sentence, 0:n_words]
      hypothesis = prediction[sentence, 0:n_words]
      print 'ref: ', reference
      print 'hyp: ', hypothesis
      chencherry = SmoothingFunction()
      bleu += sentence_bleu([[str(x) for x in reference.tolist()]], [str(x) for x in hypothesis.tolist()], smoothing_function=chencherry.method2)
    return bleu * (1.0 / batch_size)

  def __call__(self, x, y):
    self.lstm.reset_state()
    loss = 0
    decode_length = y.shape[1]
    # TODO: convert to cupy
    valid_ids = cp.asnumpy(y.data)
    predictions = np.zeros_like(valid_ids)
    for word_id in range(decode_length):
      word_batch = self.lstm(x)
      word_id_batch = self.lin(word_batch)
      truth = y[:, word_id]
      loss += softmax_cross_entropy.softmax_cross_entropy(word_id_batch, truth)
      self.get_prediction(word_id, valid_ids, cp.asnumpy(word_id_batch.data), predictions)
    bleu = self.get_bleu(y.data, predictions)
    return (loss * (1.0 / decode_length), bleu)

class Seq2SeqModel(Chain):
  def __init__(self, n_vocab, n_embed):
    super(Seq2SeqModel, self).__init__(
      embed   = L.EmbedID(n_vocab, n_embed),
      encoder = LstmEncoder(n_embed, n_embed),
      decoder = LstmDecoder(n_embed, n_vocab, 10))

  def __call__(self, x, t):
    y = self.embed(x)
    z = self.encoder(y)
    n_words = x.shape[1]
    loss, bleu = self.decoder(z, t)
    reporter.report({'loss': loss}, self)
    reporter.report({'bleu': bleu}, self)
    return loss

def main():
  train_data, valid_data, test_data = get_ordering_dataset()
  batch_size = 100

  PAD_ID = 0
  BOS_ID = 1
  EOS_ID = 2

#   train_iter = MultiprocessIterator(train_data, batch_size, repeat=True,  shuffle=True)
#   valid_iter = MultiprocessIterator(valid_data, batch_size, repeat=False, shuffle= False)
#   test_iter  = MultiprocessIterator(test_data,  batch_size, repeat=False, shuffle=False)

  train_iter = SerialIterator(train_data, batch_size, repeat=True,  shuffle=True)
  valid_iter = SerialIterator(valid_data, batch_size, repeat=False, shuffle= False)
  test_iter  = SerialIterator(test_data,  batch_size, repeat=False, shuffle=False)

  max_length = 40


#   train_iter = Seq2SeqIterator(np, train_data, batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle=True  )
#   valid_iter = Seq2SeqIterator(np, valid_data, batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle= False)
#   test_iter  = Seq2SeqIterator(np, test_data,  batch_size, max_length, BOS_ID, EOS_ID, PAD_ID, shuffle=False )

  n_vocab    = 15947
  n_embed    = 500

  model = Seq2SeqModel(n_vocab, n_embed)
  # TODO: convert to cupy
  model.to_gpu()
  optimizer = optimizers.SGD()
  optimizer.setup(model)
  converter = partial(convert.concat_examples, padding=PAD_ID)
  updater = training.StandardUpdater(train_iter, optimizer, converter=converter)

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
