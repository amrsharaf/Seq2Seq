import random
import chainer
from collections import defaultdict

class Seq2SeqIterator(chainer.dataset.Iterator):
  def __init__(self, xp, dataset, batch_size, max_length, bos, eos, pad,
        shuffle=True):
    self.xp = xp
    self.batch_size = batch_size
    assert batch_size <= len(dataset)
    self.max_length = max_length
    self.bos = bos
    self.eos = eos
    self.pad = pad
    self.shuffle = shuffle

    self.create_batches(dataset, batch_size)
    self.size = len(self.batches)

    self.epoch = 0
    # It's easier to reason about the end instead of the beginning of epochs
    self.is_end_epoch = False
    self.iteration = 0
    self.batch_index = 0

  def preprocess_sentence(self, x, bucket_step):
    x.append(self.eos)
    while len(x) % bucket_step > 0:
      x.append(self.pad)
    return x
    
  def create_batches(self, dataset, batch_size, src_bucket_step=1,
      trg_bucket_step=1):
    self.batches = []
    # Buckets enable an efficient handling for padding
    buckets = defaultdict(list)
    for x, y in dataset:
      x = x[:self.max_length]
      x = self.preprocess_sentence(x, src_bucket_step)
      # We don't truncate the output to make sure evaluation is correct
      y = self.preprocess_sentence(y, trg_bucket_step)        
      buckets[len(x), len(y)].append((x, y))
    for samples in buckets.values():
      for i in range(0, len(samples), batch_size):
        x_batch, y_batch = zip(*samples[i:(i + batch_size)])
        # We don't create Variable objects to avoid unnecessary back props
        x_batch = self.xp.array(x_batch, dtype=self.xp.int32).transpose()
        # First dimension will be the sentence length instead of batch size
        # this is easier for looping over words in RNNs
        y_batch = self.xp.array(y_batch, dtype=self.xp.int32).transpose()
        self.batches.append((x_batch, y_batch))

  # This is similar to __next__ in the Chainer interface
  def __next__(self):
    self.iteration += 1
    if self.batch_index == 0:
      self.epoch += 1
    self.is_end_epoch = (self.batch_index == self.size - 1)
    x_batch, y_batch = self.batches[self.batch_index]
    self.batch_index = (self.batch_index + 1) % self.size
    if self.is_end_epoch and self.shuffle:
      random.shuffle(self.batches)
    return x_batch, y_batch

  @property
  def epoch_detail(self):
    return 1 +  (self.epoch - 1) + (self.batch_index * 1.0 / len(self.batches))

  def serialize(self, serializer):
    self.iteration = serializer("iterator_iteration", self.iteration)
    self.epoch = serializer("iterator_epoch", self.epoch)
