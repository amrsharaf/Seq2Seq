import numpy as np
import cupy as cp

def read_lines(dir):
  with open(dir, 'r') as reader:
    lines = [np.array(line.strip().split(), dtype=np.int32) for line in reader]
#     lines = [line.strip().split() for line in reader]
    return lines
  return None

def read_data(src_dir, trg_dir):
  src_lines = read_lines(src_dir)
  trg_lines = read_lines(trg_dir)
  assert(len(src_lines) == len(trg_lines))
  data = zip(src_lines, trg_lines)
  return data

def get_ordering_dataset():
  train_data = read_data('word_ordering/train_src_ids.txt', 'word_ordering/train_trg_ids.txt')
  valid_data = read_data('word_ordering/valid_src_ids.txt', 'word_ordering/valid_trg_ids.txt')
  test_data  = read_data('word_ordering/test_src_ids.txt',  'word_ordering/test_trg_ids.txt')
  return train_data, valid_data, test_data
  
def main():
  get_ordering_dataset()
    
if __name__ == '__main__':
  main()