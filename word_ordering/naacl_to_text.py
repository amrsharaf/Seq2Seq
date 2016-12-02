from argparse import ArgumentParser
from numpy.random import RandomState

def parse_args():
    ap = ArgumentParser()
    ap.add_argument('--input',  '-i', type=str, required=True)
    ap.add_argument('--output', '-o', type=str, required=True)
    ap.add_argument('--task',   '-t', type=str, required=True)
    args = ap.parse_args()
    return args

def get_sentences(lines):
    sentence = []
    for line in lines:
        if line == '\n':
            sentence.append('\n')
            yield ' '.join(sentence)
            sentence = []
        else:
            sentence.append(line.strip().split()[1])

def convert_to_txt(input, output):
    print 'Input file: ',  input
    print 'Output file: ', output
    with open(output, 'w') as writer:
        with open(input, 'r') as reader:
            lines = (line for line in reader)
            sentences = get_sentences(lines)
            writer.writelines(sentences)

def shuffle_input(input, output):
    print 'Input file: ', input
    print 'Output file: ', output
    prng = RandomState(0)
    with open(output, 'w') as writer:
        with open(input, 'r') as reader:
            lines = (line for line in reader)
            word_lists = map(lambda x: x.strip().split(), lines)
            map(prng.shuffle, word_lists)
            shuffled_lines = map(lambda x: (' '.join(x) + '\n'), word_lists)
            writer.writelines(shuffled_lines)
    print 'prng :', prng
    print 'done'

def main():
    args = parse_args()
    if args.task == 'txt':
        convert_to_txt(args.input, args.output)
    elif args.task =='shuffle':
        shuffle_input(args.input, args.output)
    else:
        print 'unknown task!'

if __name__ == '__main__':
    main()
