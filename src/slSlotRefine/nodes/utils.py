import argparse
import json
import logging
import logging.config
import re
import sys

import numpy as np
import yaml


def parse_args():
   
    """Parse pipeline's parameters
    """
    # parse run hyperparameters to be used by NatSLU model
    # parse command line strings into Python objects
    parser = argparse.ArgumentParser()
    
    # fmt: off
    parser.add_argument(
        '--pipeline',
        dest="pipeline",
        default="train",
        help="train or predict"
        )
    return parser


def get_catalog():
    
    # get parser
    parser = parse_args()

    # get pipeline
    args = parser.parse_args()
    with open(f"conf/{args.pipeline}/catalog.yml") as file:
        catalog = yaml.load(file)
    return catalog


def get_params():
    
    # get parser
    parser = parse_args()

    # get pipeline
    args = parser.parse_args()
    with open(f"conf/{args.pipeline}/parameters.yml") as file:
        PARAMS = yaml.load(file)
    
    # set pipeline parameters
    parser.add_argument(
        '-name',
        dest=PARAMS["name"]["dest"],
        default=PARAMS["name"]["current"],
        help='Name of the run'
        )
    parser.add_argument(
        "--encode_mode",
        type=eval(PARAMS["encode_mode"]["type"]),
        default=PARAMS["encode_mode"]["current"],
        help=PARAMS["encode_mode"]["help"],
        )
    parser.add_argument(
        "--split",
        type=eval(PARAMS["split"]["type"]),
        default=PARAMS["split"]["current"],
        help=PARAMS["split"]["help"],
        )
    parser.add_argument(
        '--restore', 
        dest=PARAMS["restore"]["dest"],
        action=PARAMS["restore"]["action"],
        help=PARAMS["restore"]["help"],
        )
    parser.add_argument(
        '--dump', 
        type=eval(PARAMS["dump"]["type"]), 
        default=False, 
        help="is dump"
        )
    parser.add_argument(
        "--rm_nums", 
        type=bool, 
        default=False, 
        help="rm nums"
        )
    parser.add_argument(
        "--remain_diff", 
        type=bool, 
        default=True, 
        help="just remain diff"
        )

    """ Transformer """
    parser.add_argument(
        "--hidden_size", 
        type=int, 
        default=32, 
        help="hidden_size"
        )
    parser.add_argument(
        "--filter_size", 
        type=int, 
        default=32, 
        help="filter_size"
        )
    parser.add_argument(
        "--num_heads", 
        type=int, 
        default=8, 
        help="num_heads"
        )
    parser.add_argument(
        "--num_encoder_layers", 
        type=int, 
        default=2, 
        help="num_encoder_layers"
        )
    parser.add_argument(
        '--attention_dropout', 
        default=0.0, 
        type=float,
        help='att_dropout'
        )
    parser.add_argument(
        '--residual_dropout', 
        default=0.1, 
        type=float, 
        help='res_dropout'
        )
    parser.add_argument(
        '--relu_dropout', 
        dest="relu_dropout", 
        default=0.0, type=float, 
        help='relu_dropout'
        )
    parser.add_argument(
        '--label_smoothing', 
        dest="label_smoothing", 
        default=0.1, 
        type=float, 
        help='label_smoothing'
        )
    parser.add_argument(
        '--attention_key_channels', 
        dest="attention_key_channels",
        default=0, 
        type=int, 
        help='attention_key_channels'
        )
    parser.add_argument(
        '--attention_value_channels', 
        dest="attention_value_channels",
        default=0, 
        type=int, 
        help='attention_value_channels'
        )
    parser.add_argument(
        "--layer_preprocess", 
        type=str, 
        default='none', 
        help="layer_preprocess"
        )
    parser.add_argument(
        "--layer_postprocess", 
        type=str, 
        default='layer_norm', 
        help="layer_postprocess"
        )
    parser.add_argument(
        "--multiply_embedding_mode", 
        type=str, 
        default='sqrt_depth', 
        help="multiply_embedding_mode"
        )
    parser.add_argument(
        "--shared_embedding_and_softmax_weights", 
        type=bool,
        default=False, 
        help="shared_embedding_and_softmax_weights."
        )
    parser.add_argument(
        "--shared_source_target_embedding", 
        type=bool,
        default=False, 
        help="shared_source_target_embedding."
        )
    parser.add_argument(
        "--position_info_type", 
        type=str, 
        default='relative', 
        help="position_info_type"
        )
    parser.add_argument(
        "--max_relative_dis", 
        type=int, 
        default=16, 
        help="max_relative_dis"
        )

    """Training Environment"""

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=512, 
        help="Batch size."
        )
    parser.add_argument(
        "--max_epochs", 
        type=eval(PARAMS["max_epochs"]["type"]), 
        default=PARAMS["max_epochs"]["current"], 
        help="Max epochs to train."
        )
    parser.add_argument(
        "--no_early_stop", 
        action='store_false', 
        dest='early_stop',
        help="Disable early stop, which is based on sentence level accuracy."
        )
    parser.add_argument(
        "--patience", 
        type=int, 
        default=5, 
        help="Patience to wait before stop."
        )
    parser.add_argument(
        '--lr', 
        dest="lr", 
        default=0.01, 
        type=float, 
        help='Learning rate'
        )
    parser.add_argument(
        '-opt', 
        dest="opt", 
        default='adam', 
        help='Optimizer to use for training'
        )
    parser.add_argument(
        "--alpha", 
        type=float, 
        default=0.5, 
        help="balance the intent & slot filling task"
        )
    parser.add_argument(
        "--learning_rate_decay", 
        type=bool, 
        default=True, 
        help="learning_rate_decay"
        )
    parser.add_argument(
        "--decay_steps", 
        type=int, 
        default=300 * 4, 
        help="decay_steps."
        )
    parser.add_argument(
        "--decay_rate", 
        type=float, 
        default=0.9, 
        help="decay_rate."
        )

    """" Model and Vocabulary """

    parser.add_argument(
        "--dataset", 
        type=eval(PARAMS["dataset"]["type"]), 
        default=PARAMS["dataset"]["current"],
        help="""
            Type 'atis' or 'snips' to use dataset provided by us or enter 
            what ever you named your own dataset. Note, if you don't want 
            to use this part, enter --dataset=''. It can not be None
            """
            )
    parser.add_argument(
        "--model_path", 
        type=eval(PARAMS["model_path"]["type"]), 
        default='./model', 
        help="Path to save model."
        )
    parser.add_argument(
        "--vocab_path", 
        type=eval(PARAMS["vocab_path"]["type"]), 
        default='./vocab', 
        help="Path to vocabulary files."
        )

    # Data
    parser.add_argument(
        "--train_data_path", 
        type=str, 
        default='train', 
        help="Path to training data files."
        )
    parser.add_argument(
        "--test_data_path", 
        type=str, 
        default='test', 
        help="Path to testing data files."
        )
    parser.add_argument(
        "--valid_data_path", 
        type=str, 
        default='test', 
        help="Path to validation data files."
        )
    parser.add_argument(
        "--input_file", 
        type=str, 
        default='data', 
        help="Input file name."
        )
    parser.add_argument(
        "--infer_file", 
        type=str, 
        default='infer', 
        help="Infer file name"
        )
    parser.add_argument(
        '--inference_data_path', 
        dest="inference_data_path", 
        default='inference', 
        help="Path to run inference on real data files."
        )

    # parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
    # parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
    # parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

    # Others
    parser.add_argument(
        '-logdir', 
        dest="log_dir", 
        default='./log/', 
        help='Log directory'
        )
    parser.add_argument(
        '-config', 
        dest="config_dir", 
        default='./config/', 
        help='Config directory'
        )
    return parser.parse_args()


def createVocabulary(input_path, output_path, no_pad=False):
    if not isinstance(input_path, str):
        raise TypeError('input_path should be string')

    if not isinstance(output_path, str):
        raise TypeError('output_path should be string')

    vocab = {}
    with open(input_path, 'r') as fd, \
            open(output_path, 'w+') as out:
        for line in fd:
            line = line.rstrip('\r\n')
            words = line.split()

            for w in words:
                if w == '_UNK':
                    break
                if str.isdigit(w) == True:
                    w = '0'
                if w in vocab:
                    vocab[w] += 1
                else:
                    vocab[w] = 1
        if no_pad == False:
            vocab = ['_PAD', '_UNK'] + sorted(vocab, key=vocab.get, reverse=True)
        else:
            vocab = ['_UNK'] + sorted(vocab, key=vocab.get, reverse=True)

        for v in vocab:
            out.write(v + '\n')


def loadVocabulary(path):
    if not isinstance(path, str):
        raise TypeError('path should be a string')

    vocab = []
    rev = []
    with open(path) as fd:
        for line in fd:
            line = line.rstrip('\r\n')
            rev.append(line)
        vocab = dict([(x, y) for (y, x) in enumerate(rev)])

    return {'vocab': vocab, 'rev': rev}


def sentenceToIds(data, vocab):
    if not isinstance(vocab, dict):
        raise TypeError('vocab should be a dict that contains vocab and rev')
    vocab = vocab['vocab']
    if isinstance(data, str):
        words = data.split()
    elif isinstance(data, list):
        words = data
    else:
        raise TypeError('data should be a string or a list contains words')

    ids = []
    for w in words:
        if str.isdigit(w) == True:
            w = '0'
        ids.append(vocab.get(w, vocab['_UNK']))

    return ids


def padSentence(s, max_length, vocab):
    return s + [vocab['vocab']['_PAD']] * (max_length - len(s))


# compute f1 score is modified from conlleval.pl
def __startOfChunk(prevTag, tag, prevTagType, tagType, chunkStart=False):
    if prevTag == 'B' and tag == 'B':
        chunkStart = True
    if prevTag == 'I' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'B':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if prevTag == 'E' and tag == 'E':
        chunkStart = True
    if prevTag == 'E' and tag == 'I':
        chunkStart = True
    if prevTag == 'O' and tag == 'E':
        chunkStart = True
    if prevTag == 'O' and tag == 'I':
        chunkStart = True

    if tag != 'O' and tag != '.' and prevTagType != tagType:
        chunkStart = True
    return chunkStart


def __endOfChunk(prevTag, tag, prevTagType, tagType, chunkEnd=False):
    if prevTag == 'B' and tag == 'B':
        chunkEnd = True
    if prevTag == 'B' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'B':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag == 'E' and tag == 'E':
        chunkEnd = True
    if prevTag == 'E' and tag == 'I':
        chunkEnd = True
    if prevTag == 'E' and tag == 'O':
        chunkEnd = True
    if prevTag == 'I' and tag == 'O':
        chunkEnd = True

    if prevTag != 'O' and prevTag != '.' and prevTagType != tagType:
        chunkEnd = True
    return chunkEnd


def __splitTagType(tag):
    s = tag.split('-')
    if len(s) > 2 or len(s) == 0:
        raise ValueError('tag format wrong. it must be B-xxx.xxx')
    if len(s) == 1:
        tag = s[0]
        tagType = ""
    else:
        tag = s[0]
        tagType = s[1]
    return tag, tagType


def computeF1Score(correct_slots, pred_slots):
    correctChunk = {}
    correctChunkCnt = 0
    foundCorrect = {}
    foundCorrectCnt = 0
    foundPred = {}
    foundPredCnt = 0
    correctTags = 0
    tokenCount = 0
    for correct_slot, pred_slot in zip(correct_slots, pred_slots):
        inCorrect = False
        lastCorrectTag = 'O'
        lastCorrectType = ''
        lastPredTag = 'O'
        lastPredType = ''
        for c, p in zip(correct_slot, pred_slot):
            correctTag, correctType = __splitTagType(c)
            predTag, predType = __splitTagType(p)

            if inCorrect == True:
                if __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                        (lastCorrectType == lastPredType):
                    inCorrect = False
                    correctChunkCnt += 1
                    if lastCorrectType in correctChunk:
                        correctChunk[lastCorrectType] += 1
                    else:
                        correctChunk[lastCorrectType] = 1
                elif __endOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) != \
                        __endOfChunk(lastPredTag, predTag, lastPredType, predType) or \
                        (correctType != predType):
                    inCorrect = False

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True and \
                    __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True and \
                    (correctType == predType):
                inCorrect = True

            if __startOfChunk(lastCorrectTag, correctTag, lastCorrectType, correctType) == True:
                foundCorrectCnt += 1
                if correctType in foundCorrect:
                    foundCorrect[correctType] += 1
                else:
                    foundCorrect[correctType] = 1

            if __startOfChunk(lastPredTag, predTag, lastPredType, predType) == True:
                foundPredCnt += 1
                if predType in foundPred:
                    foundPred[predType] += 1
                else:
                    foundPred[predType] = 1

            if correctTag == predTag and correctType == predType:
                correctTags += 1

            tokenCount += 1

            lastCorrectTag = correctTag
            lastCorrectType = correctType
            lastPredTag = predTag
            lastPredType = predType

        if inCorrect == True:
            correctChunkCnt += 1
            if lastCorrectType in correctChunk:
                correctChunk[lastCorrectType] += 1
            else:
                correctChunk[lastCorrectType] = 1

    if foundPredCnt > 0:
        precision = 100 * correctChunkCnt / foundPredCnt
    else:
        precision = 0

    if foundCorrectCnt > 0:
        recall = 100 * correctChunkCnt / foundCorrectCnt
    else:
        recall = 0

    if (precision + recall) > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0

    return f1, precision, recall


def get_logger(name, log_dir, config_dir):
    """
    Creates a logger object

    Parameters
    ----------
    name:           Name of the logger file
    log_dir:        Directory where logger file needs to be stored
    config_dir:     Directory from where log_config.json needs to be read

    Returns
    -------
    A logger object which writes to both file and stdout

    """
    config_dict = json.load(open(config_dir + 'log_config.json'))
    config_dict['handlers']['file_handler']['filename'] = log_dir + name.replace('/', '-')
    logging.config.dictConfig(config_dict)
    logger = logging.getLogger(name)

    std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logging.Formatter(std_out_format))
    logger.addHandler(consoleHandler)

    return logger


def remove_digital_sentence_processer(line):
    def istime(input):
        regexp = re.compile("(24:00|24:00:00|2[0-3]:[0-5][0-9]|2[0-3]:[0-5][0-9]:[0-5][0-9]|"
                            "[0-1][0-9]:[0-5][0-9]|[0-1][0-9]:[0-5][0-9]:[0-5][0-9]|"
                            "([0-9][0-9]|[0-9])/([0-9][0-9]|[0-9])/([0-9][0-9][0-9][0-9]|[0-9][0-9]))")
        return bool(regexp.match(input))

    line_array = line.strip().split()
    line_array = ['0' if str.isdigit(w) else w for w in line_array]
    line_array = ['$time$' if istime(w) else w for w in line_array]

    return ' '.join(line_array)


class DataProcessor(object):
    def __init__(self, in_path, slot_path, intent_path, in_vocab, slot_vocab, intent_vocab):
        self.__fd_in = open(in_path, 'r')
        self.__fd_slot = open(slot_path, 'r')
        self.__fd_intent = open(intent_path, 'r')
        self.__in_vocab = in_vocab
        self.__slot_vocab = slot_vocab
        self.__intent_vocab = intent_vocab
        self.end = 0

    def close(self):
        self.__fd_in.close()
        self.__fd_slot.close()
        self.__fd_intent.close()

    def get_batch(self, batch_size):
        in_data = []
        slot_data = []
        slot_weight = []
        length = []
        intents = []

        batch_in = []
        batch_slot = []
        max_len = 0

        # used to record word(not id)
        in_seq = []
        slot_seq = []
        intent_seq = []
        for i in range(batch_size):
            inp = self.__fd_in.readline()
            if inp == '':
                self.end = 1
                break
            slot = self.__fd_slot.readline()
            intent = self.__fd_intent.readline()
            inp = inp.rstrip()
            slot = slot.rstrip()
            intent = intent.rstrip()

            in_seq.append(inp)
            slot_seq.append(slot)
            intent_seq.append(intent)

            iii = inp
            sss = slot
            inp = sentenceToIds(inp, self.__in_vocab)
            slot = sentenceToIds(slot, self.__slot_vocab)
            intent = sentenceToIds(intent, self.__intent_vocab)
            batch_in.append(np.array(inp))
            batch_slot.append(np.array(slot))
            length.append(len(inp))
            intents.append(intent[0])
            if len(inp) != len(slot):
                print(iii, sss)
                print(inp, slot)
                exit(0)
            if len(inp) > max_len:
                max_len = len(inp)

        length = np.array(length)
        intents = np.array(intents)
        # print(max_len)
        # print('A'*20)
        for i, s in zip(batch_in, batch_slot):
            in_data.append(padSentence(list(i), max_len, self.__in_vocab))
            slot_data.append(padSentence(list(s), max_len, self.__slot_vocab))
            # print(s)
        in_data = np.array(in_data)
        slot_data = np.array(slot_data)
        # print(in_data)
        # print(slot_data)
        # print(type(slot_data))
        for s in slot_data:
            weight = np.not_equal(s, np.zeros(s.shape))
            weight = weight.astype(np.float32)
            slot_weight.append(weight)
        slot_weight = np.array(slot_weight)
        return in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq


def get_uncoordinated_chunking_nums(path):
    uncoordinated_num = 0

    def process_batch(batch):
        pred_slot_array = batch[1].strip().split('\t')[2].split(' ')
        pred_chunk_array = []

        # get chunks
        st = 0
        for i in range(len(pred_slot_array) - 1):
            if pred_slot_array[i][0] == 'b':
                st = i
            if pred_slot_array[i][0] == 'o' and pred_slot_array[i + 1][0] == 'i':  # o o o o i i o o
                st = i + 1
            if pred_slot_array[i][0] != 'o' and pred_slot_array[i + 1][0] in ('o', 'b'):
                end = i + 1
                pred_chunk_array.append((st, end))

        if pred_slot_array[-1][0] == 'i':
            pred_chunk_array.append((st, len(pred_slot_array)))
        elif pred_slot_array[-1][0] == 'b':
            pred_chunk_array.append((len(pred_slot_array) - 1, len(pred_slot_array)))

        # check all the chunks
        num = 0
        for st, end in pred_chunk_array:
            if pred_slot_array[st][0] == 'i':
                num += 1
                continue
            else:
                flag = True
                chunk_label = ""
                for i in range(st, end):
                    if i == st:
                        chunk_label = pred_slot_array[i].strip().split('-')[1]  # for example: b-fromloc.city_name
                    elif pred_slot_array[i].strip().split('-')[1] != chunk_label:
                        flag = False
                        break
                if chunk_label and not flag:
                    num += 1
        return num

    num = 1
    batch = []
    for line in open(path):
        batch.append(line)
        if num % 2 == 0:
            uncoordinated_num += process_batch(batch)
            batch = []
        num += 1
    return uncoordinated_num
