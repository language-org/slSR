import argparse
import json
import logging
import logging.config
import re
import sys

import matplotlib as mp
import numpy as np
import tensorflow as tf
import yaml
from sklearn.manifold import TSNE

mp.use("tkAgg") # enable plot when multiple threads
def parse_args(parser):
   
    """Parse pipeline's parameters
    """
    # fmt: off
    parser.add_argument(
        '--pipeline',
        dest="pipeline",
        default="None",
        help="train, predict or predict stream"
        )
    return parser.parse_known_args()[0]


def get_catalog():
    
    # get parser
    parser = argparse.ArgumentParser(add_help=False)
    args = parse_args(parser)

    # get pipeline
    with open(f"conf/{args.pipeline}/catalog.yml") as file:
        catalog = yaml.load(file)
    return catalog


def get_params():
    """ Get pipeline configuration run in terminal and 
    from conf/.. 
    """
    # get parser
    parser = argparse.ArgumentParser(add_help=False)    
    args = parse_args(parser)

    # get pipeline
    with open(f"conf/{args.pipeline}/parameters.yml") as file:
        PARAMS = yaml.load(file)

    # set pipeline parameters
    parser.add_argument(
        '-name',
        dest=PARAMS["name"]["dest"],
        default=PARAMS["name"]["current"],
        help=PARAMS["name"]["help"]
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
        type=eval(PARAMS["restore"]["type"]),
        dest=PARAMS["restore"]["dest"],
        default=PARAMS["restore"]["current"],
        help=PARAMS["restore"]["help"],
        )
    parser.add_argument(
        '--dump', 
        type=eval(PARAMS["dump"]["type"]), 
        default=PARAMS["dump"]["current"], 
        help=PARAMS["dump"]["help"], 
        )
    parser.add_argument(
        "--rm_nums", 
        type=eval(PARAMS["rm_nums"]["type"]),
        default=PARAMS["rm_nums"]["current"],
        help="rm nums"
        )
    parser.add_argument(
        "--remain_diff", 
        type=eval(PARAMS["remain_diff"]["type"]), 
        default=PARAMS["remain_diff"]["current"],
        help=PARAMS["remain_diff"]["help"]
        )

    """ Transformer """
    parser.add_argument(
        "--hidden_size", 
        type=eval(PARAMS["hidden_size"]["type"]), 
        default=PARAMS["hidden_size"]["current"], 
        help=PARAMS["hidden_size"]["help"]
        )
    parser.add_argument(
        "--filter_size", 
        type=eval(PARAMS["filter_size"]["type"]), 
        default=PARAMS["filter_size"]["current"],
        help=PARAMS["filter_size"]["help"],
        )
    parser.add_argument(
        "--num_heads", 
        type=eval(PARAMS["num_heads"]["type"]),
        default=PARAMS["num_heads"]["current"],
        help=PARAMS["num_heads"]["help"],
        )
    parser.add_argument(
        "--num_encoder_layers", 
        type=eval(PARAMS["num_encoder_layers"]["type"]),
        default=PARAMS["num_encoder_layers"]["current"], 
        help=PARAMS["num_encoder_layers"]["help"]
        )
    parser.add_argument(
        '--attention_dropout', 
        type=eval(PARAMS["attention_dropout"]["type"]),
        default=PARAMS["attention_dropout"]["current"],
        help=PARAMS["attention_dropout"]["help"],
        )
    parser.add_argument(
        '--residual_dropout', 
        type=eval(PARAMS["residual_dropout"]["type"]), 
        default=PARAMS["residual_dropout"]["current"],
        help=PARAMS["residual_dropout"]["help"],
        )
    parser.add_argument(
        '--relu_dropout', 
        dest=PARAMS["relu_dropout"]["dest"],
        type=eval(PARAMS["relu_dropout"]["type"]),
        default=PARAMS["relu_dropout"]["current"],
        help=PARAMS["relu_dropout"]["help"],
        )
    parser.add_argument(
        '--label_smoothing', 
        type=eval(PARAMS["label_smoothing"]["type"]), 
        dest=PARAMS["label_smoothing"]["dest"],
        default=PARAMS["label_smoothing"]["current"],
        help=PARAMS["label_smoothing"]["help"]
        )
    parser.add_argument(
        '--attention_key_channels', 
        dest=PARAMS["attention_key_channels"]["dest"],
        help=PARAMS["attention_key_channels"]["help"]
        )
    parser.add_argument(
        '--attention_value_channels', 
        dest=PARAMS["attention_value_channels"]["dest"],
        help=PARAMS["attention_value_channels"]["help"],
        )
    parser.add_argument(
        "--layer_preprocess", 
        type=eval(PARAMS["layer_preprocess"]["type"]), 
        default=PARAMS["layer_preprocess"]["current"],
        help=PARAMS["layer_preprocess"]["help"]
        )
    parser.add_argument(
        "--layer_postprocess", 
        type=eval(PARAMS["layer_postprocess"]["type"]), 
        default=PARAMS["layer_postprocess"]["current"], 
        help=PARAMS["layer_postprocess"]["help"]
        )
    parser.add_argument(
        "--multiply_embedding_mode", 
        type=eval(PARAMS["multiply_embedding_mode"]["type"]),
        default=PARAMS["multiply_embedding_mode"]["current"],
        help=PARAMS["multiply_embedding_mode"]["help"]
        )
    parser.add_argument(
        "--shared_embedding_and_softmax_weights", 
        type=eval(PARAMS["shared_embedding_and_softmax_weights"]["type"]),
        default=PARAMS["shared_embedding_and_softmax_weights"]["current"], 
        help=PARAMS["shared_embedding_and_softmax_weights"]["help"]
        )
    parser.add_argument(
        "--shared_source_target_embedding", 
        type=eval(PARAMS["shared_source_target_embedding"]["type"]),
        default=PARAMS["shared_source_target_embedding"]["current"], 
        help=PARAMS["shared_source_target_embedding"]["help"], 
        )
    parser.add_argument(
        "--position_info_type", 
        type=eval(PARAMS["position_info_type"]["type"]), 
        default=PARAMS["position_info_type"]["current"], 
        help=PARAMS["position_info_type"]["help"]
        )
    parser.add_argument(
        "--max_relative_dis", 
        type=eval(PARAMS["max_relative_dis"]["type"]),
        default=PARAMS["max_relative_dis"]["current"], 
        help=PARAMS["max_relative_dis"]["help"]
        )

    """Training Environment"""

    parser.add_argument(
        "--batch_size", 
        type=eval(PARAMS["batch_size"]["type"]), 
        default=PARAMS["batch_size"]["current"], 
        help=PARAMS["batch_size"]["help"]
        )
    parser.add_argument(
        "--max_epochs", 
        type=eval(PARAMS["max_epochs"]["type"]), 
        default=PARAMS["max_epochs"]["current"], 
        help=PARAMS["max_epochs"]["help"]
        )
    parser.add_argument(
        "--no_early_stop", 
        action=PARAMS["no_early_stop"]["action"],
        dest=PARAMS["no_early_stop"]["dest"],
        help=PARAMS["no_early_stop"]["help"]
        )
    parser.add_argument(
        "--patience", 
        type=eval(PARAMS["patience"]["type"]), 
        default=PARAMS["patience"]["current"], 
        help=PARAMS["patience"]["help"]
        )
    parser.add_argument(
        '--lr', 
        dest=PARAMS["lr"]["dest"], 
        default=PARAMS["lr"]["current"], 
        type=eval(PARAMS["lr"]["type"]),
        help=PARAMS["lr"]["help"]
        )
    parser.add_argument(
        '-opt', 
        dest=PARAMS["opt"]["dest"], 
        default=PARAMS["opt"]["current"],
        help=PARAMS["opt"]["help"]
        )
    parser.add_argument(
        "--alpha", 
        type=eval(PARAMS["alpha"]["type"]),
        default=PARAMS["alpha"]["current"],
        help=PARAMS["alpha"]["help"]
        )
    parser.add_argument(
        "--learning_rate_decay", 
        type=eval(PARAMS["learning_rate_decay"]["type"]),
        default=PARAMS["learning_rate_decay"]["current"],
        help=PARAMS["learning_rate_decay"]["help"]
        )
    parser.add_argument(
        "--decay_steps", 
        type=eval(PARAMS["decay_steps"]["type"]), 
        default=eval(PARAMS["decay_steps"]["current"]), 
        help=PARAMS["decay_steps"]["help"]
        )
    parser.add_argument(
        "--decay_rate", 
        type=eval(PARAMS["decay_rate"]["type"]),
        default=PARAMS["decay_rate"]["current"],
        help=PARAMS["decay_rate"]["help"]
        )

    """" Model and Vocabulary """

    parser.add_argument(
        "--dataset", 
        type=eval(PARAMS["dataset"]["type"]), 
        default=PARAMS["dataset"]["current"],
        help=PARAMS["dataset"]["help"]
            )
    parser.add_argument(
        "--model_path", 
        type=eval(PARAMS["model_path"]["type"]), 
        default=PARAMS["model_path"]["current"],
        help=PARAMS["model_path"]["help"]
        )
    parser.add_argument(
        "--vocab_path", 
        type=eval(PARAMS["vocab_path"]["type"]), 
        default=PARAMS["vocab_path"]["current"], 
        help=PARAMS["vocab_path"]["help"], 
        )

    # Data
    parser.add_argument(
        "--train_data_path", 
        type=eval(PARAMS["train_data_path"]["type"]), 
        default=PARAMS["train_data_path"]["current"], 
        help=PARAMS["train_data_path"]["help"]
        )
    parser.add_argument(
        "--test_data_path", 
        type=eval(PARAMS["test_data_path"]["type"]), 
        default=PARAMS["test_data_path"]["current"], 
        help=PARAMS["test_data_path"]["help"]
        )
    parser.add_argument(
        "--valid_data_path", 
        type=eval(PARAMS["valid_data_path"]["type"]), 
        default=PARAMS["valid_data_path"]["current"],
        help=PARAMS["valid_data_path"]["help"],
        )
    parser.add_argument(
        "--input_file", 
        type=eval(PARAMS["input_file"]["type"]), 
        default=PARAMS["input_file"]["current"], 
        help=PARAMS["input_file"]["help"]
        )
    parser.add_argument(
        "--infer_file", 
        type=eval(PARAMS["infer_file"]["type"]),  
        default=PARAMS["infer_file"]["current"],
        help=PARAMS["infer_file"]["help"]
        )
    parser.add_argument(
        '--inference_data_path', 
        dest=PARAMS["inference_data_path"]["dest"],
        default=PARAMS["inference_data_path"]["current"],
        help=PARAMS["inference_data_path"]["help"]
        )

    # parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
    # parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
    # parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

    # Others
    parser.add_argument(
        '-logdir', 
        dest=PARAMS["logdir"]["dest"],
        default=PARAMS["logdir"]["current"], 
        help=PARAMS["logdir"]["help"]
        )
    parser.add_argument(
        '-config', 
        dest=PARAMS["config"]["dest"],
        default=PARAMS["config"]["current"],
        help=PARAMS["config"]["help"],
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

def list_model_nodes():
    """Print all model node variables

    Returns:
        [type]: [description]
    """
    nodes = [n.name for n in tf.get_default_graph().as_graph_def().node] 
    print(nodes)
    return nodes

def inspect_model_node(node:str="word_embedding"):
    """Print node variable's content

    Args:
        node (str, optional): [description]. Defaults to "word_embedding".
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        variable = sess.run(f"SlotRefine/{node}:0")
        print(variable)
    return variable 

def plot_embedding_tSNE(embedding:np.ndarray):

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(embedding)

    # plot 2-D
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    mp.pyplot.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        mp.pyplot.scatter(x[i],y[i])
        # plt.annotate(labels[i],
        #              xy=(x[i], y[i]),
        #              xytext=(5, 2),
        #              textcoords='offset points',
        #              ha='right',
        #              va='bottom')
    mp.pyplot.show()
