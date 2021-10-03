
import os

import src.slSlotRefine.nodes.utils as local_utils
from tensorflow.keras.preprocessing.text import Tokenizer


def create_tokenizer(data):
    """Instantiate tokenizers that vectorize the text corpus, by turning
    the text into either a sequence token indices (position in token 
    dictionary) or into a vector where the coefficient for each token 
    could be binary, based on word count, based on tf-idf token are splitted 
    by spaces ' ' andthe flak '<unk>' was added to word index and used to replace
    out-of-vocabulary (oov) words input tokenize instantiate tokenizers for 
    utterance, their IOB tags and the intent labels
    """
    data.seq_in_tokenizer = Tokenizer(
        filters='', oov_token='<unk>', split=data.arg.split
        )
    data.seq_out_tokenizer = Tokenizer(
        filters='', split=data.arg.split, oov_token='<unk>'
        )
    data.label_tokenizer = Tokenizer(
        filters='', split=data.arg.split, oov_token='<unk>'
        )

    # open training data file
    input_file_path = os.path.join(data.full_train_path, data.arg.input_file)
    with open(input_file_path, encoding=data.arg.encode_mode) as fin:
        
        # store each line (ending with \n) in a list of list of 
        # [utterance, IOB tags, intent label]
        lines = fin.readlines()
        lines = [line.strip().lower().split('\t') for line in lines]
        try:
            seq_in, seq_out, intent = zip(*lines)
            # remove numbers
            if data.arg.rm_nums:
                seq_in = [local_utils.remove_digital_sentence_processer(line) for line in seq_in]
        except Exception:
            print(lines)
            raise FileNotFoundError('The input training data file is invalid!')

        # Updates internal {index: word} and {index: doc} vocabularies 
        # based on the list of utterances, their IOB tags and their intent label
        data.seq_in_tokenizer.fit_on_texts(seq_in)
        data.seq_out_tokenizer.fit_on_texts(seq_out)
        data.label_tokenizer.fit_on_texts(intent)

    print("size of seq_in_tokenizer is {}".format(len(data.seq_in_tokenizer.word_index)))
    print("size of seq_out_tokenizer is {}".format(len(data.seq_out_tokenizer.word_index)))
    print("size of label_tokenizer is {}".format(len(data.label_tokenizer.word_index)))
    return data

