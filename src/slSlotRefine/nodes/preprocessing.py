import os

import src.slSlotRefine.nodes.utils as local_utils
from tensorflow.keras.preprocessing.text import Tokenizer


def create_train_utterance_tokenizer(data):
    """Instantiate tokenizers that vectorize the text corpus, by turning
    the text into either a sequence token indices (position in token 
    dictionary) or into a vector where the coefficient for each token 
    could be binary, based on word count, based on tf-idf token are splitted 
    by spaces ' ' and the flak '<unk>' was added to word index and used to replace
    out-of-vocabulary (oov) words input tokenize instantiate tokenizers for 
    utterance, their IOB tags and the intent labels
    """
    data.seq_in_tokenizer = Tokenizer(
        filters="", oov_token="<unk>", split=data.arg.split
    )

    # open training data file
    input_file_path = os.path.join(data.full_train_path, data.arg.input_file)
    with open(input_file_path, encoding=data.arg.encode_mode) as fin:

        # store each line (ending with \n) in a list of list of
        # [utterance, IOB tags, intent label]
        lines = fin.readlines()
        lines = [line.strip().lower().split("\t") for line in lines]
        try:
            seq_in, _, _ = zip(*lines)
        except Exception:
            print(lines)
            raise FileNotFoundError("The input training data file is invalid!")

        # Updates internal vocabularies {index: word} and {index: doc}
        # based on the list of utterances in the "input_file_path" dataset,
        # its IOB tags and its intent labels
        data.seq_in_tokenizer.fit_on_texts(seq_in)
    return data


def create_train_slots_tokenizer(data):
    """Instantiate tokenizers that vectorize the text corpus, by turning
    the text into either a sequence token indices (position in token 
    dictionary) or into a vector where the coefficient for each token 
    could be binary, based on word count, based on tf-idf token are splitted 
    by spaces ' ' and the flak '<unk>' was added to word index and used to replace
    out-of-vocabulary (oov) words input tokenize instantiate tokenizers for 
    utterance, their IOB tags and the intent labels
    """
    data.seq_out_tokenizer = Tokenizer(
        filters="", split=data.arg.split, oov_token="<unk>"
    )

    # open training data file
    input_file_path = os.path.join(data.full_train_path, data.arg.input_file)
    with open(input_file_path, encoding=data.arg.encode_mode) as fin:

        # store each line (ending with \n) in a list of list of
        # [utterance, IOB tags, intent label]
        lines = fin.readlines()
        lines = [line.strip().lower().split("\t") for line in lines]
        try:
            _, seq_out, _ = zip(*lines)
        except Exception:
            print(lines)
            raise FileNotFoundError("The input training data file is invalid!")

        # Updates internal vocabularies {index: word} and {index: doc}
        # based on the list of utterances in the "input_file_path" dataset,
        # its IOB tags and its intent labels
        data.seq_out_tokenizer.fit_on_texts(seq_out)
    return data


def create_train_label_tokenizer(data):
    """Instantiate tokenizers that vectorize the text corpus, by turning
    the text into either a sequence token indices (position in token 
    dictionary) or into a vector where the coefficient for each token 
    could be binary, based on word count, based on tf-idf token are splitted 
    by spaces ' ' and the flak '<unk>' was added to word index and used to replace
    out-of-vocabulary (oov) words input tokenize instantiate tokenizers for 
    utterance, their IOB tags and the intent labels
    """
    data.label_tokenizer = Tokenizer(
        filters="", split=data.arg.split, oov_token="<unk>"
    )

    # open training data file
    input_file_path = os.path.join(data.full_train_path, data.arg.input_file)
    with open(input_file_path, encoding=data.arg.encode_mode) as fin:

        # store each line (ending with \n) in a list of list of
        lines = fin.readlines()
        lines = [line.strip().lower().split("\t") for line in lines]
        try:
            _, _, intent = zip(*lines)
        except Exception:
            print(lines)
            raise FileNotFoundError("The input training data file is invalid!")

        # Updates internal {index: word} and {index: doc} vocabularies
        # based on the list of utterances, their IOB tags and their intent label
        data.label_tokenizer.fit_on_texts(intent)
    return data


def create_predict_utterance_tokenizer(data):
    """Instantiate tokenizers that vectorize the text corpus, by turning
    the text into either a sequence token indices (position in token 
    dictionary) or into a vector where the coefficient for each token 
    could be binary, based on word count, based on tf-idf token are splitted 
    by spaces ' ' and the flak '<unk>' was added to word index and used to replace
    out-of-vocabulary (oov) words input tokenize instantiate tokenizers for 
    utterance, their IOB tags and the intent labels
    """
    data.seq_in_tokenizer = Tokenizer(
        filters="", oov_token="<unk>", split=data.arg.split
    )

    # open training data file
    infer_file = os.path.join(data.full_infer_path, data.arg.input_file)
    with open(infer_file, encoding=data.arg.encode_mode) as fin:

        # store each line in a list of utterances
        seq_in = fin.readlines()
        seq_in = [line.strip().lower().split("\t") for line in seq_in][0]
        # Updates internal vocabularies {index: word} and {index: doc}
        # based on the list of utterances in the "input_file_path" dataset,
        # its IOB tags and its intent labels
        # to visualize vectorized data run "data.seq_in_tokenizer.texts_to_sequences(lines)"
        data.seq_in_tokenizer.fit_on_texts(seq_in)
    return data
