import logging
import os
import random
from typing import List

import numpy as np
import src.slSlotRefine.nodes.thumt.layers as layers
import src.slSlotRefine.nodes.utils as local_utils
import tensorflow as tf
from src.slSlotRefine.nodes import etl
from src.slSlotRefine.nodes import preprocessing as prep
from src.slSlotRefine.nodes.inference import write_predictions
from src.slSlotRefine.nodes.thumt.models.transformer import (
    _ffn_layer, transformer_encoder)
from src.slSlotRefine.nodes.utils import (get_catalog, get_logger, get_params,
                                          get_uncoordinated_chunking_nums)
from tensorflow.contrib.layers import xavier_initializer

# set seed for reproducibility
tf.random.set_random_seed(0)


class Model(object):
    """SlotRefine model class

    Args:
        Model ([type]): Abstract parent model class
    """

    def __init__(self, args, catalog:dict):
        """Instantiation

        Args:
            args (NameSpace): 
                parsed pipeline parameter arguments   
            catalog (dict): 
                dictionary of dataset paths listed in conf/catalog.yml  
        """

        # print parameters & dataset
        self.arg = args
        self.catalog = catalog

        # add logger
        self.logger = get_logger(
            self.arg.name, 
            self.arg.log_dir, 
            self.arg.config_dir
            )

        # initialize data paths
        self = etl.init_data_paths(self)

        # Instantiate tokenizers to index training corpus tokens
        # They creates three separate vocabularies {index: token}
        # based on the list of utterances, slots (IOB tags) and
        # labels stored in the dataset stored in "input_file_path"
        # Index the tokens of the utterances to predict 
        self = prep.create_train_utterance_tokenizer(self)
        self = prep.create_train_slots_tokenizer(self)
        self = prep.create_train_label_tokenizer(self)                
        self = prep.create_predict_utterance_tokenizer(self)

        # get the index of the O-tag slots
        self._get_0_tag_index()

        # global step
        self.global_step = tf.Variable(
            0, 
            trainable=False, 
            name="global_step"
            )

        # initialize model graphs
        self.create_train_graphs()
        self.create_inference_graph()


    def _get_0_tag_index(self):

        """ The initial tag embedding ???O??? added to each 
        inputting position is designed for the two-pass mechanism
        e.g., 0 (B-singer I-song 0) (B-song 0 0)

        Returns:
            self: Instantiated model
        """
        self.o_idx = 0
        for word, idx in self.seq_out_tokenizer.word_index.items():
            if word == "o":
                self.o_idx = idx
                logging.info("o_idx is: ".format(self.o_idx))
                break
        return self


    def create_train_graphs(self):
        """Initialize the model's training evaluation 
        and testing graphs        
        """
        self.create_placeholder()
        self.create_train_graph()
        self.create_eval_graph()
        self.create_test_graph()


    def add_optimizer(self, loss, global_step, isAdam=True):
        """
        Add the training optimizer

        Args:
            loss: computed loss
            global_step (int32 or int64): 	
                A scalar int32 or int64 Tensor or a Python number. Global 
                step to use for the decay computation. Must not be negative.
            isAdam: Use or not the Adam optimizer, default True

        Returns:        
            train_op: training optimizer
        """
        learning_rate = tf.train.exponential_decay(
            self.arg.lr,
            global_step,
            self.arg.decay_steps,
            self.arg.decay_rate,
            staircase=False,
        )

        with tf.name_scope("Optimizer"):
            
            if isAdam and self.arg.learning_rate_decay:
                # case Adam optimizer and learning rate decay
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif isAdam:
                # case Adam optimizer only
                optimizer = tf.train.AdamOptimizer(self.arg.lr)
            else:
                # case simple gradient descent optimizer
                optimizer = tf.train.GradientDescentOptimizer(self.arg.lr)
            
            # calculate gradients
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)

            # Prevent exploding gradients by clipping the gradients by 
            # the ratio of the sum of their norms: 
            #       gradients[i] * clip_norm / max(global_norm, clip_norm)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(
                zip(clipped_gradients, params), 
                global_step=global_step
            )

        return train_op


    def batch_process(self, lines):
        """Preprocessing of batch of utterance via tokenization, padding,...

        Args:
            lines ([type]): [description]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """

        # example: ["Add Eminem to playlist\t0 B-Artist 0 B-playlist\tAddtoPlaylist", 
        # "Book 2 people tonight\t0 0 0 B-time\tBooking"]
        lines = [line.strip().lower().split("\t") for line in lines]

        try:
            # curate multi-space with one
            seq_in, seq_out, label = zip(*lines)
            seq_in = [" ".join(line.split()) for line in seq_in]
        except:
            print(lines)
            logging.error("(batch_process) Input data is unvalid!")

        # drop digits
        if self.arg.rm_nums:
            seq_in = [
                local_utils.remove_digital_sentence_processer(line) for line in seq_in
            ]

        # tokenize utterances, their IOB tags and their label
        # Transforms each text in texts to a sequence of integers.
        seq_in_ids = self.seq_in_tokenizer.texts_to_sequences(seq_in)
        seq_out_ids = self.seq_out_tokenizer.texts_to_sequences(seq_out)
        label_ids = self.label_tokenizer.texts_to_sequences(label)
        label_ids = np.array(label_ids).astype(np.int32)
        label_ids = label_ids.squeeze()

        # add padding to utterances
        # converts to array
        seq_in_ids = tf.keras.preprocessing.sequence.pad_sequences(
            seq_in_ids, padding="post", truncating="post"
        )
        temp = seq_in_ids > 0
        sequence_length = temp.sum(-1)
        sequence_length = sequence_length.astype(np.int32)

        # add padding to IOB tagss
        seq_out_ids = tf.keras.preprocessing.sequence.pad_sequences(
            seq_out_ids, padding="post", truncating="post"
        )
        seq_out_weights = seq_out_ids > 0
        seq_out_weights = seq_out_weights.astype(np.float32)
        return seq_in_ids, sequence_length, seq_out_ids, seq_out_weights, label_ids


    def batch_process_to_infer(self, lines:List[str]):
        """Preprocessing of batch of utterance via tokenization, padding,...

        Args:
            lines (List[str]): a list of utterance strings:
                example:  
                [
                    "Add Eminem to playlist\t0 B-Artist 0 B-playlist\tAddtoPlaylist", 
                    "Book 2 people tonight\t0 0 0 B-time\tBooking"
                    ]

        Raises:
            ValueError: [description]

        Returns:
            [type]: [description]
        """
        # convert the list of strings to a tuple of strings
        seq_in = tuple(lines)

        # tokenize utterances, their IOB tags and their label
        # Transforms each text in texts to a sequence of integers.
        seq_in_ids = self.seq_in_tokenizer.texts_to_sequences(seq_in)

        # add padding to utterances
        # converts to array
        seq_in_ids = tf.keras.preprocessing.sequence.pad_sequences(
            seq_in_ids, padding="post", truncating="post"
        )
        temp = seq_in_ids > 0
        seq_len = temp.sum(-1)
        seq_len = seq_len.astype(np.int32)
        return seq_in_ids, seq_len


    def get_batch(self, path, batch_size, is_train=False):
        dataset = tf.data.TextLineDataset([path])
        if is_train:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        iter = dataset.make_initializable_iterator()
        batch = iter.get_next()  # Tensor("IteratorGetNext:0", shape=(?,), dtype=string)
        input_data, sequence_length, slots, slot_weights, intent = tf.py_func(
            self.batch_process,
            [batch],
            [tf.int32, tf.int32, tf.int32, tf.float32, tf.int32],
        )

        return input_data, sequence_length, slots, slot_weights, intent, iter


    def get_batch_np_iter(self, path):
        data = []
        with open(path, "r") as fin:
            for line in fin:
                line = line.strip()
                data.append(line)

        # shuffle training data
        if self.arg.pipeline == "train":
            random.shuffle(data)
        for line in data:
            yield line


    def get_batch_np(self, iterator, path, batch_size, is_train=False):
        cnt = 0
        batch = []
        is_last_batch = False
        while True:
            try:
                line = next(iterator)
                batch.append(line)
                cnt += 1

                if batch_size == cnt:
                    break
            except StopIteration:
                iterator = self.get_batch_np_iter(path)
                is_last_batch = True
        return batch, iterator, is_last_batch


    def get_start_tags(self, slot_outputs):
        # pred_slot = slot_outputs.reshape((slot_outputs.shape[0], slot_outputs.shape[1], -1))
        pred_slot = slot_outputs[:, :, 2:].argmax(-1) + 2
        start_ids = []
        for word, idx in self.seq_out_tokenizer.word_index.items():
            if word.startswith("b-", 0, len(word)):
                start_ids.append(idx)
        start_tags = [[i if i in start_ids else 0 for i in line] for line in pred_slot]
        return start_tags

    def create_model(
        self,
        input_data,
        input_tags,
        input_size:int,
        sequence_length:int,
        slot_size:int,
        intent_size:int,
        hidden_size=128,
        is_training:bool=True,
        model_name:str="SlotRefine",
    ):
        """Create the model

        Args:
            input_data ([type]): [description]
            input_tags ([type]): [description]
            input_size (int): [description]
            sequence_length (int): [description]
            slot_size (int): [description]
            intent_size (int): [description]
            hidden_size (int, optional): [description]. Defaults to 128.
            is_training (bool, optional): [description]. Defaults to True.
            model_name (str, optional): [description]. Defaults to "SlotRefine".

        Returns:
            [List[np.ndarray]]: 
                [slot_outputs, intent_outputs]
        """
    
        with tf.variable_scope(name_or_scope=model_name, reuse=tf.AUTO_REUSE):

            dtype = tf.get_variable_scope().dtype

            # create a word embedding
            # [Size of utterance vocabulary, word dimensionality]
            # use xavier initializer to keep the scale of the gradients
            # roughly the same in all layers
            word_embedding = tf.get_variable(
                "word_embedding",
                [input_size, hidden_size],
                initializer=xavier_initializer(),
            )

            # get the current batch's word embedding
            # [batch, len_q, hidden_size]
            inputs_emb = tf.nn.embedding_lookup(word_embedding, input_data)

            # create a tag embedding 
            # [Size of slot vocabulary, word dimensionality]
            tag_embedding = tf.get_variable(
                "tag_embedding",
                [slot_size, hidden_size],
                initializer=xavier_initializer(),
            )
            # get the current batch's tag embedding
            # [batch, len_q, hidden_size]
            tags_emb = tf.nn.embedding_lookup(
                tag_embedding, input_tags
            )  
            
            # combine word and tag embeddings
            inputs = inputs_emb + tags_emb

            # insert CLS as the first token
            cls = tf.get_variable(
                "cls", 
                [hidden_size], 
                trainable=True, 
                initializer=xavier_initializer()
            )
            cls = tf.reshape(cls, [1, 1, -1])
            cls = tf.tile(cls, [tf.shape(inputs)[0], 1, 1])
            inputs = tf.concat([cls, inputs], 1)

            # [batch, len_q]
            src_mask = tf.sequence_mask(
                sequence_length + 1,
                maxlen=tf.shape(inputs)[1],
                dtype=dtype or tf.float32,
            )  
            src_mask.set_shape((None, None))

            if self.arg.multiply_embedding_mode == "sqrt_depth":
                inputs = inputs * (hidden_size ** -0.5)

            inputs = inputs * tf.expand_dims(src_mask, -1)
            bias = tf.get_variable("bias", [hidden_size])
            encoder_input = tf.nn.bias_add(inputs, bias)
            enc_attn_bias = layers.attention.attention_bias(
                src_mask, "masking", dtype=dtype
            )

            if self.arg.residual_dropout:
                if is_training:
                    keep_prob = 1.0 - self.arg.residual_dropout
                else:
                    keep_prob = 1.0
                encoder_input = tf.nn.dropout(encoder_input, keep_prob)

            # Feed into Transformer
            att_dropout = self.arg.attention_dropout
            res_dropout = self.arg.residual_dropout

            # case predict
            if not is_training:
                self.arg.attention_dropout = 0.0
                self.arg.residual_dropout = 0.0
            outputs = transformer_encoder(
                encoder_input, enc_attn_bias, self.arg
            )  # [batch, len_q + 1, out_size]
            self.arg.attention_dropout = att_dropout
            self.arg.residual_dropout = res_dropout

            intent_output, slot_output = tf.split(
                outputs, [1, tf.shape(outputs)[1] - 1], 1
            )

            with tf.variable_scope("intent_proj"):
                intent_state = intent_output
                intent_output = _ffn_layer(
                    intent_output, self.arg.hidden_size, intent_size, scope="intent"
                )

                # mask first token of intent_output forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(intent_output)) * -1e10
                mask_true = tf.ones(tf.shape(intent_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(intent_output), dtype=bool)
                intent_output_mask = tf.concat(
                    [mask_true[:, :, :2], mask_false[:, :, 2:]], -1
                )
                intent_output = tf.where(intent_output_mask, mask_values, intent_output)

            with tf.variable_scope("slot_proj"):

                slot_output = tf.concat(
                    [
                        slot_output,
                        tf.tile(intent_state, [1, tf.shape(slot_output)[1], 1]),
                    ],
                    2,
                )
                # slot_output = linear(slot_output, slot_size, True, True, scope="slot")  # [?, ?, slot_size]
                slot_output = _ffn_layer(
                    slot_output, self.arg.hidden_size, slot_size, scope="slot"
                )

                # mask first two tokens (_PAD_, _UNK_) of slot_outputs forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(slot_output)) * -1e10
                mask_true = tf.ones(tf.shape(slot_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(slot_output), dtype=bool)
                slot_outputs_mask = tf.concat(
                    [mask_true[:, :, :2], mask_false[:, :, 2:]], -1
                )
                slot_output = tf.where(slot_outputs_mask, mask_values, slot_output)

            outputs = [slot_output, intent_output]
        return outputs

    def create_model_predict(
        self,
        input_data,
        input_tags,
        input_size:int,
        sequence_length:int,
        slot_size:int,
        intent_size:int,
        hidden_size=128,
        is_training:bool=True,
        model_name:str="SlotRefine",
    ):
        """Create the model

        Args:
            input_data ([type]): [description]
            input_tags ([type]): [description]
            input_size (int): [description]
            sequence_length (int): [description]
            slot_size (int): [description]
            intent_size (int): [description]
            hidden_size (int, optional): [description]. Defaults to 128.
            is_training (bool, optional): [description]. Defaults to True.
            model_name (str, optional): [description]. Defaults to "SlotRefine".

        Returns:
            List[np.ndarray]: 
                [slot_outputs, intent_outputs]
        """
    
        with tf.variable_scope(name_or_scope=model_name, reuse=tf.AUTO_REUSE):

            dtype = tf.get_variable_scope().dtype

            # create a word embedding
            # [Size of utterance vocabulary, word dimensionality]
            word_embedding = tf.get_variable(
                "word_embedding",
                [input_size, hidden_size],
                initializer=xavier_initializer(),
            )

            # get the current batch's word embedding
            # [batch, len_q, hidden_size]
            inputs_emb = tf.nn.embedding_lookup(word_embedding, input_data)

            # create a tag embedding 
            # [Size of slot vocabulary, word dimensionality]
            tag_embedding = tf.get_variable(
                "tag_embedding",
                [slot_size, hidden_size],
                initializer=xavier_initializer(),
            )
            # get the current batch's tag embedding
            # [batch, len_q, hidden_size]
            tags_emb = tf.nn.embedding_lookup(
                tag_embedding, input_tags
            )  
            
            # combine word and tag embeddings
            inputs = inputs_emb + tags_emb

            # insert CLS as the first token
            cls = tf.get_variable(
                "cls", 
                [hidden_size], 
                trainable=True, 
                initializer=xavier_initializer()
            )
            cls = tf.reshape(cls, [1, 1, -1])
            cls = tf.tile(cls, [tf.shape(inputs)[0], 1, 1])
            inputs = tf.concat([cls, inputs], 1)

            # [batch, len_q]
            src_mask = tf.sequence_mask(
                sequence_length + 1,
                maxlen=tf.shape(inputs)[1],
                dtype=dtype or tf.float32,
            )  
            src_mask.set_shape((None, None))

            if self.arg.multiply_embedding_mode == "sqrt_depth":
                inputs = inputs * (hidden_size ** -0.5)

            inputs = inputs * tf.expand_dims(src_mask, -1)
            bias = tf.get_variable("bias", [hidden_size])
            encoder_input = tf.nn.bias_add(inputs, bias)
            enc_attn_bias = layers.attention.attention_bias(
                src_mask, "masking", dtype=dtype
            )

            if self.arg.residual_dropout:
                if is_training:
                    keep_prob = 1.0 - self.arg.residual_dropout
                else:
                    keep_prob = 1.0
                encoder_input = tf.nn.dropout(encoder_input, keep_prob)

            # Feed into Transformer
            att_dropout = self.arg.attention_dropout
            res_dropout = self.arg.residual_dropout

            # case predict
            if not is_training:
                self.arg.attention_dropout = 0.0
                self.arg.residual_dropout = 0.0
            outputs = transformer_encoder(
                encoder_input, enc_attn_bias, self.arg
            )  # [batch, len_q + 1, out_size]
            self.arg.attention_dropout = att_dropout
            self.arg.residual_dropout = res_dropout

            intent_output, slot_output = tf.split(
                outputs, [1, tf.shape(outputs)[1] - 1], 1
            )

            with tf.variable_scope("intent_proj"):
                intent_state = intent_output
                intent_output = _ffn_layer(
                    intent_output, self.arg.hidden_size, intent_size, scope="intent"
                )

                # mask first token of intent_output forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(intent_output)) * -1e10
                mask_true = tf.ones(tf.shape(intent_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(intent_output), dtype=bool)
                intent_output_mask = tf.concat(
                    [mask_true[:, :, :2], mask_false[:, :, 2:]], -1
                )
                intent_output = tf.where(intent_output_mask, mask_values, intent_output)

            with tf.variable_scope("slot_proj"):

                slot_output = tf.concat(
                    [
                        slot_output,
                        tf.tile(intent_state, [1, tf.shape(slot_output)[1], 1]),
                    ],
                    2,
                )
                # slot_output = linear(slot_output, slot_size, True, True, scope="slot")  # [?, ?, slot_size]
                slot_output = _ffn_layer(
                    slot_output, self.arg.hidden_size, slot_size, scope="slot"
                )

                # mask first two tokens (_PAD_, _UNK_) of slot_outputs forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(slot_output)) * -1e10
                mask_true = tf.ones(tf.shape(slot_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(slot_output), dtype=bool)
                slot_outputs_mask = tf.concat(
                    [mask_true[:, :, :2], mask_false[:, :, 2:]], -1
                )
                slot_output = tf.where(slot_outputs_mask, mask_values, slot_output)

            outputs = [slot_output, intent_output]
        return outputs


    def create_loss(self, training_outputs, slots, slot_weights, intent):
        slots_shape = tf.shape(slots)
        slots_reshape = tf.reshape(slots, [-1])
        slot_outputs = training_outputs[0]
        slot_outputs = tf.reshape(slot_outputs, [tf.shape(slots_reshape)[0], -1])

        with tf.variable_scope("slot_loss"):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=slots_reshape, logits=slot_outputs
            )
            # crossent = tf.compat.v1.losses.softmax_cross_entropy
            crossent = tf.reshape(crossent, slots_shape)
            slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
            total_size = tf.reduce_sum(slot_weights, 1)
            total_size += 1e-12
            slot_loss = slot_loss / total_size

        intent_output = training_outputs[1]
        intent_output = tf.reshape(intent_output, [tf.shape(intent)[0], -1])

        with tf.variable_scope("intent_loss"):
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=intent, logits=intent_output
            )
            intent_loss = tf.reduce_mean(crossent)
        return slot_loss, intent_loss


    def create_placeholder(self):
        self.input_data = tf.placeholder(tf.int32, [None, None], name="seq_input")
        self.input_tags = tf.placeholder(tf.int32, [None, None], name="input_label")
        self.sequence_length = tf.placeholder(tf.int32, None, name="seq_length")
        self.slots = tf.placeholder(tf.int32, [None, None], name="slots")
        self.slot_weights = tf.placeholder(
            tf.float32, [None, None], name="slot_weights"
        )
        self.intent = tf.placeholder(tf.int32, None, name="intent")


    def create_train_graph(self):

        # feed into model
        self.train_outputs = self.create_model(
            self.input_data,
            self.input_tags,
            len(self.seq_in_tokenizer.word_index) + 1,
            self.sequence_length,
            len(self.seq_out_tokenizer.word_index) + 1,
            len(self.label_tokenizer.word_index) + 1,
            hidden_size=self.arg.hidden_size,
        )

        # get loss
        self.slot_loss, self.intent_loss = self.create_loss(
            self.train_outputs, self.slots, self.slot_weights, self.intent
        )
        self.loss = (
            self.arg.alpha * self.slot_loss + (1 - self.arg.alpha) * self.intent_loss
        )

        if self.arg.opt == "adam":
            self.train_op = self.add_optimizer(self.loss, self.global_step)
        else:
            self.train_op = self.add_optimizer(
                self.loss, self.global_step, isAdam=False
            )

        self.merged_summ = tf.summary.merge_all()


    def create_eval_graph(self):
        # reuse and feed into model
        self.eval_outputs = self.create_model(
            self.input_data,
            self.input_tags,
            len(self.seq_in_tokenizer.word_index) + 1,
            self.sequence_length,
            len(self.seq_out_tokenizer.word_index) + 1,
            len(self.label_tokenizer.word_index) + 1,
            hidden_size=self.arg.hidden_size,
            is_training=False,
        )
        self.eval_outputs.append(self.slots)
        self.eval_outputs.append(self.intent)
        self.eval_outputs.append(self.sequence_length)


    def create_test_graph(self):
        # reuse and feed into model

        self.test_outputs = self.create_model(
            self.input_data,
            self.input_tags,
            len(self.seq_in_tokenizer.word_index) + 1,
            self.sequence_length,
            len(self.seq_out_tokenizer.word_index) + 1,
            len(self.label_tokenizer.word_index) + 1,
            hidden_size=self.arg.hidden_size,
            is_training=False,
        )

        self.test_outputs.append(self.slots)
        self.test_outputs.append(self.intent)
        self.test_outputs.append(self.sequence_length)
        self.test_outputs.append(self.input_data)


    def create_inference_graph(self):

        # create model
        self.inference_outputs = self.create_model_predict(
            input_data=self.input_data,
            input_tags=self.input_tags,
            input_size=len(self.seq_in_tokenizer.word_index) + 1,
            sequence_length=self.sequence_length,
            slot_size=len(self.seq_out_tokenizer.word_index) + 1,
            intent_size=len(self.label_tokenizer.word_index) + 1,
            hidden_size=self.arg.hidden_size,
            is_training=False,
        )

        self.inference_outputs.append(self.sequence_length)
        self.inference_outputs.append(self.input_data)


    def train_one_epoch(self, sess, epoch, shuffle=True):
        """Run one training epoch"""
        losses = []
        slot_losses = []
        intent_losses = []
        cnt = 0
        step = 0
        train_path = os.path.join(self.full_train_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(train_path)

        while 1:
            batch, iterator, last_batch = self.get_batch_np(
                batch_iter, train_path, self.arg.batch_size
            )
            batch_iter = iterator
            (
                seq_in_ids,
                sequence_length,
                seq_out_ids,
                seq_out_weights,
                label_ids,
            ) = self.batch_process(batch)
            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx
            try:
                # first pass
                train_ouput, loss, slot_loss, intent_loss, _ = sess.run(
                    [
                        self.train_outputs,
                        self.loss,
                        self.slot_loss,
                        self.intent_loss,
                        self.train_op,
                    ],
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: first_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.slot_weights: seq_out_weights,
                        self.intent: label_ids,
                    },
                )
                losses.append(loss)
                slot_losses.append(slot_loss)
                intent_losses.append(intent_loss)

                # second pass
                slot = train_ouput[0]
                second_pass_in_tags = self.get_start_tags(slot)
                train_ouput, loss, slot_loss, intent_loss, _ = sess.run(
                    [
                        self.train_outputs,
                        self.loss,
                        self.slot_loss,
                        self.intent_loss,
                        self.train_op,
                    ],
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: second_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.slot_weights: seq_out_weights,
                        self.intent: label_ids,
                    },
                )
            except:
                print("Runtime Error in train_one_epoch")
                break

            cnt += self.arg.batch_size
            if step % 20 == 0:
                self.logger.info(
                    "E:{} Sents: {}: Train Loss \t{:.5}\t{:.5}\t{:.5}".format(
                        epoch,
                        cnt,
                        np.mean(loss),
                        np.mean(slot_loss),
                        np.mean(intent_loss),
                    )
                )
                losses = []
                slot_losses = []
                intent_losses = []

            if last_batch:
                break


    def evaluation(self, sess):
        """Do Evaluation"""

        def valid(eval_outputs):

            # intent
            pred_intent = eval_outputs[1][:, :, 2:].argmax(-1).reshape(-1) + 2

            # collect training outputs
            correct_intent = eval_outputs[3]
            sequence_length = eval_outputs[4]
            correct_slot = eval_outputs[2]
            
            # calculate intent accurancy
            intent_acc_sample_wise = correct_intent == pred_intent
            intent_acc = intent_acc_sample_wise.astype(np.float)
            intent_acc = np.mean(intent_acc) * 100.0

            # calculate slot accuracy
            pred_slot = eval_outputs[0].reshape(
                (correct_slot.shape[0], correct_slot.shape[1], -1)
            )
            pred_slot = pred_slot[:, :, 2:].argmax(-1) + 2
            
            # [batch_size, max_len]
            slot_acc_sample_wise = correct_slot == pred_slot  
            a = np.arange(correct_slot.shape[1])
            mask = np.tile(
                np.expand_dims(a, 0), [correct_slot.shape[0], 1]
            ) >= np.expand_dims(sequence_length, -1)

            slot_acc_sample_wise = np.logical_or(mask, slot_acc_sample_wise)
            slot_acc_sample_wise = np.logical_and.reduce(slot_acc_sample_wise, -1)
            slot_acc_sample_wise = slot_acc_sample_wise.astype(np.float)
            slot_acc = np.mean(slot_acc_sample_wise) * 100.0

            # sent acc
            sent_acc_sampel_wise = np.logical_and(
                intent_acc_sample_wise, slot_acc_sample_wise
            )
            sent_acc = np.mean(sent_acc_sampel_wise.astype(np.float)) * 100.0

            # calculate slot F1
            pred_slot_label = []
            correct_slot_label = []

            for pred_line, correct_line, length in zip(
                pred_slot, correct_slot, sequence_length
            ):
                pred_temp = []
                correct_temp = []
                for i in range(length):
                    pred_temp.append(self.seq_out_tokenizer.index_word[pred_line[i]])
                    correct_temp.append(
                        self.seq_out_tokenizer.index_word[correct_line[i]]
                    )
                pred_slot_label.append(pred_temp)
                correct_slot_label.append(correct_temp)

            f1, precision, recall = local_utils.computeF1Score(
                correct_slot_label, pred_slot_label
            )
            return f1, slot_acc, intent_acc, sent_acc

        step = 0
        f1 = 0
        slot_acc = 0
        intent_acc = 0
        sent_acc = 0
        sample_cnt = 0

        valid_path = os.path.join(self.full_valid_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(valid_path)

        while 1:
            step = step + 1
            
            # get utterance batch
            batch, iterator, last_batch = self.get_batch_np(
                batch_iter, valid_path, 1000
            )
            batch_iter = iterator

            # get indexed utterances, actual slots and labels
            seq_in_ids, sequence_length, seq_out_ids, _, label_ids = self.batch_process(
                batch
            )

            # get tags for first pass
            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # first pass
                eval_outputs = sess.run(
                    self.eval_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: first_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )
                # second pass
                slot = eval_outputs[0]
                second_pass_in_tags = self.get_start_tags(slot)
                eval_outputs = sess.run(
                    self.eval_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: second_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )
            except:
                logging.error("(evaluation) Runtime Error")
                break
            
            # metrics in percent (1 is 1 percent)
            f1_batch, slot_acc_batch, intent_acc_batch, sent_acc_batch = valid(
                eval_outputs
            )
            f1 = (f1 * sample_cnt + f1_batch * len(eval_outputs[0])) / (
                sample_cnt + len(eval_outputs[0])
            )
            slot_acc = (
                slot_acc * sample_cnt + slot_acc_batch * len(eval_outputs[0])
            ) / (sample_cnt + len(eval_outputs[0]))
            intent_acc = (
                intent_acc * sample_cnt + intent_acc_batch * len(eval_outputs[0])
            ) / (sample_cnt + len(eval_outputs[0]))
            sent_acc = (
                sent_acc * sample_cnt + sent_acc_batch * len(eval_outputs[0])
            ) / (sample_cnt + len(eval_outputs[0]))
            sample_cnt += len(eval_outputs[0])

            if last_batch:
                break
        print("----------------------------------------------------------------")
        logging.info(
            "(evaluation) Results: F1: {}, intent_acc: {}, slot_acc: {}, sent_acc: {}".format(
                f1, intent_acc, slot_acc, sent_acc
            )
        )
        print("----------------------------------------------------------------")
        logging.info(
            "(evaluation) Running Params: {}-{}-{}-{}-{}-{}-{}-{}".format(
                self.arg.batch_size,
                self.arg.lr,
                self.arg.hidden_size,
                self.arg.filter_size,
                self.arg.num_heads,
                self.arg.num_encoder_layers,
                self.arg.attention_dropout,
                self.arg.residual_dropout,
            )
        )

        return f1, slot_acc, intent_acc, sent_acc


    def inference_old_2(self, sess, epoch, diff, dump):

        """Do Inference"""

        def post_process(outputs):
            # intent
            # pred_intent = outputs[1].argmax(-1).reshape(-1)     # [batch_size]
            pred_intent = outputs[1][:, :, 2:].argmax(-1).reshape(-1) + 2

            # [batch_size]
            correct_intent = outputs[3]

            # slot
            # [batch_size, len, size]
            sequence_length = outputs[4]

            # [batch_size, len]
            correct_slot = outputs[2]
            pred_slot = outputs[0].reshape(
                (correct_slot.shape[0], correct_slot.shape[1], -1)
            )

            # [batch_size, len]
            # pred_slot = np.argmax(pred_slot, 2)
            pred_slot = pred_slot[:, :, 2:].argmax(-1) + 2

            # input sentence
            input_data = outputs[5]  # [batch_size, len]

            ref = []
            pred = []

            for words, c_i, p_i, seq_len, c_slot, p_slot in zip(
                input_data,
                correct_intent,
                pred_intent,
                sequence_length,
                correct_slot,
                pred_slot,
            ):
                words_output = " ".join(
                    [
                        self.seq_in_tokenizer.index_word[idx]
                        for idx, _ in zip(words, range(seq_len))
                    ]
                )
                c_i_output = self.label_tokenizer.index_word[c_i]
                c_slot_output = " ".join(
                    [
                        self.seq_out_tokenizer.index_word[idx]
                        for idx, _ in zip(c_slot, range(seq_len))
                    ]
                )
                p_i_output = self.label_tokenizer.index_word[p_i]
                p_slot_output = " ".join(
                    [
                        self.seq_out_tokenizer.index_word[idx]
                        for idx, _ in zip(p_slot, range(seq_len))
                    ]
                )
                ref.append("\t".join([words_output, c_i_output, c_slot_output]))
                pred.append("\t".join([words_output, p_i_output, p_slot_output]))
            return ref, pred

        step = 0
        if dump:
            fout = open(self.full_test_write_path, "w")
        test_path = os.path.join(self.full_test_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(test_path)

        cnt = 0
        while 1:
            step = step + 1

            batch, iterator, last_batch = self.get_batch_np(
                batch_iter, test_path, self.arg.batch_size
            )
            batch_iter = iterator
            seq_in_ids, sequence_length, seq_out_ids, _, label_ids = self.batch_process(
                batch
            )
            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # first pass
                infer_outputs = sess.run(
                    self.test_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: first_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )

                # second pass
                slot = infer_outputs[0]
                second_pass_in_tags = self.get_start_tags(slot)
                infer_outputs = sess.run(
                    self.test_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: second_pass_in_tags,
                        self.sequence_length: sequence_length,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )

            except:
                logging.error("(inference) Runtime Error in inference")
                break

            # output
            cnt += self.arg.batch_size
            if dump:
                ref_batch, pred_batch = post_process(infer_outputs)
                for ref_line, pred_line in zip(ref_batch, pred_batch):
                    # if diff and ref_line == pred_line:
                    #     continue
                    fout.write(ref_line + "\n")
                    fout.write(pred_line + "\n")
            if last_batch:
                break

        if dump:
            fout.flush()
            fout.close()

            # calculate uncoordinated chunk nums
            uncoordinated_nums = get_uncoordinated_chunking_nums(
                self.full_test_write_path
            )
            logging.error("(inference) uncoordinated nums : {}".format(uncoordinated_nums))


    @staticmethod
    def _post_process(outputs):
        # intent
        # pred_intent = outputs[1].argmax(-1).reshape(-1)     # [batch_size]
        pred_intent = outputs[1][:, :, 2:].argmax(-1).reshape(-1) + 2

        # [batch_size]
        correct_intent = outputs[3]

        # slot
        # [batch_size, len, size]
        sequence_length = outputs[4]

        # [batch_size, len]
        correct_slot = outputs[2]
        pred_slot = outputs[0].reshape(
            (correct_slot.shape[0], correct_slot.shape[1], -1)
        )

        # [batch_size, len]
        # pred_slot = np.argmax(pred_slot, 2)
        pred_slot = pred_slot[:, :, 2:].argmax(-1) + 2

        # input sentence
        input_data = outputs[5]  # [batch_size, len]

        ref = []
        pred = []

        for words, c_i, p_i, seq_len, c_slot, p_slot in zip(
            input_data,
            correct_intent,
            pred_intent,
            sequence_length,
            correct_slot,
            pred_slot,
        ):
            words_output = " ".join(
                [
                    self.seq_in_tokenizer.index_word[idx]
                    for idx, _ in zip(words, range(seq_len))
                ]
            )
            c_i_output = self.label_tokenizer.index_word[c_i]
            c_slot_output = " ".join(
                [
                    self.seq_out_tokenizer.index_word[idx]
                    for idx, _ in zip(c_slot, range(seq_len))
                ]
            )
            p_i_output = self.label_tokenizer.index_word[p_i]
            p_slot_output = " ".join(
                [
                    self.seq_out_tokenizer.index_word[idx]
                    for idx, _ in zip(p_slot, range(seq_len))
                ]
            )
            ref.append("\t".join([words_output, c_i_output, c_slot_output]))
            pred.append("\t".join([words_output, p_i_output, p_slot_output]))
        return ref, pred


    def inference_old(self, sess, epoch, diff, dump):

        """Do Inference
        [DEPRECATED]
        """
        

        if dump:
            fout = open(self.full_infer_write_path, "w")
        test_path = os.path.join(self.full_infer_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(test_path)

        cnt = 0
        step = 0
        while 1:
            step = step + 1
            batch, iterator, last_batch = self.get_batch_np(
                batch_iter, test_path, self.arg.batch_size
            )
            batch_iter = iterator
            # get longest sequence lengths
            # seq_out_ids: [n_utterances, longest utterance length] matrix of slot ids
            seq_in_ids, seq_len = self.batch_process_to_infer(
                batch
            )

            # [TODO]: FIND HOW TO ADD AND REMOVE !
            seq_out_ids = np.ones(seq_in_ids.shape)
            label_ids = np.ones((seq_in_ids.shape[0],1))

            # tags to embed and add to word embedding (see paper)
            pass_1_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # initialize all variables in model graph
                sess.run(tf.global_variables_initializer())

                # get learnt vocabulary
                slot_size=len(self.seq_out_tokenizer.word_index) + 1,
                intent_size=len(self.label_tokenizer.word_index) + 1,
                hidden_size=self.arg.hidden_size,
                
                # run first pass
                infer_outputs = sess.run(
                    self.inference_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: pass_1_in_tags,
                        self.sequence_length: seq_len,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )

                # second pass
                slot = infer_outputs[0]
                second_pass_in_tags = self.get_start_tags(slot)
                infer_outputs = sess.run(
                    self.inference_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: second_pass_in_tags,
                        self.sequence_length: seq_len,
                        self.slots: seq_out_ids,
                        self.intent: label_ids,
                    },
                )

            except:
                logging.error("(Inference) Runtime Error in inference")
                break

            # output
            cnt += self.arg.batch_size
            if dump:
                pred_batch = self._post_process_prediction(infer_outputs)
                for pred_line in pred_batch:
                    fout.write(pred_line + "\n")
            if last_batch:
                break

        if dump:
            fout.flush()
            fout.close()

            # calculate uncoordinated chunk nums
            uncoordinated_nums = get_uncoordinated_chunking_nums(
                self.full_test_write_path
            )
            logging.error("uncoordinated nums : {}".format(uncoordinated_nums))


    def inference(self, sess, epoch, diff, dump):
        
        """Do Inference"""

        if dump:
            fout = open(self.full_infer_write_path, "w")
        test_path = os.path.join(self.full_infer_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(test_path)

        cnt = 0
        step = 0
        while 1:
            step = step + 1
            batch, iterator, last_batch = self.get_batch_np(
                batch_iter, test_path, self.arg.batch_size
            )
            batch_iter = iterator
            # get longest sequence lengths
            # seq_out_ids: [n_utterances, longest utterance length] matrix of slot ids
            seq_in_ids, seq_len = self.batch_process_to_infer(
                batch
            )
            # tags to embed and add to word embedding (see paper)
            pass_1_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # initialize all variables in model graph
                # sess.run(tf.global_variables_initializer())

                # run first pass
                infer_outputs = sess.run(
                    self.inference_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: pass_1_in_tags,
                        self.sequence_length: seq_len,
                    },
                )

                # run second pass
                slot = infer_outputs[0]
                pass_2_in_tags = self.get_start_tags(slot)
                infer_outputs = sess.run(
                    self.inference_outputs,
                    feed_dict={
                        self.input_data: seq_in_ids,
                        self.input_tags: pass_2_in_tags,
                        self.sequence_length: seq_len,
                    },
                )

            except:
                logging.error("(Inference) Runtime Error in inference")
                break

            # output
            cnt += self.arg.batch_size
            if dump:
                pred_batch = self._post_process_prediction(infer_outputs)
                for pred_line in pred_batch:
                    fout.write(pred_line + "\n")
            if last_batch:
                break

        if dump:
            fout.flush()
            fout.close()

            # calculate uncoordinated chunk nums
            uncoordinated_nums = get_uncoordinated_chunking_nums(
                self.full_test_write_path
            )
            print("uncoordinated nums : {}".format(uncoordinated_nums))

    def _post_process_prediction_old(self, outputs):
        """[summary]

        Args:
            outputs ([type]):
                outputs[0]: model's score outputs for slots
                    [batch size, ?, nb of candidate slots]
                outputs[1]: model's score outputs for intents: 
                    [batch size, length of max utterance, nb of candidate intents]
                outputs[4]: sequence length
                outputs[5]: input data: [batch_size, len]

        Returns:
            [type]: [description]
        """
        # collect model's outputs
        slots_outputs = outputs[0][:,:,2:]
        pred_outputs = outputs[1][:,:,2:]
        seq_len = outputs[4]
        input_data = outputs[5]  

        # Slot prediction for each word
        # [batch size, max utterance length]
        # pred_slot idx belongs to [0, nb of candidate slots] 
        pred_slot = slots_outputs.argmax(-1) + 2  

        # Intent prediction
        # [batch size, 1, nb of candidate intents]
        pred_intent = pred_outputs.argmax(-1).reshape(-1)+2

        pred = []
        for words, pred_i, seq_len, pred_s in zip(
            input_data, pred_intent, seq_len, pred_slot
        ):
            # get utterance words
            # by their stored vocabulary index 
            words_output = " ".join(
                [
                    self.seq_in_tokenizer.index_word[idx]
                    for idx, _ in zip(words, range(seq_len))
                ]
            )
            # get intent tokens by their predicted index
            p_i_output = self.label_tokenizer.index_word[pred_i]

            # get slot tokens by their predicted index
            p_slot_output = " ".join(
                [
                    self.seq_out_tokenizer.index_word[idx]
                    for idx, _ in zip(pred_s, range(seq_len))
                ]
            )
            # record all
            pred.append("\t".join([words_output, p_i_output, p_slot_output]))
        return pred


    def _post_process_prediction(self, outputs:list):
        """[summary]

        Args:
            outputs (list):
                outputs[0]: model's score outputs for slots
                    [batch size, ?, nb of candidate slots]
                outputs[1]: model's score outputs for intents: 
                    [batch size, length of max utterance, nb of candidate intents]
                outputs[2]: sequence length
                outputs[3]: input data: [batch_size, len]
                    index 1: <unk> for unknown
                    index 0: empty

        Returns:
            list: predicted utterance, intent and slots
        """
        # collect model's outputs
        slots_outputs = outputs[0][:,:,2:]
        pred_outputs = outputs[1][:,:,2:]
        seq_len = outputs[2]
        input_data = outputs[3]  

        # Slot prediction for each word
        # [batch size, max utterance length]
        # pred_slot idx belongs to [0, nb of candidate slots] 
        pred_slot = slots_outputs.argmax(-1) + 2  

        # Intent prediction
        # [batch size, 1, nb of candidate intents]
        pred_intent = pred_outputs.argmax(-1).reshape(-1)+2
        pred = []
        for words, pred_i, seq_len, pred_s in zip(
            input_data, pred_intent, seq_len, pred_slot
        ):
            # get utterance words
            # by their stored vocabulary index 
            words_output = " ".join(
                [
                    self.seq_in_tokenizer.index_word[idx]
                    for idx, _ in zip(words, range(seq_len))
                ]
            )
            # get intent tokens by their predicted index
            p_i_output = self.label_tokenizer.index_word[pred_i]

            # get slot tokens by their predicted index
            p_slot_output = " ".join(
                [
                    self.seq_out_tokenizer.index_word[idx]
                    for idx, _ in zip(pred_s, range(seq_len))
                ]
            )
            # record it all
            pred.append("\t".join([words_output, p_i_output, p_slot_output]))
        return pred


    def save(self, sess):
        """Write model checkpoint

        Args:
            sess ([type]): context session containing the model variables to save
        """
        self.saver.save(sess, "./model/checkpoints/model")
