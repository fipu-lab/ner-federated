from __future__ import absolute_import, division, print_function

import os

import tensorflow as tf

from bert_modeling import BertConfig, BertModel
# from utils import tf_utils

from tensorflow import keras


class MaskedSparseCategoricalCrossentropy(tf.keras.losses.SparseCategoricalCrossentropy):
    def __call__(self, y_true, y_pred, **kwargs):
        label_ids, label_mask = y_true[0], y_true[1]
        label_ids_masked = tf.boolean_mask(label_ids, label_mask)
        logits_masked = tf.boolean_mask(y_pred, label_mask)
        return super().__call__(label_ids_masked, logits_masked, **kwargs)


class ValidationLayer(keras.layers.Layer):
    def call(self, sequence_output, valid_ids):
        sq = sequence_output
        vi = valid_ids

        def val_fn(i):
            cond = tf.equal(vi[i], tf.constant(1, dtype=tf.int32))
            temp = tf.squeeze(tf.gather(sq[i], tf.where(cond)))
            r = tf.tile(tf.zeros(tf.shape(sq[i])[1]), [tf.math.subtract(tf.shape(sq[i])[0], tf.shape(temp)[0])])
            r = tf.reshape(r, [-1, tf.shape(sq[i])[1]])
            n = tf.concat([temp, r], 0)
            return n

        n_vo = tf.map_fn(val_fn, tf.range(tf.shape(sq)[0]), dtype=tf.float32)
        return n_vo


def build_BertNer(bert_model, num_labels, max_seq_length):
    float_type = tf.float32
    input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
    input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
    input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
    valid_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='valid_ids')

    if type(bert_model) == str:
        bert_config = BertConfig.from_json_file(os.path.join(bert_model, "bert_config.json"))
    elif type(bert_model) == dict:
        bert_config = BertConfig.from_dict(bert_model)

    bert_layer = BertModel(config=bert_config, float_type=float_type)
    _, sequence_output = bert_layer(input_word_ids, input_mask, input_type_ids)

    val_layer = ValidationLayer()(sequence_output, valid_ids)

    dropout = tf.keras.layers.Dropout(rate=bert_config.hidden_dropout_prob)(val_layer)

    initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)

    classifier = tf.keras.layers.Dense(
        num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)(dropout)

    bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids, valid_ids], outputs=[classifier])

    return bert


"""
class BertNer(tf.keras.Model):

    # default is to train model from scratch
    def __init__(self, bert_model, float_type, num_labels, max_seq_length, final_layer_initializer=None):
        '''
        bert_model : string or dict
                     string: bert pretrained model directory with bert_config.json and bert_model.ckpt
                     dict: bert model config , pretrained weights are not restored
        float_type : tf.float32
        num_labels : num of tags in NER task
        max_seq_length : max_seq_length of tokens
        final_layer_initializer : default:  tf.keras.initializers.TruncatedNormal
        '''
        super(BertNer, self).__init__()
        
        input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name='input_type_ids')
        # bert_model is str for do_train and bert_model is dict for do_eval 
        if type(bert_model) == str:
            bert_config = BertConfig.from_json_file(os.path.join(bert_model,"bert_config.json"))
        elif type(bert_model) == dict:
            bert_config = BertConfig.from_dict(bert_model)

        bert_layer = BertModel(config=bert_config,float_type=float_type)
        _, sequence_output = bert_layer(input_word_ids, input_mask,input_type_ids)

        self.bert = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=[sequence_output])
        
        # true => use pretrained model
        # if pretrained:
        #     init_checkpoint = os.path.join(bert_model,"bert_model.ckpt")
        #    checkpoint = tf.train.Checkpoint(model=self.bert)
        #    checkpoint.restore(init_checkpoint)# .assert_existing_objects_matched()
        
        self.dropout = tf.keras.layers.Dropout(
            rate=bert_config.hidden_dropout_prob)
        
        if final_layer_initializer is not None:
            initializer = final_layer_initializer
        else:
            initializer = tf.keras.initializers.TruncatedNormal(stddev=bert_config.initializer_range)
        
        self.classifier = tf.keras.layers.Dense(
            num_labels, kernel_initializer=initializer, activation='softmax', name='output', dtype=float_type)
        
        self.__do_test_call(max_seq_length)
        

    def _inner_call(self, input_word_ids, input_mask, input_type_ids, valid_ids, **kwargs):
        sequence_output = self.bert([input_word_ids, input_mask, input_type_ids], **kwargs)
        valid_output = []
        for i in range(sequence_output.shape[0]):
            r = 0
            temp = []
            for j in range(sequence_output.shape[1]):
                if valid_ids[i][j] == 1:
                    temp = temp + [sequence_output[i][j]]
                else:
                    r += 1
            temp = temp + r * [tf.zeros_like(sequence_output[i][j])]
            valid_output = valid_output + temp
        valid_output = tf.reshape(tf.stack(valid_output),sequence_output.shape)
        sequence_output = self.dropout(
            valid_output, training=kwargs.get('training', False))
        logits = self.classifier(sequence_output)
        return logits
        
        
    def call(self, batch_data, **kwargs):
        input_word_ids,input_mask, input_type_ids, valid_ids = batch_data
        return self._inner_call(input_word_ids,input_mask, input_type_ids, valid_ids, **kwargs)
    
    
    def __do_test_call(self, seq_len):
        ids = tf.ones((1, seq_len), dtype=tf.int64)
        self((ids, ids, ids, ids, ), training=False)
"""
