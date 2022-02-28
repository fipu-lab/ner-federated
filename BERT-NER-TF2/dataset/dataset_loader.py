import logging
import tensorflow as tf
import numpy as np
import os

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, valid_ids=None, label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.valid_ids = valid_ids
        self.label_mask = label_mask


def readfile(filename, sep=' '):
    '''
    read file
    '''
    f = open(filename)
    data = []
    sentence = []
    label = []
    for line in f:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(sentence) > 0:
                data.append((sentence, label))
                sentence = []
                label = []
            continue
        splits = line.split(sep)
        sentence.append(splits[0])
        label.append(splits[-1][:-1])
        
    if len(sentence) > 0:
        data.append((sentence, label))
        sentence = []
        label = []

    return data



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list, 0)}

    features = []
    for (ex_index, example) in enumerate(examples):
        textlist = example.text_a.split(' ')
        labellist = example.label
        tokens = []
        labels = []
        valid_ids = []
        label_mask = []
        for i, word in enumerate(textlist):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                    valid_ids.append(1)
                    label_mask.append(True)
                else:
                    valid_ids.append(0)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
            valid_ids = valid_ids[0:(max_seq_length - 2)]
            label_mask = label_mask[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        valid_ids.insert(0, 1)
        label_mask.insert(0, True)
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            if len(labels) > i:
                label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        valid_ids.append(1)
        label_mask.append(True)
        label_ids.append(label_map["[SEP]"])
        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        label_mask = [True] * len(label_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)
            valid_ids.append(1)
            label_mask.append(False)
        while len(label_ids) < max_seq_length:
            label_ids.append(0)
            label_mask.append(False)


        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(valid_ids) == max_seq_length
        assert len(label_mask) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_ids,
                          valid_ids=valid_ids,
                          label_mask=label_mask))
    return features


def batch_features(features, label_list, seq_len, tokenizer, batch_size=64):

    all_input_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_ids for f in features]))
    all_input_mask = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.input_mask for f in features]))
    all_segment_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.segment_ids for f in features]))
    all_valid_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.valid_ids for f in features]))

    all_label_ids = tf.data.Dataset.from_tensor_slices(
        np.asarray([f.label_id for f in features]))

    eval_data = tf.data.Dataset.zip(
        (all_input_ids, all_input_mask, all_segment_ids, all_valid_ids, all_label_ids))
    batched_eval_data = eval_data.batch(batch_size)
    return batched_eval_data



def create_tf_dataset_for_client(input_ids, input_masks, segment_ids, label_ids, valid_ids, label_mask, batch_size=32):
    return tf.data.Dataset.from_tensor_slices(
        (
            (np.asarray(input_ids), np.asarray(input_masks), np.asarray(segment_ids), np.asarray(valid_ids)),
            (np.asarray(label_ids), np.asarray(label_mask))
        )
    ).shuffle(int(len(input_ids) * 0.1)).batch(batch_size, drop_remainder=True)



def split_to_tf_datasets(features, num_clients, batch_size=32):
    features_split = np.array_split(features, num_clients)
    ds_list = []
    for client_data in features_split:
        c_input_ids = [f.input_ids for f in client_data]
        c_input_mask = [f.input_mask for f in client_data]
        c_segment_ids = [f.segment_ids for f in client_data]
        c_valid_ids = [f.valid_ids for f in client_data]
        c_label_mask = [f.label_mask for f in client_data]
        c_label_id = [f.label_id for f in client_data]
        ds_list.append(
            create_tf_dataset_for_client(c_input_ids, 
                                         c_input_mask, 
                                         c_segment_ids, 
                                         c_label_id, 
                                         c_valid_ids, 
                                         c_label_mask, 
                                         batch_size))
    return ds_list



class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    
    def __init__(self, data_dir, initial_tokens=['[PAD]', 'O', '[CLS]', '[SEP]']):
        self.initial_tokens = initial_tokens or []
        self.data_dir = data_dir

    def get_train_examples(self):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        return self.initial_tokens
    
    def label_len(self):
        return len(self.get_labels())
    
    def get_label_map(self):
        return {i: label for i, label in enumerate(self.get_labels(), 0)}
    
    def token_ind(self, token):
        label_map = self.get_label_map()
        if token not in label_map.values():
            return -1
        return [k for k, v in label_map.items() if token == v][0]
    
    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        return readfile(input_file)
    
    
    def get_train_as_features(self, max_seq_length, tokenizer):
        return convert_examples_to_features(self.get_train_examples(), self.get_labels(), max_seq_length, tokenizer)
    
    def get_dev_as_features(self, max_seq_length, tokenizer):
        return convert_examples_to_features(self.get_dev_examples(), self.get_labels(), max_seq_length, tokenizer)
    
    def get_test_as_features(self, max_seq_length, tokenizer):
        return convert_examples_to_features(self.get_test_examples(), self.get_labels(), max_seq_length, tokenizer)
    
    

class NerProcessor(DataProcessor):
    """Processor for the CoNLL-2003 data set."""

    def get_train_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "train.txt")), "train")

    def get_dev_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "valid.txt")), "dev")

    def get_test_examples(self):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(self.data_dir, "test.txt")), "test")

    def get_labels(self):
        return super(NerProcessor, self).get_labels() + ["B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
    
    

class FewNERDProcessor(DataProcessor):
    """Processor for the Few-NERD data set."""
    
    def parse_examples(self, filename, set_type):
        examples = readfile(os.path.join(self.data_dir, filename), sep='\t')
        for i in range(len(examples)):
            for j in range(len(examples[i][1])):
                if examples[i][1][j] != 'O':
                    if examples[i][1][j] == '':
                        examples[i][1][j] = 'O'
                        continue
                    if len(examples[i][1][j]) > 3:
                        examples[i][1][j] = examples[i][1][j][:3]
                    examples[i][1][j] = examples[i][1][j].upper()
                    if examples[i][1][j] == 'OTH':
                        examples[i][1][j] = 'MISC'
                    examples[i][1][j] = 'I-' + examples[i][1][j]
        
        return self._create_examples(examples, set_type)
        
    
    def get_train_examples(self):
        """See base class."""
        return self.parse_examples("train.txt", "train")

    def get_dev_examples(self):
        """See base class."""
        return self.parse_examples("valid.txt", "dev")

    def get_test_examples(self):
        """See base class."""
        return self.parse_examples("test.txt", "test")

    def get_labels(self):
        return super(FewNERDProcessor, self).get_labels() + ["I-LOC", "I-ORG",  "I-PER", "I-MISC", "I-PRO", "I-ART", "I-BUI", "I-EVE"]

    def _create_examples(self, lines, set_type):
        examples = []
        for i, (sentence, label) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = ' '.join(sentence)
            text_b = None
            label = label
            examples.append(InputExample(
                guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples
