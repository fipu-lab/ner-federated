import numpy as np
import random
import tensorflow as tf
import tensorflow_federated as tff
from seqeval.metrics import classification_report, accuracy_score
import json
import os

# from sklearn.metrics import classification_report as sk_cp
# from tqdm.notebook import tqdm


def set_seed(seed):
    if seed:
        random.seed(seed)
        tf.random.set_seed(seed)
        np.random.seed(seed)
        print("Seed:", seed)


def get_sample_clients(c_list, num_clients):
    random_indices = np.random.choice(len(c_list), size=num_clients, replace=False)
    return np.array(c_list)[random_indices]


def state_to_model(server_state, keras_model):
    server_state.model.assign_weights_to(keras_model)
    return keras_model


def restore_model_ckpt(model, checkpoint_path):
    checkpoint = tf.train.Checkpoint(model=model)
    latest_chkpt = tf.train.latest_checkpoint(checkpoint_path)
    checkpoint.restore(latest_chkpt).run_restore_ops()
    return model


def state_from_checkpoint(server_state, model, checkpoint_path):
    restore_model_ckpt(model, checkpoint_path)
    server_state = tff.learning.state_with_new_model_weights(
        server_state,
        trainable_weights=[v.numpy() for v in model.trainable_weights],
        non_trainable_weights=[v.numpy() for v in model.non_trainable_weights]
    )
    return server_state


def state_to_pretrained_model(current_state, model_instance, model_path):
    model_instance = restore_model_ckpt(model_instance, model_path)
    model_instance.layers[-1].set_weights(current_state.model.trainable)
    return model_instance


def evaluate_state(server_state, keras_model, batched_eval_data, label_map, out_ind, sep_ind, do_print=True):
    return evaluate(state_to_model(server_state, keras_model), batched_eval_data, label_map, out_ind, sep_ind, do_print)


def evaluate(model, batched_eval_data, label_map, out_ind, sep_ind, pad_ind, do_print=True):
    y_true, y_pred = [], []
    for (input_ids, input_mask, segment_ids, valid_ids, label_ids) in batched_eval_data:
        logits = model((input_ids, input_mask, segment_ids, valid_ids), training=False)
        logits = tf.argmax(logits, axis=2)
        for i in range(label_ids.shape[0]):
            lbl_ids = label_ids[i]
            pred_ids = logits[i]
            cond = tf.not_equal(lbl_ids, tf.constant(pad_ind, dtype=tf.int64))
            lbl_ids = tf.squeeze(tf.gather(lbl_ids, tf.where(cond)))[1:-1]
            pred_ids = tf.squeeze(tf.gather(pred_ids, tf.where(cond)))[1:-1]

            temp_1 = list(map(lambda x: label_map[x.numpy()], lbl_ids))
            temp_2 = list(map(
                lambda x: label_map[x.numpy()] if label_map[x.numpy()] not in ['[SEP]', '[PAD]', '[CLS]'] else
                label_map[out_ind], pred_ids))

            y_true.append(temp_1)
            y_pred.append(temp_2)

    accuracy = accuracy_score(y_true, y_pred)

    if do_print:
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    out_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    out_dict['accuracy'] = accuracy
    return out_dict


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_json(filename, out_dict):
    os.makedirs('log/', exist_ok=True)
    i = 1
    while os.path.exists("log/{}.json".format(filename)):
        filename = filename.replace('+({})'.format(i), '+({})'.format(i+1))
        i += 1
    with open("log/{}.json".format(filename), "w") as outfile:
        json.dump(out_dict, outfile, indent=None, cls=NpEncoder)


"""
def evaluate(model, batched_eval_data, label_map, out_ind, sep_ind, do_print=True):
    # batched_eval_data =  parse_evaluation_data(eval_examples, label_list, seq_len, tokenizer, eval_batch_size)
    
    y_true, y_pred = [], []
    # label_map = {i : label for i, label in enumerate(label_list, 0)}
    
    # out_ind = OUT_IND
    # sep_ind = SEP_IND
    for (input_ids, input_mask, segment_ids, valid_ids, label_ids) in batched_eval_data: #  tqdm(batched_eval_data, position=0, leave=False, desc="Evaluating"):
            logits = model((input_ids, input_mask, segment_ids, valid_ids), training=False)
            logits = tf.argmax(logits, axis=2)
            for i, label in enumerate(label_ids):
                temp_1 = []
                temp_2 = []
                for j, m in enumerate(label):
                    # Skipping padding when evaluating
                    if j == 0:
                        continue
                    elif label_ids[i][j].numpy() == sep_ind:
                        y_true.append(temp_1)
                        y_pred.append(temp_2)
                        break
                    else:
                        temp_1.append(label_map[label_ids[i][j].numpy()])
                        pred_ind = logits[i][j].numpy()
                        # SEP, PAD and CLS are considered 'O' (out of vocabulary)
                        if label_map[pred_ind] in ['[SEP]', '[PAD]', '[CLS]']:
                            pred_ind = out_ind
                        temp_2.append(label_map[pred_ind])
    
    accuracy = accuracy_score(y_true, y_pred)
    
    if do_print:
        # print("Accuracy: {}".format(accuracy))
        print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    out_dict = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    out_dict['accuracy'] = accuracy
    return out_dict
"""
