import os
import logging

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def do_train(model_name, lr, dataset, num_clients, num_train_clients,
             pretrained=False, frozen_bert=False,
             seq_len=128, batch_size=32,
             parallel_clients=None, devices='GPUS', is_notebook=True):
    assert num_clients >= num_train_clients

    if devices == 'CPU':
        os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
    elif devices == 'GPU:0':
        os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    elif devices == 'GPU:1':
        os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    import tensorflow as tf
    from dataset.dataset_loader import NerProcessor, FewNERDProcessor, split_to_tf_datasets, batch_features
    from utils.fl_utils import state_from_checkpoint, state_to_pretrained_model, state_to_model, evaluate, save_json
    from models.model import build_BertNer, MaskedSparseCategoricalCrossentropy
    from tokenization import FullTokenizer

    import numpy as np
    if is_notebook:
        import nest_asyncio
        nest_asyncio.apply()

    import tensorflow_federated as tff

    import bert_modeling

    if devices == 'CPU':
        cl_tf_devices = tf.config.list_logical_devices('CPU')
    elif devices == 'GPUS':
        cl_tf_devices = tf.config.list_logical_devices('GPU')
    else:
        cl_tf_devices = tf.config.list_logical_devices('GPU')[:1]

    clients_per_thread = 1 if parallel_clients is None else int(num_clients / parallel_clients)
    tff.backends.native.set_local_execution_context(
        # default_num_clients=10,
        clients_per_thread=clients_per_thread,
        # max_fanout=10,
        server_tf_device=tf.config.list_logical_devices('CPU')[0],
        client_tf_devices=cl_tf_devices)

    if dataset == 'conll':
        processor = NerProcessor('dataset/conll')
    else:
        processor = FewNERDProcessor('dataset/few_nerd')
    model_path = os.path.join("models", model_name)
    tokenizer = FullTokenizer(os.path.join(model_path, "vocab.txt"), True)
    train_features = processor.get_train_as_features(seq_len, tokenizer)
    eval_features = processor.get_test_as_features(seq_len, tokenizer)

    eval_data_batched = batch_features(eval_features, processor.get_labels(), seq_len, tokenizer,
                                       batch_size=batch_size * 2)

    def eval_model(model, eval_data, do_print=True):
        return evaluate(model, eval_data,
                        processor.get_label_map(),
                        processor.token_ind('O'),
                        processor.token_ind('[SEP]'),
                        processor.token_ind('[PAD]'),
                        do_print=do_print)

    dataset_list = split_to_tf_datasets(train_features, num_clients, batch_size=batch_size)

    def freeze_bert_layer(bert_model):
        for m_layer in bert_model.layers:
            if isinstance(m_layer, bert_modeling.BertModel):
                m_layer.trainable = False

    # Wrap a Keras model for use with TFF.
    def model_fn():
        model = build_BertNer(model_path, processor.label_len(), seq_len)
        if pretrained and frozen_bert:
            freeze_bert_layer(model)
        return tff.learning.from_keras_model(
            model,
            input_spec=dataset_list[0].element_spec,
            loss=MaskedSparseCategoricalCrossentropy())

    trainer = tff.learning.build_federated_averaging_process(
        model_fn=model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr['client']),
        server_optimizer_fn=lambda: tf.keras.optimizers.Adam(lr['server']),
        use_experimental_simulation_loop=True
    )

    state = trainer.initialize()
    global_model = build_BertNer(model_path, processor.label_len(), seq_len)
    if pretrained:
        if frozen_bert:
            freeze_bert_layer(global_model)
        state = state_from_checkpoint(state, global_model, model_path)

    res_list = []
    for rnd_ind in range(1, 501):
        train_data = list(np.random.choice(dataset_list, num_train_clients, replace=False))
        state, metrics = trainer.next(state, train_data)
        print("Round", rnd_ind, "Loss:", metrics['train']['loss'], "Examples:", metrics['stat']['num_examples'])
        examples = metrics['stat']['num_examples']
        if rnd_ind % num_train_clients == 0:
            if frozen_bert:
                state_to_pretrained_model(state, global_model, model_path)
            else:
                state_to_model(state, global_model)
            res = eval_model(global_model, eval_data_batched, do_print=True)
            res['Round'] = rnd_ind
            res['Examples'] = examples
            res_list.append(res)

    if clients_per_thread > 1:
        # This prevents the VRAM from growing to much since we already have
        # a memory constraint due to limited client number
        tff.framework.get_context_stack().current.executor_factory.clean_up_executors()

    save_json(filename='results-{}+{}+{}+(1)'.format(model_name,
                                                     'pretrained' if pretrained else 'nontrained',
                                                     processor.data_dir.split('/')[-1]),
              out_dict={'results': res_list, 'model': model_name, 'seq_len': seq_len,
                        'pretrained': pretrained, 'batch_size': batch_size, 'frozen_bert': frozen_bert}
              )
