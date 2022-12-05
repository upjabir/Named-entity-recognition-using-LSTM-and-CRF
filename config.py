class CFG:
    # Special tokens and corresponding _indexes
    word_pad = ('<pad>', 0)
    word_oov = ('<oov>', 1)
    entity_pad = ('<p>', 0)
    entity_bos = ('<bos>', 1)
    entity_eos = ('<eos>', 2)

    data_dir = 'data/'
    raw_data_dir = 'data/train.csv'
    lookup_path = data_dir + '/lookup.pkl'

    train_data ='data/grouped_train.csv'
    test_data = 'data/grouped_test.csv'
    val_data = 'data/grouped_val.csv'

    pretrained_embed_path = 'data/glove.6B.100d.txt'
    word_embed_dim=100

    batch_size = 16
    num_epochs = 100
    max_len = 300

    num_blocks = 1
    num_heads = 4
    model_dim = 256
    ff_hidden_dim = 512
    dropout_rate = 0.2
    lstm_hidden_dim = model_dim // 2

    min_delta = 0.
    patience = 5
    lr = 1e-5
    lr_decay_factor = 0.9
    weight_decay = 0.001
    min_lr = 5e-5

    saved_model_path = 'savedModels'
    trained_best_model = saved_model_path + '/model_best.pt'

    result_csv = data_dir + 'result.csv'