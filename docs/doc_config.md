# Configuration
The configuration file is organized as follows:

- _data_:
    - _len_input_: length of the input series;
    - _horizon_forecast_: forecasting horizon (length of the output series);
    - _size_train_: fraction of data used for training;
    - _size_valid_: fraction of data used for validation;
- _model_:
    - _stack_1_:
        - _kernel_size_pool_: kernel size of the MaxPool layer of the first stack;
        - _expr_ratio_m1_: inverse of the expressiveness ratio of the first stack;
    - _stack_2_:
        - _kernel_size_pool_: kernel size of the MaxPool layer of the second stack;
        - _expr_ratio_m1_: inverse of the expressiveness ratio of the second stack;
    - _stack_3_:
        - _kernel_size_pool_: kernel size of the MaxPool layer of the third stack;
        - _expr_ratio_m1_: inverse of the expressiveness ratio of the third stack;
    - _n_neur_hidden_: number of neurons in the hidden layer;
    - _frac_dropout_: fraction of dropout;
    - _stride_maxpool_: stride to use in the MaxPool layer;
- _training_:
    - _batch_size_: batch size used for training;
    - _n_epochs_: number of epochs used for training;
    - _patience_: patience used for early stopping;
    - _min_delta_loss_perc_: minimum percentage required improvement of the validation loss; if its change is smaller than this value, then the patience counter increases.