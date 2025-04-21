conv_layers = 1
fc_layers = 2
max_pooling_size = 4
dropout_rate = 0.5
conv_filters = 8
conv_kernel = 8
fc_units = 32
epochs = 10
batch_size = 3

# TODO: maybe load this from a yaml file
CNN_ARCHITECTURE = {
    "conv_layers": conv_layers,
    "conv_filters": conv_filters,
    "dropout_rate": dropout_rate,
    "conv_kernel": conv_kernel,
    "max_pooling_size": max_pooling_size,
    "fc_units": fc_units,
    "fc_layers": fc_layers,
}


CHROMOSOME_CNN_ARCHITECTURE = CNN_ARCHITECTURE.copy()
