''' Settings types for the settings file. 

Dictionary containing the types of each setting. This is used to parse the settings from the TKinter GUI to the JSON file in the correct format.

TODO: write tests for this
'''

types = {
    "gene": {
        "min_height": int,
        "max_height": int,
        "min_position": int,
        "max_position": int
    },
    "chromosome": {
        "num_genes": int,
        "min_num_genes": int,
        "max_num_genes": int,
        "stack": bool
    },
    "algorithm": {
        "population_size": int,
        "max_generations": int,
        "mutation_rate": float,
        "crossover_rate": float,
        "reproduction_rate":float,
        "tournament_size": int
    },
    "preprocessing": {
        "preprocess": bool,
        "force_recreate_dataset": bool,
        "sample_rate": int,
        "lowpass_cutoff": int,
        "downsample_rate": int,
        "nyquist_rate": int,
        "segment_duration": int,
        "positive_class": str,
        "negative_class": str,
        "nb_negative_class": int,
        "file_type": str,
        "audio_extension": str,
        "n_fft": int,
        "hop_length": int,
        "n_mels": int,
        "f_min": int,
        "f_max": int,
        "species_folder": str,
        "keep_in_memory": bool,
    },
    "training":
    {
        "optimizer":str,
        "loss": str,
        "num_epochs": int,
        "batch_size":int,
        "learning_rate":float,
        "shuffle":bool,
        "lambda_1":float,
        "lambda_2":float,
    },
    "architecture": {
        "conv_layers": int,
        "conv_filters": int,
        "dropout_rate": float,
        "conv_kernel": int,
        "max_pooling_size": int,
        "fc_units": int,
        "fc_layers": int,

    }
}