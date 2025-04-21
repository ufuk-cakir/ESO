from settings import Config


b = Config()
# default settings
print("default")
print(b.chromosome.dict())
print(b.input, b.algorithm)
c = Config("/Users/ufuk/1. Research/AIMS/Project Repo/ESO/settings/settings.json")
# default settings
print("json file")
print(c.chromosome.dict())
print(c.input, c.algorithm)
settings = {
    "algorithm": {
        "max_generations": 100,
    },
    "genetic_operator": {
        "mutation_rate": 0.1,
        "crossover_rate": 0.8,
        "reproduction_rate": 0.1,
    },
    "selection_operator": {"tournament_size": 10},
    "data": {
        "preprocess": True,
        "force_recreate_dataset": False,
        "lowpass_cutoff": 2000,
        "downsample_rate": 4800,
        "nyquist_rate": 2400,
        "segment_duration": 4,
        "positive_class": "gibbon",
        "negative_class": "no-gibbon",
        "nb_negative_class": 20,
        "file_type": "svl",
        "audio_extension": ".wav",
        "n_fft": 1024,
        "hop_length": 256,
        "n_mels": 128,
        "f_min": 4000,
        "f_max": 9000,
        "keep_in_memory": False,
        "species_folder": "/Users/ufuk/1. Research/AIMS/Project Repo/ESO/data",
    },
    "population": {
        "pop_size": 10,
    },
    "gene": {"min_position": 0, "max_position": -1, "min_height": 1, "max_height": 10},
    "chromosome": {
        "num_genes": 10,
        "min_num_genes": -1,
        "max_num_genes": -1,
        "lambda_1": 0.5,
        "lambda_2": 0.5,
    },
    "model": {
        "optimizer": "adam",
        "loss": "cross_entropy",
        "num_epochs": 1,
        "batch_size": 128,
        "learning_rate": 0.001,
        "shuffle": True,
    },
}

a = Config(settings)
print("loaded from dict")
print(a.input, a.algorithm)
print(a.chromosome.dict())
