from copy import deepcopy
from eso.model.data import Data
from eso.model.model import Model
from eso.ga.selection import SelectionOperator
from eso.ga.operator import GeneticOperator
from eso.ga.population import Population
from eso.utils.settings import Config
from eso.utils.Evaluation import Evaluation
from eso.utils.unpickler import CPU_Unpickler
from eso.utils.logger import setup_logger, log_tensorboard, setup_tensorboard, plot_chromosome
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
import pickle
import pandas as pd

plt.switch_backend("agg")


class ESO:
    """The main class for the ESO algorithm.

    This class is responsible for training the baseline model and performing the genetic algorithm to find the optimal band positions and heights.
    The ESO class is initialized with a settings file. The settings file is a json file that contains all the parameters for the algorithm, or a dictionary containing the parameters.
    The settings file must contain the following parameters:
    - data: The parameters for the data
    - preprocessing: The parameters for the preprocessing
    - model: The parameters for the model
    - chromosome: The parameters for the chromosome
    - gene: The parameters for the gene
    - population: The parameters for the population
    - selection_operator: The parameters for the selection operator
    - genetic_operator: The parameters for the genetic operator
    - algorithm: The parameters for the algorithm

    Check the documentation for the parameters of each class.

    Parameters
    ----------
    settings : str or dict
        The path to the settings file or a dictionary containing the parameters
    stop_event : threading.Event, optional
        The event to stop the algorithm, by default None. Used if run from the GUI.
    logger : logging.Logger, optional
        The logger to use, by default None. If None, a logger is created.
    population_file_path : str, optional
        The path to the population file, by default None. If None, a new population is created.
    log_level : int, optional
        The log level to use, by default 0. Check the logging module for the different log levels.
    log_path : str, optional
        The path to the log file, by default None. If None, the log is not saved to a file.
    tensorboard_log_dir : str, optional
        The directory to log to tensorboard, by default None. If None, tensorboard is not used.
    results_path : str, optional
        The path to save the results, by default "results"
    progress_bar : tkinter progress bar, optional
        The progress bar to update, by default None. Only used if run from the GUI.
    progress_bar_training : tkinter progress bar, optional
        The progress bar to update during training, by default None. Only used if run from the GUI.

    Raises
    ------
    ImportError
        If the baseline.json file exists but could not be loaded.
    ValueError
        If max_generations is not specified in the settings file or as an argument to the method
    ValueError
        If the sum of mutation_rate, crossover_rate and reproduction_rate is not 1
    ValueError
        If the number of new chromosomes does not match the population size

    Examples
    --------
    >>> from eso import ESO
    >>> eso = ESO(settings="settings.json")
    >>> eso.opimize(max_generations=100)
    """

    def __init__(
        self,
        settings,
        stop_event=None,
        logger=None,
        population_file_path=None,
        log_level=0,
        log_path=None,
        tensorboard_log_dir=None,
        results_path="results",
        progress_handler=None,
    ):
        
        self.logger = setup_logger(
            logger=logger, log_path=log_path, log_level=log_level, name="eso"
        )
        self.evolution_logger = setup_logger(
            logger=None,
            log_path=log_path,
            log_level=log_level,
            name="evolution",
            add_stream_handler=False,
        )
        self.population_logger = setup_logger(
            logger=None,
            log_path=log_path,
            log_level=log_level,
            name="population",
            add_stream_handler=False,
        )

        


        init_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{init_time} Initializing ESO...")
        self.results_path = results_path
        #save the settings
        with open(os.path.join(self.results_path, "settings.json"), "w") as f:
            json.dump(settings, f, indent=4)
        
        
        self.config = Config(settings)
        self.logger.debug(f"Config: {self.config}")
        self.stop_event = stop_event
        self.population_file_path = population_file_path
        self.tensorboard_log_dir = tensorboard_log_dir
        self.progress_handler = progress_handler
        # If tensorboard_log_dir is None, self.writer is None
        os.makedirs(self.results_path, exist_ok=True)
        self._all_time_best_fitness = -np.inf
        self._best_chromosome = None

        # Allow -1 as a special value for the GUI, meaning "automatic sizing"
        if self.config.gene.band_height == -1:
            self.config.gene.band_height = None

        if self.config.gene.band_height is not None:
            self.band_height_fixed=True 
        else : self.band_height_fixed = False 

        if self.config.chromosome.num_genes is not None and self.config.chromosome.num_genes != -1 : 
            self.nb_genes_fixed = True 
        else : self.nb_genes_fixed = False

        if self.config.gene.band_position is not None and self.config.gene.band_position != -1 : 
            self.band_position_fixed = True
        else : self.band_position_fixed = False 
        

            

    def _get_baseline_results(self, ignore_results=False):
        baseline_path = os.path.join(self.results_path, "baseline.json")
        if os.path.exists(baseline_path):
            if ignore_results is False:
                try:
                    results = self._load_base_results(baseline_path)
                except Exception:
                    raise ImportError(
                        f"Could not load Baseline result from {baseline_path}."
                    )
            else:
                # Ignore results --> retrain
                results = self._train_baseline()
        else:
            results = self._train_baseline()
        return results

    def _write_baseline_results_to_config(self, results):
        # Set the values
        self.config.chromosome.baseline_parameters = results[
            "baseline_trainable_params"
        ]
        self.config.chromosome.baseline_metric = results["baseline_metric"]
        self.config.gene.spec_height = results["image_shape"][0]
        self._minimum_input_shape = results["minimum_input_shape"]
        self.logger.debug(f"Image shape: {results['image_shape'][0]}")
        self.logger.debug(
            f"Minimum Input Shape To Model:{results['minimum_input_shape']}"
        )

    def _train_baseline(self, full = False):
        """Train the baseline model

        First checks if the baseline results are already stored in the results folder. If not, it trains the baseline model and stores the results in the results folder.
        The baseline results are stored in a json file called baseline.json and contains the following information:
        - image_shape: The shape of the input images
        - baseline_trainable_params: The number of trainable parameters in the baseline model
        - baseline_metric: The metric of the baseline model

        Parameters
        ----------
        ignore_results : bool, optional
            If True, the baseline model is trained even if the baseline.json file exists, by default False

        Raises
        ------
        ImportError
            If the baseline.json file exists but could not be loaded.
        """
        self.logger.info("Training baseline model...")
        #TODO CHANGE THIS
        if type(self.config.model) == dict:
            model_args = self.config.model
        else:
            model_args = self.config.model.dict()
        # Create the dataset from Data class
        data = Data(
            **self.config.data.dict(),
            apply_preprocessing=True,
            logger=self.logger,
            preprocessing_args=self.config.preprocessing.dict(),
        )
        data.create_datasets()
        # load the data created
        X_train, Y_train = data.get_data("train")
        X_val, Y_val = data.get_data("validation")

        image_shape = data.get_image_shape()

        self.logger.info(f"Image shape: {image_shape}")
        self.logger.info(f"model_args: {model_args}")
        self.logger.info(f"architecture_args: {self.config.cnn_architecture.dict()}")
        self.logger.info(f"results_path: {self.results_path}")
        model = Model(self.results_path,
            input_shape=(1, image_shape[0], image_shape[1]),
            logger=self.logger, architecture_args= self.config.cnn_architecture.dict(),
            **model_args,
        )
        self.logger.info("Baseline Model initialized. Training...")
        # train the model
        model.train(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val)

        # save weight to compare with the best chromosome
        #model.save_model(self.results_path, "baseline")

        # evaluate
        self.baseline_metric, self.baseline_metric_name = model.evaluate(
            X_val=X_val, Y_val=Y_val
        )
        self.baseline_trainable_params = model.get_number_of_parameters()
        
        
        self.minimum_input_shape = Model(self.results_path,
            input_shape=(1, image_shape[0], image_shape[1]),
            logger=self.logger, architecture_args= self.config.cnn_architecture.dict(),
            **model_args, use_chromosome=True
        ).get_minimum_input_shape()
        results = {
            "image_shape": data.get_image_shape(),
            "baseline_trainable_params": self.baseline_trainable_params,
            "baseline_metric": self.baseline_metric,
            "baseline_metric_name": self.baseline_metric_name,
            "minimum_input_shape": self.minimum_input_shape,
            "full": full,
    
        }
        # write to json
        with open(os.path.join(self.results_path, "baseline.json"), "w") as f:
            json.dump(results, f, indent=4)
            self.logger.info("Saved Baseline results!")

        return results

    def _check_minimum_image_shape(self):
        minimum_shape = self._minimum_input_shape
        minimum_image_height = minimum_shape[0]
        
        
        
        if self.config.chromosome.min_num_genes != -1:
            min_num_genes = self.config.chromosome.min_num_genes
        else:
            min_num_genes = self.config.chromosome.num_genes
        if self.config.gene.band_height is None :
            min_height = self.config.gene.min_height
        else:
            min_height = self.config.gene.band_height
        # Calculate minimum image height of chromosome
        # TODO ADD A CHECK IF THE IMAGES ARE STACKED OR CONCATENATED
        # Images are stacked if band_height is set in config
        if (self.config.gene.band_height is None) & (self.config.chromosome.stack):
            raise ValueError(
                f"If the parameter stack == True for the chromosome, please add a value to band_height for the gene "
            )
        
        elif (self.config.gene.band_height is not None) & (self.config.chromosome.stack):
            height = self.config.gene.band_height
        elif (self.config.gene.band_height is not None) & (not self.config.chromosome.stack):
            height = min_height * min_num_genes
        else:
            height = self.config.gene.max_height * min_num_genes #added a step at _init_chromosome() that adapt min_height according to num_gene.
        
        self.logger.debug("---------------------------")
        self.logger.debug(f"Minimum Image Height: {minimum_image_height}")
        self.logger.debug(f"Minimum Number of Genes:{min_num_genes}")
        self.logger.debug(f"Minimum Height of one Gene:{min_height}")
        self.logger.debug(f"Band Height: {self.config.gene.band_height}")
        self.logger.debug(f"Images are stacked: {self.config.gene.band_height is not None}")
        self.logger.debug(f"Calculated Minimum Chromosome Image Shape: {height}")
        self.logger.debug("---------------------------")

        if height < minimum_image_height:
            raise ValueError(
                f"Input shape too small. The minimum image height of a chromosome is {height}, while it need to be at least {minimum_image_height}. Please change the settings."
            )
        else:
            self.logger.info("Minimum Image shape check passed!")
        
        self.config.gene.minimum_gene_height = minimum_shape[0]
        del self._minimum_input_shape

    def run(self):
        max_generations = self.config.algorithm.max_generations
        self.writer = setup_tensorboard(self.tensorboard_log_dir, self.logger)
        base_results = self._get_baseline_results()
        self._write_baseline_results_to_config(base_results)
        self._check_minimum_image_shape()
        if self.progress_handler:
            self.progress_handler.set_main_value(0)
            self.progress_handler.set_main_max(max_generations)

        self.optimize(max_generations)
        # Clean up and save
        if self.results_path is not None:
            self._save_results()
        if self.tensorboard_log_dir is not None:
            # Save hyperparameters
            metric_dict = {
                "metric": self._best_chromosome.get_metric(),
                "fitness": self._best_chromosome.get_fitness(),
                "trainable_params": self._best_chromosome.get_trainable_parameters(),
            }

            self.writer.add_hparams(
                self.config.get_params(),
                metric_dict=metric_dict,
            )
            # add dict as text
            self.writer.add_text(
                "Best Chromosome",
                str(self._best_chromosome),
                0,
            )
            # add hyperparams as text
            self.writer.add_text(
                "Hyperparameters",
                str(self.config.get_params()),
                0,
            )
            self.writer.close()
            
        self.logger.info("Algorithm finished!")
        self.logger.info("All-time best Chromosome:")
        self.logger.info(self._best_chromosome)
        self.logger.info("Now retraining for full epochs...")
        self._retrain_full()
        return self._best_chromosome

    def optimize(self, max_generations, log_evolution=False):
        """Perform Genetic Algorithm to find optimal band positions and heights.

        This method will first train the baseline model and then perform the genetic algorithm to find the optimal band positions and heights.
        At each epoch, the population is trained and then evolved. The best chromosome is logged to tensorboard.

        Parameters
        ----------
        max_generations : int, optional
            The maximum number of generations to run the algorithm for, by default None

        Raises
        ------
        ValueError
            If max_generations is not specified in the settings file or as an argument to the method
        ValueError
            If the sum of mutation_rate, crossover_rate and reproduction_rate is not 1
        ValueError
            If the number of new chromosomes does not match the population size
        """
        if log_evolution:
            with open(os.path.join(self.results_path, "evolution.log"), "w"):
                pass
        # TODO refactor this
        # only implement the optimization here
        self.logger.info("Optimizing...")
        data_dict = self.config.data.dict().copy()
        data_dict["force_recreate_dataset"] = False
        data = Data(
            apply_preprocessing=False,
            logger=self.logger,
            preprocessing_args=self.config.preprocessing.dict(),
            **data_dict,
        )
        self.logger.debug("Creating datasets for chromosomes...")
        data.create_datasets()
        # Check distribution
        self.logger.info(f"Encoding: {data.get_encoded_mapping()}")
        # Initialize Population
        if self.population_file_path is not None:
            # NOTE maybe this breaks if Baseline was trained again
            self.population = Population.load(
                self.population_file_path, data=data, logger=self.logger
            )
            self.logger.info(f"Loaded population from {self.population_file_path}")
            self.population_logger.info(f"Loaded population from {self.population_file_path}")
            self.population_logger.info(self.population)
        else:
            self.population = Population(self.results_path,
                **self.config.population.dict(),
                chromosome_args=self.config.chromosome.dict(),
                gene_args=self.config.gene.dict(),
                model_args=self.config.model.dict(),
                architecture_args=self.config.cnn_architecture.dict(),
                logger=self.logger,
                data=data,
            )
            self.logger.info("Creating Population from scratch.")
            self.population_logger.info("Creating Population from scratch.")
            self.population_logger.info(self.population)

        # Initialize Selection operator
        self.parent_selector = SelectionOperator(
            **self.config.selection_operator.dict(), 
        )

        # Initiliaze Genetic Operator
        self.genetic_operator = GeneticOperator(self.band_height_fixed,self.band_position_fixed,self.config.gene.spec_height,**self.config.genetic_operator.dict())
        start_eso_loop = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.logger.info(f"{start_eso_loop} Starting ESO...")
        for epoch in range(max_generations):
            # Check if the stop event is set
            if self.stop_event is not None:
                if self.stop_event.is_set():
                    self.logger.info("Stopping ESO...")
                    break

            if self.progress_handler: # << MODIFIED
                self.progress_handler.set_main_value(epoch + 1) # +1 because epoch is 0-indexed

            if epoch != 0:
                self.population.reset_trained_flags()
            # Train the population
            self.logger.info(f"---------- Epoch {epoch} / {max_generations} ----------")
            self.evolution_logger.info(
                f"---------- Epoch {epoch} / {max_generations} ----------"
            )
            self.population_logger.info(
                f"---------- Epoch {epoch} / {max_generations} ----------"
            )
            # this will evaluate the fitness of each chromosome
            stop = self.population.train_population(
                progress_handler=self.progress_handler,
                stop_event=self.stop_event,
            )
            
            self.evolve_population()

            # Log the best chromosome
            best_chromosome = self.population.get_best_chromosome()
            self.logger.debug(
                f"Current Best Chromosome Fitness: {best_chromosome.get_fitness()}"
            )
            if best_chromosome.get_fitness() > self._all_time_best_fitness:
                self.logger.debug("Better Chromosome!")
                
                self._best_chromosome = deepcopy(best_chromosome)
                self._all_time_best_fitness = best_chromosome.get_fitness()
                image_name_base = "all_time_best_chromosome"
                image_full_path = os.path.abspath(
                    os.path.join(self.results_path, f"{image_name_base}.png")
                )
                plot_chromosome(
                    best_chromosome,
                    self.config.gene.spec_height,
                    self.config.model.metric,
                    self.results_path,
                    name=image_name_base,
                )
                
                if self.progress_handler: # << NOTIFY HANDLER
                    self.progress_handler.notify_best_chromosome_image_updated(image_full_path)
                
                if self.results_path is not None:
                    self._save_results()

            # Log to Tensorboard
            log_tensorboard(
                best_chromosome=best_chromosome,
                epoch=epoch,
                writer=self.writer,
                tensorboard_log_dir=self.tensorboard_log_dir,
                image_height=self.config.gene.spec_height,
                metric_name=self.config.model.metric,
                results_path=self.results_path,
            )
            if stop:
                self.logger.debug("Stopped Training...")
                break
        # Stop the thread
        if self.stop_event is not None:
            self.stop_event.set()

    def _load_base_results(self, path):
        with open(path, "rb") as f:
            results = json.load(f)
        self.logger.info("Loaded Baseline results!")
        return results

    def evolve_population(self):
        """Evolve the population using the genetic operator and selection operator

        Creates new chromosomes using the genetic operator and replaces the old population with the new one.

        Raises
        ------
        ValueError
            If the sum of mutation_rate, crossover_rate and reproduction_rate is not 1
        ValueError
            If the number of new chromosomes does not match the population size
        """
        mutation_rate = self.config.genetic_operator.mutation_rate
        crossover_rate = self.config.genetic_operator.crossover_rate
        reproduction_rate = self.config.genetic_operator.reproduction_rate

        # Check if they add up to 1
        if round (mutation_rate + crossover_rate + reproduction_rate, 2) != 1:
            raise ValueError(
                "The sum of mutation_rate, crossover_rate and reproduction_rate must be 1"
            )
        population_size = len(self.population)
        self.evolution_logger.debug(f"Population size before: {population_size}")
        mutation_size = int(population_size * mutation_rate)
        # because crossover creates 2 offspring
        crossover_size = int((population_size * crossover_rate)) // 2
        # reproduction_size = population_size - mutation_size - crossover_size

        self.evolution_logger.debug(f"Mutation size: {mutation_size}")
        self.evolution_logger.debug(f"Crossover size: {crossover_size}")
        # TODO MAYBE MOVE THIS TO GENETIC OPERATOR CLASS
        new_chromosomes = []
        for _ in range(mutation_size):
            self.evolution_logger.info("....")
            self.evolution_logger.debug("Mutating...")
            parent = self.parent_selector.select_one_parent(self.population)
            self.evolution_logger.debug(f"Parent Mutation: {str(parent)}")
            offspring = self.genetic_operator.mutate(parent)
            self.evolution_logger.debug(f"Offspring Mutation: {str(offspring)}")
            new_chromosomes.append(offspring)
        
        for _ in range(crossover_size):
            self.evolution_logger.info("....")
            self.evolution_logger.debug("Crossover...")
            
            #need to take into account when several genes possible and the height of offsprings is < minimum_gene_height
            max_retries=5 #to avoid infinite loop
            retry_count =0
            while retry_count < max_retries:
                self.evolution_logger.debug(f"Retry count: {retry_count}")
                try:
                    self.evolution_logger.debug("Selecting parents for crossover...")
                    # Place your code here that might raise an error
                    parent1, parent2 = self.parent_selector.select_parents(self.population)
                    self.evolution_logger.debug(f"Parent1:{str(parent1)}")
                    self.evolution_logger.debug(f"Parent2:{str(parent2)}")

                    offspring1, offspring2 = self.genetic_operator.crossover(parent1, parent2)
                    self.evolution_logger.debug(f"Offspring1:{str(offspring1)}")
                    self.evolution_logger.debug(f"Offspring2:{str(offspring2)}")
                    break  # Exit the loop if successful
                
                except Exception as e:  # if height of the offsprings is too small need to redo the process
                    self.evolution_logger.error(f"Error during crossover: {e}")
                    print(f"Error occurred: {e}. Retrying... ({retry_count + 1}/{max_retries})")
                    retry_count += 1
                    if retry_count == max_retries:
                        print("Max retries reached. Exiting...")
            
            new_chromosomes.append(offspring1)
            new_chromosomes.append(offspring2)

        reproduction_size = population_size - len(new_chromosomes)
        self.evolution_logger.debug(f"Reproduction size: {reproduction_size}")
        for _ in range(reproduction_size):
            self.evolution_logger.info("....")
            self.evolution_logger.debug("Reproduction...")
            parent = self.parent_selector.select_one_parent(self.population)
            # Keep the parent
            new_chromosomes.append(parent)

        # Replace the old population with the new one
        if len(new_chromosomes) != population_size:
            raise ValueError(
                "The number of new chromosomes does not match the population size"
            )
        self.population.replace_chromosomes(new_chromosomes)
        del new_chromosomes
        self.evolution_logger.debug(
            f"Population evolved: population size: {len(self.population)}"
        )
        self.population_logger.info("new population : ")
        self.population_logger.info(self.population)
        self.evolution_logger.info("--------------------------------")
        self.population_logger.info("--------------------------------")

    def save(self):
        self._save_results()
        #with open(os.path.join(self.results_path, "eso.pkl"), "wb") as output_file:
          #  pickle.dump(self, output_file)

    def _save_results(self):
        self.population.save(os.path.join(self.results_path, "population"))
        if self._best_chromosome is not None:
            self._best_chromosome.save(self.results_path, "eso_chromosome")
            self._best_chromosome.save_model(self.results_path, name="eso_chromosome_cnn_state")
            self.logger.info(
                f"All-time best Chromosome from ESO saved to: {self.results_path}"
            )
        else:
            self.logger.info("No trained results, nothing saved...")

    
    
    def evaluate(self, test_type="simple", overlap=0.25, nb_to_group=2, threshold=0.8 ,save_name=None, force_calc_spectrograms=False):
        starting = datetime.now()
        
        
        f_baseline, confusion_matrix_baseline, baseline_params, baseline_image_shape, baseline_pixels, baseline_execution_time = self._evaluate_model(model_type="baseline", test_type=test_type, overlap=overlap, nb_to_group=nb_to_group, force_calc_spectrograms=force_calc_spectrograms, threshold=threshold)
        f_chromosome, confusion_matrix_chromosome, chromosome_params, chromosome_image_shape, chromosome_pixels, chromosome_execution_time = self._evaluate_model(model_type="chromosome", test_type=test_type, overlap=overlap, nb_to_group=nb_to_group, force_calc_spectrograms=force_calc_spectrograms, threshold=threshold)
        
        
        # Make confusion matrix into 1d string
        confusion_matrix_baseline_str = " ".join([" ".join(map(str, row)) for row in confusion_matrix_baseline])
        confusion_matrix_chromosome_str = " ".join([" ".join(map(str, row)) for row in confusion_matrix_chromosome])
        
        # Creates pandas dataframe
        df = pd.DataFrame(columns=["F1", "CONFUSION", "TIME", "PARAMS", "Image Shape", "Image Size"])
        df.loc["baseline"] = [
            f_baseline,
            confusion_matrix_baseline_str,
            baseline_execution_time,
            baseline_params,
            baseline_image_shape,
            baseline_pixels,
        ]
        df.loc["chromosome"] = [
            f_chromosome,
            confusion_matrix_chromosome_str,
            chromosome_execution_time,
            chromosome_params,
            chromosome_image_shape,
            chromosome_pixels,
        ]
        
        # Calculate improvement of chromosome model over baseline
        df["F1_improvement"] = (df["F1"] - df["F1"].shift(1)) / df["F1"].shift(1)
        df["TIME_improvement"] = (df["TIME"] - df["TIME"].shift(1)) / df["TIME"].shift(1)
        df["PARAMS_improvement"] = (df["PARAMS"] - df["PARAMS"].shift(1)) / df["PARAMS"].shift(1)
        df["Image Size Improvement"] = (df["Image Size"] - df["Image Size"].shift(1)) / df["Image Size"].shift(1)
              
        
        # Save to csv
        now = datetime.now()
        now = now.strftime("%Y-%m-%d_%H-%M-%S")
        if save_name is not None:
            path = os.path.join(self.results_path, save_name + "_evaluation_" + now + ".csv")
            df.to_csv(path)
            
            self.logger.info(f"Evaluation saved to: {path} ")
        else:    
            path = os.path.join(self.results_path, "evaluation_" + now + ".csv")
            df.to_csv(path)
            self.logger.info(f"Evaluation saved to: {path} ")
        print("------------------")
        print("RESULTS")
        print(df)
        return df
    
    def _retrain_full(self, num_epochs = 50, batch_size = 64, learning_rate = 0.001, force_retrain_baseline = True):
        model_args = {
            "optimizer_name": "adam",
            "loss_function_name": "cross_entropy",
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "shuffle": True,
            "metric": "f1",
        }
        self.config.model = model_args
        # Check if the baseline model is trained from baseline.json
        
        with open(os.path.join(self.results_path, "baseline.json"), "r") as read_file:
            data = json.load(read_file)
        
        skip_baseline = data["full"]
        
        if not skip_baseline:
            self.logger.info("TRAINING BASELINE MODEL")
            if force_retrain_baseline:
                full = False
            else:
                full = True
            self._train_baseline(full=full)
    
        self.logger.info("TRAINING CHROMOSOME MODEL")
        chrom_train_X,chrom_train_Y = self.population._data.get_data("train")
        chrom_val_X,chrom_val_Y = self.population._data.get_data("validation")
        
        chromosome = self._best_chromosome
        chromosome._model_args = model_args
        chromosome.train(chrom_train_X, chrom_train_Y, chrom_val_X, chrom_val_Y, save=True, model_name="eso_chromosome")
        chromosome.save_model(self.results_path,"eso_chromosome")
        chromosome.save_model(self.results_path, "eso_chromosome_cnn_state")
        self.logger.info("CHROMOSOME MODEL TRAINING FINISHED")
        
        
            
        
    
    def _evaluate_model(self, model_type="baseline",test_type="presegmented_dataset", overlap=0.25, nb_to_group=2, threshold=0.8, force_calc_spectrograms=False):
        if model_type == "baseline":
            self.logger.info(
                "Evaluate performance of the baseline model on the testing dataset"
            )

            # load the model
            self.logger.info("load the model ...")

            model_path = os.path.join(self.results_path, "baseline_cnn_state.pth")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Model.load_cnn(model_path, device)

            evaluation = Evaluation(
                species_folder=self.config.data.dict()["species_folder"],
                settings=self.config,
                overlap=overlap, 
                nb_to_group=nb_to_group,
                threshold=threshold, 
                force_calc_spectrograms=force_calc_spectrograms,
                logger=self.logger,
            )

        else:
            self.logger.info(
                "Evaluate performance of the model obtained with the best chromosome on the testing dataset"
            )

            # load the model
            self.logger.info("load the model ...")

            model_path = os.path.join(self.results_path, "eso_chromosome_cnn_state.pth")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = Model.load_cnn(model_path, device)

            # load the chromosome
            self.logger.info("load the best chromosome ...")

            infile = open(os.path.join(self.results_path, "eso_chromosome.pkl"), "rb")
            chromosome = CPU_Unpickler(infile).load()
            infile.close()

            evaluation = Evaluation(
                species_folder=self.config.data.dict()["species_folder"],
                settings=self.config,
                overlap=overlap, 
                nb_to_group=nb_to_group,
                threshold=threshold,
                chromosome=chromosome,
                apply_preprocessing=False,
                force_calc_spectrograms=force_calc_spectrograms,
                logger=self.logger,
            )

        return evaluation.run(model, test_type=test_type)
