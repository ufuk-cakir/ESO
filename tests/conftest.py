# conftest.py (or at the top of your test file)
import pytest
import copy
from typing import List, Optional


class DummyGene:
    def __init__(
        self, position: int, height: int, min_pos=0, max_pos=100, min_h=1, max_h=10
    ):
        self._position = position
        self._height = height
        self.min_position = min_pos
        self.max_position = max_pos
        self.min_height = min_h
        self.max_height = max_h

    def get_band_position(self) -> int:
        return self._position

    def get_band_height(self) -> int:
        return self._height

    def set_band_position(self, position: int):
        self._position = max(
            self.min_position, min(position, self.max_position - self.get_band_height())
        )

    def set_band_height(self, height: int):
        # Add bounds checking
        self._height = max(self.min_height, min(height, self.max_height))
        # Adjust position if height change makes it invalid
        self.set_band_position(self._position)

    def __eq__(self, other):
        if not isinstance(other, DummyGene):
            return NotImplemented
        return self._position == other._position and self._height == other._height

    def __repr__(self):
        return f"G(p={self._position}, h={self._height})"


class DummyChromosome:
    def __init__(self, genes: List[DummyGene], n_channels: int = 1, id_=-1):
        self._genes = sorted(
            genes, key=lambda g: g.get_band_position()
        )  # Keep sorted internally
        self._n_channels = n_channels
        self.id = id  # For potential identification

    def get_genes(self) -> List[DummyGene]:
        return self._genes

    def set_gene(
        self,
        position: int,
        band_position: Optional[int] = None,
        band_height: Optional[int] = None,
    ):
        if 0 <= position < len(self._genes):
            gene_to_modify = self._genes[position]
            original_pos = gene_to_modify.get_band_position()
            _ = gene_to_modify.get_band_height()

            if band_height is not None:
                # Apply bounds from the gene itself
                new_height = max(
                    gene_to_modify.min_height,
                    min(band_height, gene_to_modify.max_height),
                )
                gene_to_modify._height = new_height  # Directly set, assuming set_gene overrides internal logic temporarily

                # If height changed, the new position might need re-evaluation based on the *new* height
                if band_position is None:
                    band_position = (
                        original_pos  # Keep original position if not specified
                    )

            if band_position is not None:
                # Apply bounds, considering the *current* height (which might have just been updated)
                new_position = max(
                    gene_to_modify.min_position,
                    min(
                        band_position,
                        gene_to_modify.max_position - gene_to_modify.get_band_height(),
                    ),
                )
                gene_to_modify._position = new_position  # Directly set

            # Note: The original code calls sort() after mutation.
            # set_gene itself doesn't sort in this dummy, mimicking the original structure.
        else:
            raise IndexError("Gene position out of bounds")

    def sort(self):
        # In-place sort based on band position
        self._genes.sort(key=lambda g: g.get_band_position())
        return self  # Return self to allow chaining like in original code

    def __len__(self) -> int:
        return len(self._genes)

    def __eq__(self, other):
        if not isinstance(other, DummyChromosome):
            return NotImplemented
        # Compare number of genes and each gene individually
        return len(self) == len(other) and all(
            g1 == g2 for g1, g2 in zip(self._genes, other._genes)
        )

    def __deepcopy__(self, memo):
        # Ensure deepcopy works correctly for the list of genes
        new_genes = [copy.deepcopy(gene, memo) for gene in self._genes]
        new_chromosome = DummyChromosome(new_genes, self._n_channels, self.id)  # type: ignore
        memo[id(self)] = new_chromosome
        return new_chromosome

    def __repr__(self):
        return f"Chr(id={self.id}, n_chan={self._n_channels}, genes={self._genes})"


class DummyPopulation:
    def __init__(self, chromosomes: List[DummyChromosome]):
        self._chromosomes = chromosomes

    def get_chromosomes(self) -> List[DummyChromosome]:
        return self._chromosomes

    def __len__(self) -> int:
        return len(self._chromosomes)


# --- Fixtures ---


@pytest.fixture
def default_rates():
    return {
        "crossover_rate": 0.6,
        "mutation_rate": 0.3,
        "reproduction_rate": 0.1,
        "mutation_height_range": 2,
        "mutation_position_range": 5,
    }


@pytest.fixture
def genetic_operator(default_rates):
    # Import inside fixture to ensure it uses potentially mocked dependencies if needed later
    from eso.ga.operator import GeneticOperator

    return GeneticOperator(**default_rates)


@pytest.fixture
def chromosome1():
    # Simple chromosome with 3 genes
    genes = [DummyGene(10, 5), DummyGene(30, 8), DummyGene(60, 3)]
    return DummyChromosome(genes, n_channels=1, id_=1)


@pytest.fixture
def chromosome2():
    # Another chromosome, potentially different length
    genes = [DummyGene(5, 7), DummyGene(25, 4), DummyGene(50, 6), DummyGene(80, 2)]
    return DummyChromosome(genes, n_channels=1, id_=2)


@pytest.fixture
def chromosome_multi_channel():
    genes = [
        DummyGene(15, 5),
        DummyGene(40, 5),
        DummyGene(70, 5),
    ]  # Height might be fixed
    return DummyChromosome(genes, n_channels=3, id_=3)  # Indicate multiple channels


@pytest.fixture
def population(chromosome1, chromosome2, chromosome_multi_channel):
    return DummyPopulation([chromosome1, chromosome2, chromosome_multi_channel])
