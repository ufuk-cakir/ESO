from .conftest import DummyChromosome

import pytest
from eso.ga.operator import GeneticOperator
import copy


def test_genetic_operator_initialization(default_rates):
    """Test successful initialization and rate storage."""
    print("Testing GeneticOperator initialization with default rates.")
    print(f"Default rates: {default_rates}")
    op = GeneticOperator(**default_rates)
    assert op._crossover_rate == default_rates["crossover_rate"]
    assert op._mutation_rate == default_rates["mutation_rate"]
    assert op._reproduction_rate == default_rates["reproduction_rate"]
    assert op._mutation_hight_range == default_rates["mutation_height_range"]
    assert op._mutation_position_range == default_rates["mutation_position_range"]


def test_genetic_operator_initialization_invalid_rates():
    """Test initialization fails if rates don't sum to 1."""
    invalid_rates = {
        "crossover_rate": 0.5,
        "mutation_rate": 0.3,
        "reproduction_rate": 0.1,  # Sum is 0.9
        "mutation_height_range": 2,
        "mutation_position_range": 5,
    }
    with pytest.raises(ValueError, match="sum of mutation_rate.*must be 1"):
        GeneticOperator(**invalid_rates)


def test_set_and_get_crossover_rate(genetic_operator):
    """Test setting and getting the crossover rate with validation."""
    original_rate = genetic_operator.get_crossover_rate()
    new_rate = 0.75
    genetic_operator.set_crossover_rate(new_rate)
    assert genetic_operator.get_crossover_rate() == new_rate

    # Test invalid types
    with pytest.raises(TypeError, match="crossover rate should be of type 'float'"):
        genetic_operator.set_crossover_rate("invalid")
    with pytest.raises(TypeError, match="crossover rate should be of type 'float'"):
        genetic_operator.set_crossover_rate(1)  # Integer

    # Test invalid values
    with pytest.raises(
        ValueError, match="crossover rate should be between 0.0 and 1.0"
    ):
        genetic_operator.set_crossover_rate(-0.1)
    with pytest.raises(
        ValueError, match="crossover rate should be between 0.0 and 1.0"
    ):
        genetic_operator.set_crossover_rate(1.0)  # Must be < 1.0
    with pytest.raises(
        ValueError, match="crossover rate should be between 0.0 and 1.0"
    ):
        genetic_operator.set_crossover_rate(1.1)

    # Restore original rate to not affect other tests if operator is shared
    genetic_operator.set_crossover_rate(original_rate)


def test_set_and_get_mutation_rate(genetic_operator):
    """Test setting and getting the mutation rate with validation."""
    original_rate = genetic_operator.get_mutation_rate()
    new_rate = 0.25
    genetic_operator.set_mutation_rate(new_rate)
    assert genetic_operator.get_mutation_rate() == new_rate

    # Test invalid types
    with pytest.raises(TypeError, match="mutation rate should be of type 'float'"):
        genetic_operator.set_mutation_rate("invalid")
    with pytest.raises(TypeError, match="mutation rate should be of type 'float'"):
        genetic_operator.set_mutation_rate(0)  # Integer

    # Test invalid values
    with pytest.raises(ValueError, match="mutation rate should be between 0.0 and 1.0"):
        genetic_operator.set_mutation_rate(-0.1)
    with pytest.raises(ValueError, match="mutation rate should be between 0.0 and 1.0"):
        genetic_operator.set_mutation_rate(1.0)  # Must be < 1.0
    with pytest.raises(ValueError, match="mutation rate should be between 0.0 and 1.0"):
        genetic_operator.set_mutation_rate(1.1)

    # Restore original rate
    genetic_operator.set_mutation_rate(original_rate)


def test_random_crossover(genetic_operator, chromosome1, chromosome2):
    """Test the random crossover exchanges genes between parents."""
    parent1_orig = copy.deepcopy(chromosome1)
    parent2_orig = copy.deepcopy(chromosome2)

    offspring1, offspring2 = genetic_operator.random_crossover(chromosome1, chromosome2)

    # Check return types
    assert isinstance(offspring1, DummyChromosome)
    assert isinstance(offspring2, DummyChromosome)

    # Check parents were not modified
    assert chromosome1 == parent1_orig
    assert chromosome2 == parent2_orig

    # Check offspring are different from parents (highly likely with random exchange)
    # Note: There's a small chance they end up identical if random choices swap identical genes.
    # Run multiple times or use a fixed seed if this becomes flaky.
    assert offspring1 != parent1_orig or offspring2 != parent2_orig

    # Check offspring length (can be same as parents or mixed if parents had different lengths)
    # The number of genes exchanged `m` is based on min(n1, n2). The lengths remain unchanged.
    assert len(offspring1) == len(parent1_orig)
    assert len(offspring2) == len(parent2_orig)

    # Basic check: sum of gene positions/heights might differ if genes were swapped
    sum_pos1 = sum(g.get_band_position() for g in offspring1.get_genes())
    sum_pos_orig1 = sum(g.get_band_position() for g in parent1_orig.get_genes())
    sum_pos2 = sum(g.get_band_position() for g in offspring2.get_genes())
    sum_pos_orig2 = sum(g.get_band_position() for g in parent2_orig.get_genes())

    # This is not a guarantee, but likely indicates a change occurred
    assert sum_pos1 != sum_pos_orig1 or sum_pos2 != sum_pos_orig2


def test_crossover(genetic_operator, chromosome1, chromosome2, mocker):
    """Test the standard (slice) crossover."""
    parent1_orig = copy.deepcopy(chromosome1)
    parent2_orig = copy.deepcopy(chromosome2)

    # Mock randint to control the crossover points for predictability
    # Crossover between index 1 (inclusive) and 3 (exclusive) in the shorter chromosome (len 3)
    mocker.patch(
        "random.randint", side_effect=[1, 3]
    )  # start_idx=1, end_idx=3 (exclusive)

    offspring1, offspring2 = genetic_operator.crossover(chromosome1, chromosome2)

    # Check return types and parent immutability
    assert isinstance(offspring1, DummyChromosome)
    assert isinstance(offspring2, DummyChromosome)
    assert chromosome1 == parent1_orig
    assert chromosome2 == parent2_orig

    # Verify the specific gene swap for the chosen slice [1, 3)
    # Offspring1 should have parent1[0], parent2[1], parent2[2]
    # Offspring2 should have parent2[0], parent1[1], parent1[2], parent2[3] (since len(p2)=4)
    assert offspring1.get_genes()[0] == parent1_orig.get_genes()[0]
    assert offspring1.get_genes()[1] == parent2_orig.get_genes()[1]  # Swapped
    assert offspring1.get_genes()[2] == parent2_orig.get_genes()[2]  # Swapped

    assert offspring2.get_genes()[0] == parent2_orig.get_genes()[0]
    assert offspring2.get_genes()[1] == parent1_orig.get_genes()[1]  # Swapped
    assert offspring2.get_genes()[2] == parent1_orig.get_genes()[2]  # Swapped
    assert (
        offspring2.get_genes()[3] == parent2_orig.get_genes()[3]
    )  # Unchanged (outside min length slice)


def test_crossover_avoids_full_swap(genetic_operator, chromosome1, chromosome2, mocker):
    """Test that crossover avoids swapping the entire chromosome when start=0, end=n."""
    parent1_orig = copy.deepcopy(chromosome1)
    parent2_orig = copy.deepcopy(chromosome2)
    n = min(len(chromosome1), len(chromosome2))  # n = 3

    # Mock randint to force start=0, end=n (which is 3)
    mocker.patch("random.randint", side_effect=[0, n])

    offspring1, offspring2 = genetic_operator.crossover(chromosome1, chromosome2)

    # The logic should change end_idx to n-1 = 2. Swap should happen for range [0, 2).
    # Offspring1 genes: p2[0], p2[1], p1[2]
    # Offspring2 genes: p1[0], p1[1], p2[2], p2[3]
    assert offspring1.get_genes()[0] == parent2_orig.get_genes()[0]  # Swapped
    assert offspring1.get_genes()[1] == parent2_orig.get_genes()[1]  # Swapped
    assert offspring1.get_genes()[2] == parent1_orig.get_genes()[2]  # Unchanged

    assert offspring2.get_genes()[0] == parent1_orig.get_genes()[0]  # Swapped
    assert offspring2.get_genes()[1] == parent1_orig.get_genes()[1]  # Swapped
    assert offspring2.get_genes()[2] == parent2_orig.get_genes()[2]  # Unchanged
    assert offspring2.get_genes()[3] == parent2_orig.get_genes()[3]  # Unchanged

    # Ensure they are different from a full swap
    full_swap_offspring1_genes = parent2_orig.get_genes()[
        :n
    ]  # This would be p2[0], p2[1], p2[2]
    assert offspring1.get_genes() != full_swap_offspring1_genes


# --- Mutation Tests ---


def test_mutate_gene_position(genetic_operator, chromosome1, mocker):
    """Test mutation of gene position."""
    original_chromosome = copy.deepcopy(chromosome1)
    _ = len(original_chromosome)

    # Mock the random choices:
    # - randint(0, n-1) to select the gene index (e.g., index 1)
    # - randint for mutation amount (e.g., +3)
    mocker.patch("random.randint", side_effect=[1, 3])  # Select gene 1, mutate by +3

    # Spy on the sort method
    sort_spy = mocker.spy(DummyChromosome, "sort")

    mutated_chromosome = genetic_operator._mutate_gene_position(chromosome1)

    # Assertions
    assert isinstance(mutated_chromosome, DummyChromosome)
    assert chromosome1 == original_chromosome  # Original unchanged
    assert mutated_chromosome != original_chromosome  # Mutated is different
    sort_spy.assert_called_once_with(mutated_chromosome)  # Check sort was called

    # Check specific gene change (gene at index 1 was chosen)
    original_gene = original_chromosome.get_genes()[1]
    mutated_gene = mutated_chromosome.get_genes()[
        1
    ]  # Assume sort keeps it at index 1 for this simple case

    expected_new_pos = original_gene.get_band_position() + 3  # 30 + 3 = 33
    # Check bounds (max_pos=100, height=8 -> max start = 92) -> 33 is okay
    assert mutated_gene.get_band_position() == expected_new_pos
    assert (
        mutated_gene.get_band_height() == original_gene.get_band_height()
    )  # Height unchanged
    # Check other genes unchanged
    assert mutated_chromosome.get_genes()[0] == original_chromosome.get_genes()[0]
    assert mutated_chromosome.get_genes()[2] == original_chromosome.get_genes()[2]


def test_mutate_gene_height(genetic_operator, chromosome1, mocker):
    """Test mutation of gene height."""
    original_chromosome = copy.deepcopy(chromosome1)
    _ = len(original_chromosome)

    # Mock the random choices:
    # - randint(0, n-1) to select the gene index (e.g., index 0)
    # - randint for mutation amount (e.g., -1)
    mocker.patch("random.randint", side_effect=[0, -1])  # Select gene 0, mutate by -1

    mutated_chromosome = genetic_operator._mutate_gene_height(chromosome1)

    # Assertions
    assert isinstance(mutated_chromosome, DummyChromosome)
    assert chromosome1 == original_chromosome  # Original unchanged
    assert mutated_chromosome != original_chromosome  # Mutated is different

    # Check specific gene change (gene at index 0 was chosen)
    original_gene = original_chromosome.get_genes()[0]
    mutated_gene = mutated_chromosome.get_genes()[0]

    expected_new_height = original_gene.get_band_height() - 1  # 5 - 1 = 4
    # Check bounds (min_h=1, max_h=10) -> 4 is okay
    assert mutated_gene.get_band_height() == expected_new_height
    assert (
        mutated_gene.get_band_position() == original_gene.get_band_position()
    )  # Position unchanged
    # Check other genes unchanged
    assert mutated_chromosome.get_genes()[1] == original_chromosome.get_genes()[1]
    assert mutated_chromosome.get_genes()[2] == original_chromosome.get_genes()[2]


def test_mutate_gene_both(genetic_operator, chromosome1, mocker):
    """Test mutation of both position and height."""
    original_chromosome = copy.deepcopy(chromosome1)
    _ = len(original_chromosome)

    # Mock the random choices:
    # - randint(0, n-1) to select the gene index (e.g., index 2)
    # - randint for height mutation (e.g., +2)
    # - randint for position mutation (e.g., -4)
    mocker.patch("random.randint", side_effect=[2, 2, -4])  # Gene 2, height +2, pos -4

    # Spy on sort
    sort_spy = mocker.spy(DummyChromosome, "sort")

    mutated_chromosome = genetic_operator._mutate_gene(chromosome1)

    # Assertions
    assert isinstance(mutated_chromosome, DummyChromosome)
    assert chromosome1 == original_chromosome  # Original unchanged
    assert mutated_chromosome != original_chromosome  # Mutated is different
    sort_spy.assert_called_once_with(mutated_chromosome)

    # Check specific gene change (gene at index 2 was chosen)
    original_gene = original_chromosome.get_genes()[2]
    # Find the potentially moved gene (might not be at index 2 after sort)
    # This is tricky. Let's assume sort keeps it at index 2 for simplicity here,
    # or test the state *before* sort if needed. Assuming it stays index 2:
    mutated_gene = mutated_chromosome.get_genes()[2]

    expected_new_height = original_gene.get_band_height() + 2  # 3 + 2 = 5
    expected_new_pos = original_gene.get_band_position() - 4  # 60 - 4 = 56
    # Check bounds: Height 5 (1-10 ok). Pos 56 (max start = 100 - 5 = 95) -> ok.
    assert mutated_gene.get_band_height() == expected_new_height
    assert mutated_gene.get_band_position() == expected_new_pos
    # Check other genes unchanged
    assert mutated_chromosome.get_genes()[0] == original_chromosome.get_genes()[0]
    assert mutated_chromosome.get_genes()[1] == original_chromosome.get_genes()[1]


def test_mutate_dispatch_multi_channel(
    genetic_operator, chromosome_multi_channel, mocker
):
    """Test that mutate() only calls _mutate_gene_position for n_channels > 1."""
    # Mock/Spy the position mutation method
    mock_pos = mocker.patch.object(
        genetic_operator,
        "_mutate_gene_position",
        wraps=genetic_operator._mutate_gene_position,
    )
    # Mock the other two just to ensure they are NOT called
    mock_height = mocker.patch.object(genetic_operator, "_mutate_gene_height")
    mock_both = mocker.patch.object(genetic_operator, "_mutate_gene")

    # No need to mock random.randint here, as the channel check bypasses it
    mutated = genetic_operator.mutate(chromosome_multi_channel)

    assert mutated != chromosome_multi_channel  # Ensure mutation happened
    mock_pos.assert_called_once_with(chromosome_multi_channel)
    mock_height.assert_not_called()
    mock_both.assert_not_called()


# --- Reproduction Test ---


def test_reproduce(genetic_operator, population, mocker):
    """Test reproduction selects a chromosome from the population."""
    pop_list = population.get_chromosomes()

    # Mock random.choice to return a specific element for predictability
    mocker.patch(
        "random.choice", return_value=pop_list[1]
    )  # Force selection of chromosome2

    selected_chromosome = genetic_operator.reproduce(population)

    assert selected_chromosome == pop_list[1]  # Check the specific one was returned
    assert selected_chromosome is pop_list[1]  # Check it's the *same* object (no copy)
    assert selected_chromosome in pop_list  # Check it belongs to the original list
