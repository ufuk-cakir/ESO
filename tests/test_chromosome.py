from eso.ga.chromosome import Chromosome

import pickle

import numpy as np
import pytest
import torch


class DummyGene:
    """A stand‑in for Gene that lets us control band_position and band_height."""

    def __init__(self, **kwargs):
        # allow passing explicit band_position/band_height, else default
        self.band_position = kwargs.get("band_position", 0)
        self.band_height = kwargs.get("band_height", 1)

    def get_band_position(self):
        return self.band_position

    def get_band_height(self):
        return self.band_height

    def _init_set_gene(self, new_pos, new_height):
        self.band_position = new_pos
        self.band_height = new_height

    def __str__(self):
        return f"({self.band_position},{self.band_height})"

    def __repr__(self):
        return f"DummyGene({self.band_position},{self.band_height})"


@pytest.fixture(autouse=True)
def stub_gene(monkeypatch):
    """Replace the real Gene with DummyGene so we get predictable behavior."""
    monkeypatch.setattr("eso.ga.chromosome.Gene", DummyGene)
    yield


def test_min_num_greater_than_max_raises():
    with pytest.raises(ValueError):
        Chromosome(
            num_genes=5,
            min_num_genes=10,
            max_num_genes=3,
            baseline_metric=0.0,
            baseline_parameters=1,
            gene_args={},
            model_args={},
        )


def test_random_num_genes(monkeypatch):
    # force np.random.randint to always pick 4
    monkeypatch.setattr(np.random, "randint", lambda a, b: 4)  # type: ignore
    c = Chromosome(
        num_genes=0,  # gets overridden
        min_num_genes=2,
        max_num_genes=10,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    assert c.num_genes == 4
    assert len(c.get_genes()) == 4


def test_sort_and_get_genes():
    c = Chromosome(
        num_genes=3,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    # overwrite with unsorted dummy genes
    g1 = DummyGene(band_position=5)
    g2 = DummyGene(band_position=1)
    g3 = DummyGene(band_position=3)
    c._genes = [g1, g2, g3]  # type: ignore
    sorted_list = c.sort().get_genes()
    assert [g.get_band_position() for g in sorted_list] == [1, 3, 5]


@pytest.mark.parametrize(
    "pos, band_pos, band_h",
    [
        (1, 7, None),  # only position set
        (0, None, 4),  # only height set
    ],
)
def test_set_gene_valid(pos, band_pos, band_h):
    c = Chromosome(
        num_genes=3,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )

    # grab the exact Gene instance that we'll mutate
    target = c.get_genes()[pos]
    old_pos = target.get_band_position()
    old_h = target.get_band_height()

    c.set_gene(pos, band_position=band_pos, band_height=band_h)

    # now assert on that same object
    if band_pos is not None:
        assert target.get_band_position() == band_pos
    else:
        assert target.get_band_position() == old_pos

    if band_h is not None:
        assert target.get_band_height() == band_h
    else:
        assert target.get_band_height() == old_h


@pytest.mark.parametrize(
    "args",
    [
        dict(position=-1, band_position=1, band_height=1),
        dict(position=5, band_position=1, band_height=1),
        dict(position=0, band_position=None, band_height=None),
    ],
)
def test_set_gene_invalid(args):
    c = Chromosome(
        num_genes=3,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    with pytest.raises(ValueError):
        c.set_gene(**args)


def test_get_info_contains_all_fields():
    c = Chromosome(
        num_genes=2,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=10.0,
        baseline_parameters=1000,
        gene_args={},
        model_args={},
    )
    # stub some internals
    c._metric = 0.75
    c._metric_name = "f1"
    c._trainable_parameters = 1234
    c._fitness = -0.5
    # force two dummy genes
    c._genes = [  # type: ignore
        DummyGene(band_position=0, band_height=1),
        DummyGene(band_position=2, band_height=3),
    ]
    info = c.get_info()
    assert "Chromosome Info:" in info
    assert "Number of Genes: 2" in info
    assert "F1: 0.75" in info
    assert "Trainable parameters: 1234" in info
    assert "Fitness: -0.5" in info
    assert "(0,1)" in info and "(2,3)" in info


def test_repr_and_str():
    c = Chromosome(
        num_genes=1,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    # stub one gene
    c._genes = [DummyGene(band_position=4, band_height=2)]  # type: ignore
    rep = repr(c)
    s = str(c)
    assert "Chromosome with 1 genes" in rep
    assert "Validation None: None" in s or "Validation" in s
    assert "Gene 1: (4,2)" in s


def test_get_bands_stack_and_concat():
    c = Chromosome(
        num_genes=2,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    # create a fake spectrogram of shape (5, 3)
    spect = np.arange(15).reshape(5, 3)
    # equal height → stack
    g1 = DummyGene(band_position=0, band_height=2)
    g2 = DummyGene(band_position=0, band_height=2)
    c._genes = [g1, g2]  # type: ignore
    out = c._get_bands(spect)
    # two bands of shape (2,3) stacked → (2,2,3)
    assert out.shape == (2, 2, 3)

    # unequal heights → concat along rows
    g1 = DummyGene(band_position=0, band_height=1)
    g2 = DummyGene(band_position=1, band_height=2)
    c._genes = [g1, g2]  # type: ignore
    out2 = c._get_bands(spect)
    # 1+2 rows → shape (3,3)
    assert out2.shape == (3, 3)


def test_create_dataset():
    c = Chromosome(
        num_genes=1,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )

    # stub _get_bands to tag each image
    def fake_bands(img):
        return img * 10

    c._get_bands = fake_bands  # type: ignore
    ds = np.array([[1, 2], [3, 4]])  # two "images"
    out = c._create_dataset(ds)
    # each sample multiplied by 10
    assert np.array_equal(out, ds * 10)


def test_save_and_load(tmp_path):
    c = Chromosome(
        num_genes=2,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    _ = tmp_path / "chromo.pkl"  # type: ignore
    c.save(str(tmp_path), "chromo")
    # ensure file was created
    p = tmp_path / "chromo.pkl"
    assert p.exists()
    # load and verify type and key attribute
    loaded = pickle.load(open(p, "rb"))
    assert isinstance(loaded, Chromosome)
    assert loaded.num_genes == c.num_genes


def test_save_model(tmp_path, monkeypatch):
    c = Chromosome(
        num_genes=0,
        min_num_genes=-1,
        max_num_genes=-1,
        baseline_metric=0.0,
        baseline_parameters=1,
        gene_args={},
        model_args={},
    )
    # stub a model state dict
    c._model_state_dict = {"foo": "bar"}
    # capture the torch.save call
    called = {}

    def fake_save(obj, path):
        called["obj"] = obj
        called["path"] = path

    monkeypatch.setattr(torch, "save", fake_save)

    # stub a logger
    class DummyLog:
        def __init__(self):
            self.msg = None

        def info(self, m):
            self.msg = m

    c.logger = DummyLog()  # type: ignore

    c.save_model(str(tmp_path), name="cstate")
    expected_path = str(tmp_path / "cstate.pth")
    assert called["obj"] == c._model_state_dict
    assert called["path"] == expected_path
    assert "saved to" in c.logger.msg  # type: ignore
