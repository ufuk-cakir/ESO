# Installation

ESO is a Python package with an optional PyQt5 GUI. Supported Python versions: 3.8 to 3.12.

## Clone

```bash
git clone https://github.com/ufuk-cakir/ESO.git
cd ESO
```

## Create an environment

=== "venv"

    ```bash
    python -m venv .venv
    source .venv/bin/activate          # macOS / Linux
    # .venv\Scripts\activate           # Windows
    ```

=== "uv"

    ```bash
    uv venv
    source .venv/bin/activate
    ```

=== "conda"

    ```bash
    conda create -n eso python=3.11
    conda activate eso
    ```

## Install PyTorch

PyTorch is not pinned. Choose a wheel that matches your hardware. The [official selector](https://pytorch.org/get-started/locally/) generates the right command.

=== "CUDA 12.6"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu126
    ```

=== "CPU only"

    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    ```

=== "macOS (MPS)"

    ```bash
    pip install torch
    ```

## Install ESO

```bash
pip install -e .
```

For development with tests and docs:

```bash
pip install -e ".[tests,docs]"
```

## Verify

```bash
python -c "from eso import ESO; print(ESO)"
```

If this prints the class object without an `ImportError`, the install is good.

## Next

- [First run](first-run.md). A minimal end-to-end experiment.
- [GUI](gui.md). The same pipeline driven by a graphical interface.
