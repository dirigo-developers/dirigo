# Getting started

## Install

We recommend a fresh virtual environment (Conda or venv).

```bash
pip install -U pip
pip install dirigo
```

For development from source:

```bash
git clone https://github.com/dirigo-developers/dirigo
cd dirigo
pip install -e .
pip install -r docs/requirements.txt
```

## Quickstart (simulated hardware)

```python
# minimal skeleton — adjust to your actual API
import dirigo

def main():
    system = dirigo.System(simulated=True)
    with system.acquire() as run:
        frame = run.grab()
        print(frame.shape, frame.dtype)

if __name__ == "__main__":
    main()
```

::::{note}
Real hardware examples are provided as tutorials but are **not executed** on Read the Docs.
::::
