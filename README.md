# Densenet with CBAM and Triplet Network

>Description coming soon ^_^

## What is this repository?

This repository contains the training pipeline code for our thesis proposal on offline signature verification using an **Enhanced Densenet with CBAM and Triplet Network**. 



## How to run?
> **Note**: [uv](https://docs.astral.sh/uv/) is recommended for running and managing this repository. Install [uv](https://docs.astral.sh/uv/guides/install-python/).

#####  1. Clone the repository locally

```bash
git clone 
```

##### 2. Install dependencies
```bash
uv venv && uv sync
```

##### 3. Run pipeline
```bash
uv run main.py
```
> That's it! All that's left is to wait for the pipeline to finish :>
 


### Footnotes

Densenet and CBAM building blocks were referenced from this repository [^1].


[^1]: DenseNet + CBAM [repository](https://github.com/kobiso/CBAM-keras/blob/master/models/densenet.py).

