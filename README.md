# Facial Expression Detection - CNN

Convolutional Neural Network model to recognize facial expressions.

Trained and validated on [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013) and combination of other datasets - not provided.

Place `train/` and (`test/` or `validation/`) directories in `data/`

## Install requirements to python environment:

```bash
pip install -r requirements.txt
```

## Set PYTHONPATH if imported modules cannot be found

```bash
export PYTHONPATH=.
```

## Train the model:

To train the model and save it to a specific path, run the following command:

```bash
python fed/main.py --model <PATH_TO_MODEL>
# e.g.
python fed/main.py --model data/models/ExpressionNet_fer
```

> [!NOTE]
> Model class name must be in model filename

## Evaluate model on custom images:

```bash
python fed/main.py --validate --model <PATH_TO_MODEL> --folder <PATH_TO_IMAGES>
# e.g.
python fed/main.py --validate --model data/models/ExpressionNet_fer --folder imagecsv
```

## Evaluate model on webcam

To run a webcam facial expression recognition don't specify `--folder`

```bash
python fed/main.py --validate --model <PATH_TO_MODEL>
# e.g.
python fed/main.py --validate --model data/models/ExpressionNet_2fer
```

## More info

```bash
python fed/main.py --help
```
