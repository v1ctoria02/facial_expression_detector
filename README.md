# Facial expression generator

Convolutional Neural Network model to recognize facial expression.

## Install requirements to python environment:

```bash
pip install -r requirements.txt
```

## train and evaluate model:

```bash
python main.py --model <PATH_WHERE_TO_SAVE_TRAINED_MODEL>
# e.g.
python main.py --model data/model
```

## evaluate model on custom images:

```bash
python main.py --validate --model <PATH_TO_MODEL> --folder <PATH_TO_IMAGES>
# e.g.
python main.py --validate --model data/model --folder imagecsv
```

## Evaluate model on webcam video (Don't specify folder)

```bash
python main.py --validate --model <PATH_TO_MODEL>
# e.g.
python main.py --validate --model data/model
```