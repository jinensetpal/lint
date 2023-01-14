# Interpretability Tools as Feedback Loops

## Idea
[Class Activation Mappings (CAMs)](http://cnnlocalization.csail.mit.edu/) are a popular interpretability technique for CNNs. It allows us to identify the regions within the image that contribute to the final classification.

I think they can be used within the training sequence against a kernel loss, so as to leveraged `shared knowledge' within a dataset. In this case, most targets in the dataset are centred. Using the CAM, we check if the region outside the center has a high weightage. We take the mean squared error of this region, with a higher weight the further we move from the center.

Since there is a high loss assigned to the background, the model is less likely to overfit, and attempt to memorize the background, even if the dataset itself doesn't account for that.

Since CNNs are resistant to pixel-shift, positional generalization is not an issue, and during prediction, teh model ide

## Task Setup
In this (largely contrived) example, the CALTECH-256 dataset is restricted to two classes: orcas and leopards, reducing it to a binary classification task.

I've split up the test subset such that the leopards have a blue background, and the orcas have uncommon backgrounds (I could not find green, but I did find one with a yellow tint!). I'm trying to mislead the model as much as possible. 

The model architecture is standard, and doesn't change between experiments. It is only the training strategy that is updated. 

The model with the updated training strategy tended to have better performance than the standard model over a large set of runs.

## Running the Code

The repository is structured as a package, so most of the relative imports are handled when the code is run using the `-m` flag.

### Setup

```bash
pip install -r requirements.txt  # please use a virtual environment!
dvc pull -r origin
```

### Training

```bash
python -m src.model.train {camloss|default}
```

The 'default' argument trains the default model. Any other name will train using the updated strategy. The resultant model is saved under `models/{model_name}`

### CAMs

```bash
python -m src.data.generator {model_name}
```

CAMs are evaluated on `{model_name}`. It should be in the `models/` directory

## Project Structure

    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── cams                <- generated activation mappings from models
    │   ├── test
    │   ├── train
    │   └── val
    │
    ├── models                  <- trained and serialized models
    │   ├── default             <- standard model
    │   └── camloss             <- model with cam loss and training strategy applied
    │
    ├── slides                  <- talk slides
    │   └── img                 <- source images
    │
    ├── requirements.txt
    │
    └── src
        ├── data
        │   ├── augment.py      <- make sure leopards and orcas have equal training samples
        │   └── generator.py    <- dataset + cams generator
        │
        ├── models
        │   ├── arch.py         <- model architecture
        │   ├── callbacks.py    <- callbacks for managing training parameters
        │   ├── loss.py         <- kernel loss function
        │   ├── train.py        <- driver
        │   └── predict.py      <- prediction script
        │
        └── scripts             <- helper scripts
            ├── pipeline.sh     <- run all the code at once
            └── colab_setup.sh  <- automate colab setup
--------
