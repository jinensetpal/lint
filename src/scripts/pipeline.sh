#!/bin/bash

python -m src.model.train
python -m src.model.train camloss

python -m src.data.generator
python -m src.data.generator camloss
