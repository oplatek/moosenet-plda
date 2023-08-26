<h1 align="center">MooseNet PLDA</h1>

<p align="center">
<a href="https://arxiv.org/abs/2301.07087"><b>MooseNet: A Trainable Metric for Synthesized Speech with a PLDA Module, on Arxiv.
</b></a><br>
<i>Accepted to Speech Synthesis Workshop 12, 2023, Grenoble</i></br>
<a href="https://docs.google.com/file/d/10pp0-azO-GOlxD_CCxlfRncfI9m7WixFOciv6cyNtZU/"><b>Presentation slides</b></a><br>
</p>

<p>&nbsp;</p>

# Moosenet PLDA
MooseNet is a trainable metric for synthesized speech. 
We experimented with SSL NN models and PLDA module.
See the [MooseNet-PLDA paper](https://arxiv.org/abs/2301.07087).


## Installation
```
# Optional for reinstallation
conda deactivate; rm -rf env; 
# Installing new conda environment and editable pip moosenet package
conda env create --prefix ./env -f environment.yml \
  && conda activate ./env \
  && pip install -e .[dev] 
```

## Reproducing the Experiments
- The commands for fine-tuning a SSL models (XLS-R and Wav2Vec 2.0) to MooseNet NN on the English data from the main track can be found in `./main.sh`
- For the commands for fine-tuning MooseNet NN on main and the Chinese set from OOD track see `./ood.sh`
