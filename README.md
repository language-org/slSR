# SlotRefine: A Fast Non-Autoregressive Model for Joint Intent Detection and Slot Filling

**Modified by**: Steeve Laquitaine  

**Original reference**:

Main paper to be cited ([Di Wu et al., 2020](https://www.aclweb.org/anthology/2020.emnlp-main.152.pdf))

```
@article{wu2020slotrefine,
  title={Slotrefine: A fast non-autoregressive model for joint intent detection and slot filling},
  author={Wu, Di and Ding, Liang and Lu, Fan and Xie, Jian},
  booktitle={EMNLP},
  year={2020}
}
```

## Requirements

* The system is built upon the [THUMT](https://github.com/THUNLP-MT/THUMT) codebase.
* conda

## Setup & fully train the model

* Setup:  
  
```bash
setup.sh      # create virtual env. and install dependencies
mkdir log     # create folder for logs  
```

Train on atis dataset: 

```bash
train.atis.sh # train model
```

Train on snips dataset: 

```bash
train.snips.sh
```

## Quick runs for testing

```bash
python models.py --patience=0 --dataset=atis --split=' ' --max_epochs=1
```

# Specs and stats

* Specs original paper:  
  * decoding:  
    * single Tesla P40 GPU

* Stats:  
  * inference latency: 3.02 ms  

* Specs:  
  * MacBook Pro (Mid 2012)
  * 2.3 GHz Quad-Core Intel Core i7
  * RAM: 8 GB 1600 MHz DDR3

* Stats:  
  * train: 4 hours (200 epochs)

