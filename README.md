# D2G2
This repository maintains the implementation of D2G2 (SDM 21), a generic framework of deep generative models for interpretable dynamic graph generation. Detailded information about D2G2 is provided in [D2G2](https://github.com/vanbanTruong/vanbanTruong.github.io/blob/master/assets/SDM21.pdf). 

## Requirements
* PyTorch 1.4 or higher
* Python 3.7


## Instructions
1. Clone/download this repository.
2. Add datasets consisting of adjacency matrix and feature matrix with time dimension to dataset folder.   
3. Run the code.  
      * model.py: the D2G2 model.
      * trainer.py: train D2G2.
      * evaluate.py: eveluate D2G2.
  
## Reference
@inproceedings{zhang2021disentangled,  
     title={Disentangled Dynamic Graph Deep Generation},  
     author={Zhang, Wenbin and Zhang, Liming and Pfoser, Dieter and Zhao, Liang},  
     booktitle={Proceedings of the 2021 SIAM International Conference on Data Mining (SDM)},  
     pages={738--746},  
     year={2021},  
     organization={SIAM}  
}
