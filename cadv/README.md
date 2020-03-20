# cAdv

## Background
This code implements the cAdv attack in [Big but Imperceptible Adversarial Perturbations via Semantic Manipulation](https://arxiv.org/abs/1904.06347). cAdv adaptively changes the colors in certain regions of an image to create an adversarial examples. We do not bound Lp Norm but is still able to achieve realistic images.

Original Image of class Umbrella

<img src="test_images/n04507155_191.JPEG" width="224" height="224">

Output Image of class Golfcart

<img src="results/n04507155_191.png">


## Download the pretrained models
```bash
$ sh download_model.sh
```

## Running tests
To run a targeted attack, run the following
```bash
$ python test.py --num_iter 500 --targeted 1 --target 575 --gpu 0
```
num_iter is the number of update steps for the attack. Set targeted to 0 for untargeted attack.

## Citation
If you use our code for your research, please cite us 
```
@article{bhattad2019big,
  title={Big but Imperceptible Adversarial Perturbations via Semantic Manipulation},
  author={Bhattad, Anand and Chong, Min Jin and Liang, Kaizhao and Li, Bo and Forsyth, David A},
  journal={arXiv preprint arXiv:1904.06347},
  year={2019}
}
```

## Acknowledgements
This work and code is heavily based on <https://github.com/richzhang/colorization-pytorch>

