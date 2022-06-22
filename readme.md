## Saliency Detection

This repo contains assignment of course "Visual cognitive engineering" in HUST.AIA

#### Dataset

Using [SALICON](http://salicon.net/challenge-2017/) dataset.

#### ITTI & MDC

Apply [ITTI](https://link.zhihu.com/?target=https%3A//citeseerx.ist.psu.edu/viewdoc/download%3Fdoi%3D10.1.1.53.2366%26rep%3Drep1%26type%3Dpdf) algorithm and [MDC](https://ieeexplore.ieee.org/abstract/document/7937833/?casa_token=P4mq3ptifM4AAAAA:B3ZWwyqqCkmdDMaQT-Z5VOIZ3cOXcw9SRTupXU70jZRz7d0zkiFwQR93g11spWd8ya7BsNM9TQ8c) algorithm for saliency detection. See more details in `demo.ipynb`

#### MLNet

apply deep learning model mentioned in [A deep multi-level network for saliency prediction](https://ieeexplore.ieee.org/abstract/document/7900174/?casa_token=fNw2eBOemAQAAAAA:_3k0-byfIXcO3ZUIRanft8D28b1dD9Idt26JcatRMbvF5IWGbvjUs-s8rUEeQEh2iYElfv4ACPsI)

To train the model, run:

```sh
bash run.sh
```

To see a demo, please refer to `inference.ipynb`

#### Reference

```
@inproceedings{cornia2016deep,
  title={A deep multi-level network for saliency prediction},
  author={Cornia, Marcella and Baraldi, Lorenzo and Serra, Giuseppe and Cucchiara, Rita},
  booktitle={2016 23rd International Conference on Pattern Recognition (ICPR)},
  pages={3488--3493},
  year={2016},
  organization={IEEE}
}

@article{huang2017300,
  title={300-FPS salient object detection via minimum directional contrast},
  author={Huang, Xiaoming and Zhang, Yu-Jin},
  journal={IEEE Transactions on Image Processing},
  volume={26},
  number={9},
  pages={4243--4254},
  year={2017},
  publisher={IEEE}
}

@article{itti1998model,
  title={A model of saliency-based visual attention for rapid scene analysis},
  author={Itti, Laurent and Koch, Christof and Niebur, Ernst},
  journal={IEEE Transactions on pattern analysis and machine intelligence},
  volume={20},
  number={11},
  pages={1254--1259},
  year={1998},
  publisher={Ieee}
}
```

