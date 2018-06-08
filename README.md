# chainer_extensions

## Installation
- Install [chainer](https://github.com/chainer/chainer) 
- Install [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) 
```bash
pip install chainer
pip install tensorboardX
```

## ParameterStatisticsX
```python
from extensions import ParameterStatisticsX
trainer.extend(ParameterStatisticsX(model, prefix='model'))
```

<img src='figs/parameter_statistics_x.png' width="800px"/>

```bash
tensorboard --logdir=./result/.tensorboard
```

## Outputting images by the dot command
```python
from extensions import graphviz_dot
trainer.extend(graphviz_dot(file_name='*.dot'))
```