import os
import warnings
import subprocess
import glob
from chainer.training import extension
from chainer import configuration

def graphviz_dot(file_name='*.dot'):

    def trigger(trainer):
        return trainer.updater.iteration == 1

    _file_name = file_name

    def initializer(_):pass

    @extension.make_extension(trigger=trigger, initializer=initializer)
    def graphviz_dot(trainer):

        file_name = os.path.join(trainer.out, _file_name)
        file_list = glob.glob(file_name)
        try:
            for f in file_list:
                out_name,_ = os.path.splitext(f)
                out_name += '.png'
                subprocess.call('dot -T png %s -o %s' % (f, out_name), shell=True)
        except:
            warnings.warn('please install graphviz and set your environment.')

    return graphviz_dot


