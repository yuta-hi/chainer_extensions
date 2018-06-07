import json
import os
import shutil
import tempfile
import six
import numpy

import chainer
from chainer.backends import cuda
from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module

from tensorboardX import SummaryWriter

class ParameterStatisticsX(extension.Extension):

    default_name = 'parameter_statistics'
    priority = extension.PRIORITY_WRITER

    # prefix ends with a '/' and param_name is preceded by a '/'
    scalar_report_key_template = ('{prefix}{link_name}{param_name}/{attr_name}/'
                           '{function_name}')

    histgram_report_key_template = ('{prefix}{link_name}{param_name}/{attr_name}')

    default_statistics = {
        'mean': lambda x: cuda.get_array_module(x).mean(x),
        'std': lambda x: cuda.get_array_module(x).std(x),
        'min': lambda x: cuda.get_array_module(x).min(x),
        'max': lambda x: cuda.get_array_module(x).max(x),
        'zeros': lambda x: cuda.get_array_module(x).count_nonzero(x == 0),
        'percentile': lambda x: cuda.get_array_module(x).percentile(
            x, (0.13, 2.28, 15.87, 50, 84.13, 97.72, 99.87))
    }

    def __init__(self, links, statistics=default_statistics,
                 report_params=True, report_grads=True, prefix=None,
                 histogram=True, trigger=(1, 'epoch'), skip_nan_params=False, log_dir=None,
                 ):

        if not isinstance(links, (list, tuple)):
            links = links,
        self._links = links

        if statistics is None:
            statistics = {}
        self._statistics = statistics

        attrs = []
        if report_params:
            attrs.append('data')
        if report_grads:
            attrs.append('grad')
        self._attrs = attrs

        self._prefix = prefix
        self._histogram = histogram
        self._trigger = trigger_module.get_trigger(trigger)
        self._skip_nan_params = skip_nan_params
        self._logger = SummaryWriter(log_dir=log_dir)

    def __call__(self, trainer):
        """Execute the statistics extension.
        Collect statistics for the current state of parameters.
        Note that this method will merely update its statistic summary, unless
        the internal trigger is fired. If the trigger is fired, the summary
        will also be reported and then reset for the next accumulation.
        Args:
            trainer (~chainer.training.Trainer): Associated trainer that
                invoked this extension.
        """

        if self._trigger(trainer):

            statistics = {}

            for link in self._links:
                link_name = getattr(link, 'name', 'None')
                for param_name, param in link.namedparams():
                    for attr_name in self._attrs:
                        for function_name, function in \
                                six.iteritems(self._statistics):
                            # Get parameters as a flattend one-dimensional array
                            # since the statistics function should make no
                            # assumption about the axes
                            params = getattr(param, attr_name).ravel()

                            # save as scalar
                            if (self._skip_nan_params
                                and (cuda.get_array_module(params).isnan(params)
                                    .any())):
                                value = numpy.nan
                            else:
                                value = function(params)
                            key = self.scalar_report_key_template.format(
                                prefix=self._prefix + '/' if self._prefix else '',
                                link_name=link_name,
                                param_name=param_name,
                                attr_name=attr_name,
                                function_name=function_name
                            )

                            if (isinstance(value, chainer.get_array_types())
                                    and value.size > 1):
                                # Append integer indices to the keys if the
                                # statistic function return multiple values
                                statistics.update({'{}/{}'.format(key, i): v for
                                                i, v in enumerate(value)})
                            else:
                                statistics[key] = value

                            # save as histogram
                            if self._histogram:
                                key = self.histgram_report_key_template.format(
                                    prefix=self._prefix + '/' if self._prefix else '',
                                    link_name=link_name,
                                    param_name=param_name,
                                    attr_name=attr_name,
                                )
                                
                                self._logger.add_histogram(key, cuda.to_cpu(params), trainer.updater.iteration)

            for k, v in statistics.items():
                self._logger.add_scalar(k, v, trainer.updater.iteration)