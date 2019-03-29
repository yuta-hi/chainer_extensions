import json
import os
import shutil
import tempfile
import warnings

import six

from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module


class LogReport(extension.Extension):

    """Trainer extension to output the accumulated results to a log file.

    This extension accumulates the observations of the trainer to
    :class:`~chainer.DictSummary` at a regular interval specified by a supplied
    trigger, and writes them into a log file in JSON format.

    There are two triggers to handle this extension. One is the trigger to
    invoke this extension, which is used to handle the timing of accumulating
    the results. It is set to ``1, 'iteration'`` by default. The other is the
    trigger to determine when to emit the result. When this trigger returns
    True, this extension appends the summary of accumulated values to the list
    of past summaries, and writes the list to the log file. Then, this
    extension makes a new fresh summary object which is used until the next
    time that the trigger fires.

    It also adds some entries to each result dictionary.

    - ``'epoch'`` and ``'iteration'`` are the epoch and iteration counts at the
      output, respectively.
    - ``'elapsed_time'`` is the elapsed time in seconds since the training
      begins. The value is taken from :attr:`Trainer.elapsed_time`.

    Args:
        keys (iterable of strs): Keys of values to accumulate. If this is None,
            all the values are accumulated and output to the log file.
        trigger: Trigger that decides when to aggregate the result and output
            the values. This is distinct from the trigger of this extension
            itself. If it is a tuple in the form ``<int>, 'epoch'`` or
            ``<int>, 'iteration'``, it is passed to :class:`IntervalTrigger`.
        log_json_name (str): Name of the log file for json format under the output
            directory. It can be a format string: the last result dictionary is
            passed for the formatting. For example, users can use '{iteration}'
            to separate the log files for different iterations. If the log name
            is None, it does not output the log to any file.
        log_csv_name (str): Name of the log file for csv format under the output
            directory.
    """

    def __init__(self, keys=None, trigger=(1, 'iteration'),
                 log_json_name='log', log_csv_name='log.csv'):
        self._keys = keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._log_csv_name = log_csv_name
        self._log_json_name = log_json_name
        self._log = []

        self._init_summary()

        self._is_initialized = False

    def _init_trigger(self, trainer):
        
        # NOTE: In case of fine-tuning (i.e., remuse is used), this line may raise minor bug
        # return trainer.updater.iteration == 1 

        if self._is_initialized:
            return False
        else:
            self._is_initialized = True
            return True

    def _write_json_log(self, path, _dict, indent=4):

        """ Append data in JSON format to the end of a JSON file.
        NOTE: Assumes file contains a JSON object (like a Python
        dict) ending in '}'.
        """

        with open(path, 'ab') as fp:
            fp.seek(0,2)              #Go to the end of file
            if fp.tell() == 0 :       #Check if file is empty
                new_ending = json.dumps(_dict, indent=indent)
                new_ending = new_ending.split('\n')
                new_ending = [' '*indent + x for x in new_ending]
                new_ending = '\n'.join(new_ending)
                new_ending = '[\n' + new_ending + '\n]'
                fp.write(new_ending.encode())

            else :
                fp.seek(-2,2)
                fp.truncate() #Remove the last two character

                new_ending = json.dumps(_dict, indent=indent)
                new_ending = new_ending.split('\n')
                new_ending = [' '*indent + x for x in new_ending]
                new_ending = '\n'.join(new_ending)
                new_ending = ',\n' + new_ending + '\n]'
                fp.write(new_ending.encode())


    def _update(self, data):
        entry = {key:data[key] if key in data else None for key in self._keys}

        # write CSV file
        with open(self._log_csv_name, 'a') as fp:
            fp.write(','.join(str(data[h]) if h in data.keys() else ','.join(' ') for h in self._keys) + '\n')

        # write JSON file
        self._write_json_log(self._log_json_name, entry)


    def __call__(self, trainer):

        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        if self._init_trigger(trainer):

            if keys is None:
                self._keys = ['epoch', 'iteration', 'elapsed_time']
                self._keys.extend(sorted(summary._summaries.keys()))
            else:
                self._keys = ['epoch', 'iteration', 'elapsed_time']
                for k in keys:
                    if k not in self._keys: self._keys.append(k)

            self._log_csv_name  = os.path.join(trainer.out, self._log_csv_name)
            self._log_json_name = os.path.join(trainer.out, self._log_json_name)

            os.makedirs(os.path.dirname(self._log_csv_name), exist_ok=True)
            os.makedirs(os.path.dirname(self._log_json_name), exist_ok=True)

            # write header
            with open(self._log_csv_name, 'w+') as fp:
                fp.write(','.join(self._keys) + '\n')

            # write serialized logs
            for stats_cpu in self._log:
                self._update(stats_cpu)


        if self._trigger(trainer):

            # output the result
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater
            stats_cpu['epoch'] = updater.epoch
            stats_cpu['iteration'] = updater.iteration
            stats_cpu['elapsed_time'] = trainer.elapsed_time

            self._log.append(stats_cpu)

            # write to the log file
            self._update(stats_cpu)

            # reset the summary for the next output
            self._init_summary()

    @property
    def log(self):
        """The current list of observation dictionaries."""
        return self._log

    def serialize(self, serializer):

        if hasattr(self._trigger, 'serialize'):
            self._trigger.serialize(serializer['_trigger'])

        try:
            self._summary.serialize(serializer['_summary'])
        except KeyError:
            warnings.warn('The statistics are not saved.')

        # Note that this serialization may lose some information of small
        # numerical differences.
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer('_log', log)
        else:
            log = serializer('_log', '')
            self._log = json.loads(log)

    def _init_summary(self):
        self._summary = reporter.DictSummary()
