"""Project-local ALL 0.5.3 log-path compatibility; no training changes.

Preserve ExperimentWriter logging and SingleEnvExperiment training behavior.
Only the writer initializer changes the timestamp to a Windows-safe spelling.
"""
from datetime import datetime
from pathlib import Path

from all.experiments import SingleEnvExperiment
from all.experiments.writer import COMMIT_HASH, ExperimentWriter
from tensorboardX import SummaryWriter


class PortableExperimentWriter(ExperimentWriter):
    def __init__(self, experiment, agent_name, env_name, loss=True):
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S_%f')
        directory = Path('runs') / f'{agent_name} {COMMIT_HASH} {current_time}'
        (directory / env_name).mkdir(parents=True, exist_ok=False)
        self.env_name = env_name
        self.log_dir = str(directory)
        self._experiment = experiment
        self._loss = loss
        SummaryWriter.__init__(self, log_dir=self.log_dir)


class PortableSingleEnvExperiment(SingleEnvExperiment):
    def _make_writer(self, agent_name, env_name, write_loss):
        return PortableExperimentWriter(self, agent_name, env_name, loss=write_loss)
