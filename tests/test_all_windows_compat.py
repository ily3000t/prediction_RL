import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'RL-MPC-LaneMerging-master'))
from all.experiments import SingleEnvExperiment
from all_windows_compat import PortableExperimentWriter, PortableSingleEnvExperiment


def test_portable_writer_retains_logging(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    writer = PortableExperimentWriter(SimpleNamespace(frame=7, episode=1), 'ddpg', 'sumo-test')
    directory = Path(writer.log_dir)
    try:
        assert not any(c in directory.name for c in '<>:"/\\|?*')
        writer.add_loss('critic', 0.25)
        writer.add_summary('returns', 1.5, 0.5)
    finally:
        writer.close()
    assert (directory / 'sumo-test' / 'returns.csv').read_text().strip() == '7,1.5,0.5'
    assert list(directory.glob('events.out.tfevents.*'))


def test_training_loop_is_not_overridden():
    for name in ('train', 'test', '_run_training_episode', '_done'):
        assert getattr(PortableSingleEnvExperiment, name) is getattr(SingleEnvExperiment, name)
