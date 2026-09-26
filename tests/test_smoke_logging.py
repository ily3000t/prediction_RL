import importlib.util
import io
from pathlib import Path

spec = importlib.util.spec_from_file_location('smoke_baseline', Path(__file__).resolve().parents[1] / 'tools/smoke_baseline.py')
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_logger_shutdown_after_log_context_closes():
    console, log = io.StringIO(), io.StringIO()
    tee = smoke.Tee(console, log)
    assert tee.write('during run') == 10
    assert log.getvalue() == console.getvalue()
    log.close()
    tee.write('after run')
    tee.close()
    assert not console.closed
    assert console.getvalue() == 'during runafter run'
