import hashlib
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import prediction_rl.data.frozen_dataset as frozen


@pytest.mark.parametrize('record_crlf,checkout_crlf',[(False,True),(True,False),(False,False),(True,True)])
def test_hash_stays_bound_to_git_with_only_checkout_line_endings(tmp_path,monkeypatch,record_crlf,checkout_crlf):
    blob=b'{\n  "seed": 0\n}\n'
    monkeypatch.setattr(frozen.subprocess,'check_output',lambda command:blob)
    current=blob.replace(b'\n',b'\r\n') if checkout_crlf else blob
    recorded=blob.replace(b'\n',b'\r\n') if record_crlf else blob
    (tmp_path/'config.json').write_bytes(current)
    frozen.verify_historical_text_checkout(tmp_path,'config.json',hashlib.sha256(recorded).hexdigest(),'1'*40)


@pytest.mark.parametrize('fault',['parameter','formatting','hash','escape','commit'])
def test_checkout_verification_does_not_relax_content_or_identity(tmp_path,monkeypatch,fault):
    blob=b'{\n  "seed": 0\n}\n';current=blob
    monkeypatch.setattr(frozen.subprocess,'check_output',lambda command:blob)
    if fault=='parameter':current=blob.replace(b'0',b'1')
    if fault=='formatting':current=b'{"seed":0}\n'  # Even equivalent reformatting is not accepted.
    (tmp_path/'config.json').write_bytes(current)
    name='../config.json' if fault=='escape' else 'config.json'
    sha='0'*64 if fault=='hash' else hashlib.sha256(blob).hexdigest()
    commit='main' if fault=='commit' else '1'*40
    with pytest.raises(ValueError):frozen.verify_historical_text_checkout(tmp_path,name,sha,commit)
