import json
import pytest
from fe_multi_agent.prompts import save_prompt_artifacts,load_prompts,TEMPLATES


def test_prompt_artifact_roundtrip_and_corruption_detection(tmp_path):
    path=tmp_path/'prompts.json'
    save_prompt_artifacts(path,TEMPLATES)
    assert load_prompts(path)==TEMPLATES
    manifest=json.loads(path.read_text())
    manifest['prompts']['supervisor']['template']='corrupted content'
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError,match='digest mismatch'):
        load_prompts(path)
