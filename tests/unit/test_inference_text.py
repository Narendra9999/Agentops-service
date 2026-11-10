import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from framework.inference import extract_request_text, extract_response_text


def test_request_uses_latest_user_turn_and_text_parts_only():
    request={'input':[{'role':'user','content':'Earlier question'}, {'role':'assistant','content':'Earlier answer'},
        {'role':'user','content':[{'type':'input_text','text':'Current question'},{'type':'input_image','image_url':'private'}]}]}
    assert extract_request_text(request)=='Current question'
    assert extract_request_text({'messages':[{'role':'user','content':'Legacy question'}]})=='Legacy question'


def test_response_excludes_tool_and_reasoning_items_and_supports_model_objects():
    response={'output':[{'type':'function_call','name':'private','arguments':'secret'},
        {'type':'reasoning','summary':[{'text':'private reasoning'}]},
        {'role':'assistant','content':[{'type':'output_text','text':'Public answer'}]}]}
    assert extract_response_text(response)=='Public answer'
    model=SimpleNamespace(model_dump=lambda **kwargs:response)
    assert extract_response_text(model)=='Public answer'
    assert extract_response_text({'choices':[{'message':{'role':'assistant','content':'Legacy answer'}}]})=='Legacy answer'
    assert extract_response_text({'response':response})=='Public answer'
