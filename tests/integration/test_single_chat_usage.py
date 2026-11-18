"""Execute the team chat target against a gateway-style cumulative usage stream."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS


def test_single_target_preserves_usage_for_shared_adapter_without_manual_spans():
    path = Path('src/teams/platform-engineering-test/agent/Agent.py')
    source = ast.parse(path.read_text())
    target_class = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'ChatCompletionsModel')
    namespace = {'mlflow': NS(start_span=lambda **kwargs: nullcontext('model-span'))}
    exec(compile(ast.Module(body=[target_class], type_ignores=[]), str(path), 'exec'), namespace)
    target = namespace['ChatCompletionsModel']()
    target.model = 'gateway-model'
    target.rag = NS(max_tokens=100, temperature=0)
    target._prepare = lambda *args: ([], {'source_urls': []})
    first = {'prompt_tokens': 20, 'completion_tokens': 1, 'total_tokens': 21}
    final = {'prompt_tokens': 20, 'completion_tokens': 30, 'total_tokens': 50}
    chunks = [{'choices': [], 'usage': first}, {'choices': [], 'usage': final}]
    stream = [NS(model_dump=lambda exclude_none, data=data: data.copy()) for data in chunks]
    target.client = NS(chat=NS(completions=NS(create=lambda **kwargs: nullcontext(stream))))
    output = list(target.predict_stream(messages=[], custom_inputs={}))
    assert len(output) == 3
    assert [item['usage'] for item in output if 'usage' in item] == [first, final]
    assert not hasattr(target, '_native_usage')
