"""Extract assistant text from worker Responses payloads."""

def answer_text(response):
    return '\n'.join(part.get('text', '') for item in response.get('output', [])
                     if item.get('type') == 'message' and item.get('role') == 'assistant'
                     for part in item.get('content', []) if part.get('type') == 'output_text')
