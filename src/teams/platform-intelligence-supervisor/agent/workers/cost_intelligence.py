"""Connector to the existing Genie cost-intelligence space."""
import json
import asyncio
from datetime import timedelta


async def call_genie_as_user(space_id, task):
    """Use only this invocation's forwarded user credentials for Genie."""
    from databricks_ai_bridge import ModelServingUserCredentials
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.errors import PermissionDenied, Unauthenticated

    skipped = 'Skipping Genie - no user credentials or permissions for this Genie space.'
    try:
        workspace = WorkspaceClient(credentials_strategy=ModelServingUserCredentials())
    except Exception:
        return skipped
    try:
        return await call_genie(workspace, space_id, task)
    except (PermissionDenied, Unauthenticated):
        return skipped


async def call_genie(workspace, space_id, task):
    """Submit one question and let the SDK poll its conversation message."""
    async with asyncio.timeout(180):
        message = await asyncio.to_thread(
            workspace.genie.start_conversation_and_wait,
            space_id=space_id, content=task, timeout=timedelta(seconds=180),
        )
        parts = []
        for attachment in message.attachments or []:
            if attachment.text and attachment.text.content:
                parts.append(attachment.text.content)
            if attachment.query:
                if not attachment.attachment_id:
                    raise RuntimeError('Genie query attachment has no ID')
                query_result = await asyncio.to_thread(
                    workspace.genie.get_message_attachment_query_result,
                    space_id=space_id, conversation_id=message.conversation_id,
                    message_id=message.message_id, attachment_id=attachment.attachment_id,
                )
                statement = query_result.statement_response
                if statement is None or statement.result is None:
                    raise RuntimeError('Genie query returned no result')
                parts.append(json.dumps({
                    'description': attachment.query.description,
                    'query': attachment.query.query,
                    'manifest': statement.manifest.as_dict() if statement.manifest else None,
                    'result': statement.result.as_dict(),
                }))
        answer = '\n'.join(parts)
        if not answer.strip():
            raise RuntimeError('Genie returned no assistant answer')
        return answer
