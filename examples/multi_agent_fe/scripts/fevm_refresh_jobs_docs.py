"""Add official documentation coverage to the existing FEVM docs corpus.

Idempotent chunk IDs; existing unrelated documents are preserved. This script
updates knowledge data only, not the agent deployment or evaluation scenarios.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
import time
import requests
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementParameterListItem

SOURCES = {
    'serverless_jobs': ('https://docs.databricks.com/aws/en/jobs/run-serverless-jobs',
                        'serverless compute for workflows', 'agentops-jobs-'),
    'auto_loader': ('https://docs.databricks.com/aws/en/ingestion/cloud-object-storage/auto-loader/',
                    'auto loader', 'agentops-autoloader-'),
}
TABLE='fevm_catalog_naren.agentops.databricks_docs_chunks'
INDEX='fevm_catalog_naren.agentops_multi.platform_docs_index'


class ArticleText(HTMLParser):
    def __init__(self):
        super().__init__();self.active=False;self.parts=[]
    def handle_starttag(self,tag,attrs):
        if tag=='article':self.active=True
    def handle_endtag(self,tag):
        if tag=='article':self.active=False
    def handle_data(self,data):
        if self.active and data.strip():self.parts.append(data.strip())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',required=True)
    p.add_argument('--warehouse-id',required=True)
    p.add_argument('--index',default=INDEX)
    p.add_argument('--document', choices=sorted(SOURCES), default='serverless_jobs')
    args=p.parse_args()
    url, expected_text, prefix = SOURCES[args.document]
    response=requests.get(url,timeout=40);response.raise_for_status()
    parser=ArticleText();parser.feed(response.text)
    text='\n'.join(parser.parts)
    if len(text)<2000 or expected_text not in text.lower():
        raise RuntimeError('Official documentation did not contain the expected article')
    w=WorkspaceClient(profile=args.profile)
    chunks=[text[start:start+4500] for start in range(0,len(text),4000)]
    for index,chunk in enumerate(chunks):
        chunk_id=prefix+hashlib.sha256((url+':'+str(index)).encode()).hexdigest()[:24]
        result=w.statement_execution.execute_statement(warehouse_id=args.warehouse_id,
            statement=f'''MERGE INTO {TABLE} t USING (
              SELECT :id AS chunk_id, :text AS chunk_text, :url AS url,
              'official_docs' AS doc_type, :section AS section
            ) s ON t.chunk_id=s.chunk_id
            WHEN MATCHED THEN UPDATE SET *
            WHEN NOT MATCHED THEN INSERT *''',
            parameters=[StatementParameterListItem(name=k,value=v) for k,v in
                {'id':chunk_id,'text':chunk,'url':url,'section':args.document}.items()],wait_timeout='50s')
        while result.status.state.value in ('PENDING','RUNNING'):
            time.sleep(2);result=w.statement_execution.get_statement(result.statement_id)
        if result.status.state.value!='SUCCEEDED':
            raise RuntimeError(str(result.status.error))
    w.vector_search_indexes.sync_index(index_name=args.index)
    print(json.dumps({'source':url,'characters':len(text),'chunks_upserted':len(chunks),
                      'index':args.index,'sync_requested':True}))


if __name__=='__main__':main()
