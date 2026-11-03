# COMMAND ----------
# Standard library imports
import json
import time
import os
import yaml

# Third-party imports - Databricks
from databricks.vector_search.client import VectorSearchClient
# Third-party imports - LangChain
# from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatDatabricks
from langchain_community.embeddings import DatabricksEmbeddings
from utils.helpers import ContextRetriever, SQLQueryCorrection, SQLValidation, SQLQueryProcessor, RouterResponse, SQLColumnHallucinationFixer, build_table_column_embeddings
from utils.logger import get_logger, initialize_logger_context

# Third-party imports - MLflow
import mlflow
import mlflow.deployments
import mlflow.pyfunc
from mlflow.entities import SpanType
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from framework.agent_base import emit_standard_trace_metrics, invoke_with_usage_tracking
mlflow.langchain.autolog()

# AgentOPSBase is required — fail fast at import time if the wheel is missing
# rather than silently degrading to PythonModel (which has an incompatible
# predict signature and will break serving with a cryptic runtime error).
from framework.agent_base import AgentOPSBase
from framework.inference import extract_request_text, extract_response_text
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

_ChatModelBase = AgentOPSBase

# Global constants
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
UNKNOWN = "unknown"

logger = get_logger(__name__)

# ----------------------------------------------------------------------------------
# NOTE
# ----------------------------------------------------------------------------------
# HANDLING SESSION MANAGEMENT REQUIRES A NEW TABLE TO BE CREATED IN OUR SCHEMA
# CREATE TABLE IF NOT EXISTS mc_custom_analytics_solutions.mc_cac_qa.mc_cac_nonfi_v1qa_bha_shared2 (
# conversation_id STRING,
# model_input STRING,
# messages STRING,
# conversation_state STRING,
# updated_at TIMESTAMP)
# TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true');
# ----------------------------------------------------------------------------------
# model_config = mlflow.models.ModelConfig(development_config="config.yml")
class ChatModel(_ChatModelBase):  # type: ignore[misc]

    def __init__(self):
        # Initialize all attributes to None
        self.model_config = None
        self.config = None
        self.catalog_name = None
        self.model_schema_name = None
        self.gold_schema = None
        self.docs_source_table = None
        self.tables = None
        self.vector_search_context_tbl = None
        self.vector_search_entity_tbl = None
        self.vector_search_context_index = None
        self.vector_search_entity_registry_index = None
        self.vector_search_endpoint_name = None
        self.workspace_url = None
        self.llm_model = None
        self.embedding = None
        self.model_name = None
        self.max_tokens = None
        self.temperature = None
        self.warehouse_id = None
        self.prompts_template = None

        # Initialize prompt strings
        self.greeting_prompt_str = None
        self.relevance_prompt_str = None
        self.classification_prompt_str = None
        self.lookup_prompt_str = None
        self.slot_prompt_str = None
        self.clarification_prompt_str = None
        self.rewrite_prompt_str = None
        self.intent_prompt_str = None
        self.component_prompt_str = None
        self.table_selector_prompt_str = None
        self.schema_linker_prompt_str = None
        self.sql_prompt_str = None
        self.sql_validation_prompt_str = None
        self.sql_query_correction_str = None
        self.error_message_str = None
        self.sql_check_prompt_str = None

        # Initialize vector search and model attributes
        self.vs_index_fullname = None
        self.entity_vs_index_fullname = None
        self.vsc = None
        self.embedding_model = None
        self.delta_index = None
        self.delta_index_entity = None
        self.llm = None

        # Initialize prompt templates
        self.greeting_prompt = None
        self.relevance_prompt = None
        self.classification_prompt = None
        self.lookup_prompt = None
        self.slots_prompt = None
        self.clarification_prompt = None
        self.rewrite_prompt = None
        self.intent_prompt = None
        self.component_prompt = None
        self.table_selector_prompt = None
        self.schema_linker_prompt = None
        self.sql_prompt = None
        self.sql_validation_prompt = None
        self.sql_correction_prompt = None
        self.error_message_prompt = None
        self.sql_check_prompt = None
        # Initialize SQL query processor
        self.sql_query_processor = None
        self._request_context = {}
        self._default_trace_tags = {
            "agentops.agent_name": "cca_agent",
            "agentops.agent_version": "0.0.0",
            "agentops.framework_version": "0.1.0",
            "agentops.environment": os.environ.get("AGENTOPS_ENVIRONMENT", "unknown"),
        }

    def load_context(self, context):
        # Initialise AgentOPSBase — wires PreLLMGuardrails, PostLLMGuardrails,
        # SessionStore, default trace tags and token infrastructure before we
        # load the team-specific LLM / VS clients below.
        AgentOPSBase.__init__(self, context.model_config or {})

        # get values for variables stored in thr yaml files
        self.model_config = context.model_config
        self.config = self.model_config.get("parameters")
        self.catalog_name = self.config['catalog_name']
        self.model_schema_name = self.config['model_schema_name']
        self.gold_schema = self.config['gold_schema_name']

        self.docs_source_table = self.config['docs_source_table']
        self.tables = self.config['tables']
        
        self.context_index_name = self.config.get("vector_search_context_index", "context_docs_nonfi_index")
        self.entity_index_name = self.config.get("vector_search_entity_registry_index", "entity_nonfi_index")
        self.vector_search_entity_tbl = self.config.get("vector_search_entity_tbl", "entity_registry_nonfi_tbl")
        self.vector_search_context_tbl = self.config.get("vector_search_context_tbl", "context_docs_nonfi_tbl")

        
        self.vector_search_endpoint_name = self.config['vector_search_endpoint_name']
        self.workspace_url = self.config['workspace_url']
        self.llm_model = self.config['llm_model']
        self.embedding = self.config['embedding']
        self.model_name = self.config["model_name"]
        self.max_tokens = self.config['max_tokens']
        self.temperature = self.config['temperature']
        self.warehouse_id = self.config['sql_warehouse_id']

        self._default_trace_tags.update({
            "agentops.agent_name": self.model_name or self._default_trace_tags["agentops.agent_name"],
            "agentops.environment": self.config.get("environment", self._default_trace_tags["agentops.environment"]),
        })

        # load prompts from prompt_templates.yml
        with open(os.path.join(os.path.dirname(__file__), "../configs/prompt_templates.yml")) as f:
            self.prompts_template = yaml.safe_load(f)

        # initializing prompts
        self.greeting_prompt_str = self.prompts_template.get("greeting_prompt_str")
        self.relevance_prompt_str = self.prompts_template.get("relevance_prompt_str")
        self.classification_prompt_str = self.prompts_template.get("classification_prompt_str")
        self.lookup_prompt_str = self.prompts_template.get("lookup_prompt_str")
        self.slot_prompt_str = self.prompts_template.get("slot_prompt_str")
        self.clarification_prompt_str = self.prompts_template.get("clarification_prompt_str")
        self.rewrite_prompt_str = self.prompts_template.get("rewrite_prompt_str")
        self.intent_prompt_str = self.prompts_template.get("intent_prompt_str")
        self.component_prompt_str = self.prompts_template.get("component_prompt_str")
        self.table_selector_prompt_str = self.prompts_template.get("table_selector_prompt_str")
        self.schema_linker_prompt_str = self.prompts_template.get("schema_linker_prompt_str")
        self.sql_prompt_str = self.prompts_template.get("sql_prompt_str")
        self.sql_validation_prompt_str = self.prompts_template.get("sql_validation_prompt_str")
        self.sql_query_correction_str = self.prompts_template.get("sql_query_correction_str")
        self.error_message_str = self.prompts_template.get("error_message_str")
        self.sql_check_prompt_str = self.prompts_template.get("sql_check_prompt_str")

        # set up vs index names
        self.vs_index_fullname = f"{self.catalog_name}.{self.model_schema_name}.{self.context_index_name}"
        self.entity_vs_index_fullname = f"{self.catalog_name}.{self.model_schema_name}.{self.entity_index_name}"
        self.vsc = VectorSearchClient(workspace_url=self.workspace_url)
        self.embedding_model = DatabricksEmbeddings(endpoint=self.embedding, databricks_host=self.workspace_url)
        self.delta_index = self.vsc.get_index(self.vector_search_endpoint_name, self.vs_index_fullname)
        self.delta_index_entity = self.vsc.get_index(self.vector_search_endpoint_name, self.entity_vs_index_fullname)
        
        # llm initialization
        self.llm = ChatDatabricks(endpoint=self.llm_model, max_tokens=self.max_tokens, temperature=self.temperature, databricks_host=self.workspace_url, disable_streaming=True, streaming=False)
        # GET PROMPTS
        self.greeting_prompt = PromptTemplate(input_variables=["user_input"], template=self.greeting_prompt_str)
        self.relevance_prompt = PromptTemplate(input_variables=["chat_history", "user_input","awaiting_clarification"], template=self.relevance_prompt_str)
        self.classification_prompt = PromptTemplate(input_variables=["chat_history", "user_input","awaiting_clarification"], template=self.classification_prompt_str)
        self.lookup_prompt = PromptTemplate(input_variables=["user_input", "context", "chat_history"], template=self.lookup_prompt_str)
        self.slots_prompt = PromptTemplate(input_variables=["chat_history", "user_input"], template=self.slot_prompt_str)
        self.clarification_prompt = PromptTemplate(input_variables=["slots_json", "user_input"], template=self.clarification_prompt_str)
        self.rewrite_prompt = PromptTemplate(input_variables=["chat_history", "user_input", "slots_json"], template=self.rewrite_prompt_str)
        self.intent_prompt = PromptTemplate(input_variables=["user_input", "chat_history"], template=self.intent_prompt_str)
        self.component_prompt = PromptTemplate(input_variables=["user_input", "chat_history"], template=self.component_prompt_str)
        self.table_selector_prompt = PromptTemplate(input_variables=["user_input", "components"], template=self.table_selector_prompt_str)
        self.schema_linker_prompt = PromptTemplate(input_variables=["user_input", "components", "context", "tbl_select"], template=self.schema_linker_prompt_str)
        self.sql_prompt = PromptTemplate(input_variables=["user_input", "context", "chat_history", "intent", "components", "tbl_select", "sch_link"],template=self.sql_prompt_str)
        self.sql_validation_prompt = PromptTemplate(input_variables=["sql_query"], template=self.sql_validation_prompt_str)
        self.sql_correction_prompt = PromptTemplate(input_variables=["sql_query"], template=self.sql_query_correction_str)
        self.error_message_prompt = PromptTemplate(input_variables=["error_message"], template=self.error_message_str)
        self.sql_check_prompt = PromptTemplate(input_variables=["input"],template=self.sql_check_prompt_str)

        # Class instance
        self.sql_query_processor = SQLQueryProcessor(workspace_url=self.workspace_url, warehouse_id=self.warehouse_id, catalog_name=self.catalog_name, schema=self.gold_schema)
        # warm up cluster
        self.sql_query_processor.execute_query("SELECT 1")

        self.table_col_embeddings = build_table_column_embeddings(
            vector_index=self.delta_index, 
            embedding_model=self.embedding_model,
        )

        self.sql_hallucinated_col_fixer = SQLColumnHallucinationFixer(
            table_col_embeddings=self.table_col_embeddings, 
            embedding_model=self.embedding_model
        )

        self.greeting_chain = RunnablePassthrough.assign(greeting=self.greeting_prompt | self.llm | StrOutputParser())
        self.relevance_chain = RunnablePassthrough.assign(is_relevant=self.relevance_prompt | self.llm | StrOutputParser())
        self.classification_chain = RunnablePassthrough.assign(classification=self.classification_prompt | self.llm | StrOutputParser())
        self.lookup_chain = RunnablePassthrough.assign(lookup=self.lookup_prompt | self.llm | StrOutputParser())
        self.slot_extract_chain = RunnablePassthrough.assign(slots=self.slots_prompt | self.llm | StrOutputParser())
        self.clarification_chain = RunnablePassthrough.assign(clarification_msg=self.clarification_prompt | self.llm | StrOutputParser())
        self.rewrite_chain = RunnablePassthrough.assign(rewritten_question=self.rewrite_prompt | self.llm | StrOutputParser())
        self.intent_chain = RunnablePassthrough.assign(intent=self.intent_prompt | self.llm | StrOutputParser())
        self.component_chain = RunnablePassthrough.assign(components=self.component_prompt | self.llm | StrOutputParser())
        self.table_selector_chain = (RunnablePassthrough.assign(tbl_select=self.table_selector_prompt | self.llm | StrOutputParser()))
        self.schema_linker_chain = (RunnablePassthrough.assign(sch_link=self.schema_linker_prompt | self.llm | StrOutputParser()))
        self.sql_gen_chain = RunnablePassthrough.assign(sql_query=self.sql_prompt | self.llm | StrOutputParser())
        self.sql_validation_chain = RunnablePassthrough.assign(validation=self.sql_validation_prompt | self.llm | StrOutputParser())
        self.sql_correction_chain = RunnablePassthrough.assign(corrected_sql_query=self.sql_correction_prompt | self.llm | StrOutputParser())
        self.error_message_chain = RunnablePassthrough.assign(summarized_error_message=self.error_message_prompt | self.llm | StrOutputParser())
        self.sql_check_chain = RunnablePassthrough.assign(SQL_flag=self.sql_check_prompt | self.llm | StrOutputParser())

        self.chains = {
            "greeting_chain": self.greeting_chain,
            "relevance_chain": self.relevance_chain,
            "classification_chain": self.classification_chain,
            "lookup_chain": self.lookup_chain,
            "slot_extract_chain": self.slot_extract_chain,
            "clarification_chain": self.clarification_chain,
            "rewrite_chain": self.rewrite_chain,
            "intent_chain": self.intent_chain,
            "component_chain": self.component_chain,
            "table_selector_chain": self.table_selector_chain,
            "schema_linker_chain": self.schema_linker_chain,
            "sql_gen_chain": self.sql_gen_chain,
            "sql_validation_chain": self.sql_validation_chain,
            "sql_correction_chain": self.sql_correction_chain,
            "error_message_chain": self.error_message_chain,
            "sql_check_chain": self.sql_check_chain
        }

    @staticmethod
    def escape_sql(text):
        return text.replace("'", "''") if text else ""

    @staticmethod
    def _get_current_user_info():
        try:
            from flask import request as flask_request
            user_email = flask_request.headers.get(
                "X-Databricks-Internal-Ai-Gateway-Requester-Email", UNKNOWN
            )
        except Exception:
            user_email = UNKNOWN
        return {"user_name": user_email}
    
    @staticmethod
    def _get_request_id():
        """Extract X-Request-Id from the Flask request headers."""
        try:
            from flask import request as flask_request
            return flask_request.headers.get("X-Request-Id", UNKNOWN)
        except Exception:
            return UNKNOWN
        

    def build_memory_from_messages(self, messages, k=10):
        memory = ConversationBufferWindowMemory(k=k, memory_key="chat_history", input_key="user_input", return_messages=False)
        for msg in messages[:-1]:
            role = msg.get("role")
            content = msg.get("content")
            if not content:
                continue
            if role == ROLE_USER:
                memory.chat_memory.add_user_message(content)
            elif role == ROLE_ASSISTANT:
                memory.chat_memory.add_ai_message(content)
        return memory

    def get_last_assistant_message(self, messages):
        for msg in reversed(messages):
            if msg.get("role") == ROLE_ASSISTANT and msg.get("content"):
                return msg["content"]
        return None

    def save_session(self, conversation_id, model_input, messages, conversation_state):

        conversation_id = json.dumps(conversation_id)
        memory_json = json.dumps(messages)
        state_json = json.dumps(conversation_state)
        model_input_json = model_input.to_json(orient="records")
        memory_json = self.escape_sql(memory_json)
        state_json = self.escape_sql(state_json)
        model_input_json = self.escape_sql(model_input_json)
        try:
            query = f""" INSERT INTO {self.catalog_name}.{self.model_schema_name}.{self.model_name}_shared2
                        (conversation_id,model_input, messages, conversation_state, updated_at)  VALUES (
                                '{conversation_id}',
                                '{model_input_json}',
                                '{memory_json}',
                                '{state_json}',
                                current_timestamp() )"""
            self.sql_query_processor.execute_query(query)

        except Exception as e:
            logger.error("Session save failed", extra={"error": str(e)})

    @mlflow.trace(span_type="func")
    def fallback_chain(self):
        logger.info("Fallback chain invoked")
        return "I cannot answer your question. Please ask a different question."

    @mlflow.trace(span_type="func")
    def classify(self, user_input, chat_history, top_k, conversation_state):
        logger.info("Classification step started")
        classification_result = invoke_with_usage_tracking(
            self.chains["classification_chain"],
            {"user_input": user_input, "chat_history": chat_history,
             "awaiting_clarification": conversation_state["awaiting_clarification"]},
            self._request_context,
            model_name=self.llm_model,
        )
        label = classification_result.get("classification", "").strip().lower()
        logger.debug("Classification chain result: %s", classification_result)
        if label == "sql_query":
            req_context = ["SchLink", "sql"]
        else:
            req_context = ["lookup"]
        context_retriever = ContextRetriever(self.delta_index, self.delta_index_entity)
        context = context_retriever.retrieve(user_input, top_k, req_context)
        logger.info("Classification step completed", extra={"label": label})
        return label, context
    
    @mlflow.trace(span_type="func")
    def lookup_func(self, user_input, chat_history, context):
        logger.info("Lookup step started")
        lookup_result = invoke_with_usage_tracking(
            self.chains["lookup_chain"],
            {"user_input": user_input, "chat_history": chat_history, "context": context[0]},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Lookup chain result: %s", lookup_result)
        logger.info("Lookup step completed")
        return lookup_result.get("lookup", "")
    
    @mlflow.trace(span_type="func")
    def sql_func(self, user_input, chat_history, context, tables, conversation_state):
        
        logger.info("SQL function step started")
        chains = self.chains
        required_slots = ["entity", "metric", "time_period"]
        slots_result = invoke_with_usage_tracking(
            chains["slot_extract_chain"],
            {"user_input": user_input, "chat_history": chat_history},
            self._request_context,
            model_name=self.llm_model,
        )
        slots_out=slots_result.get("slots", "").strip()
        logger.debug("Slot extraction chain result: %s", slots_out)
        logger.info("Slot extraction completed", extra={"slots_output": slots_out})
        try:
            slots = json.loads(slots_out)
        except Exception as e:
            slots = {}
            for ky in required_slots:
                slots[ky] = None
            slots["is_complete"] = False

        for key in required_slots:
            if isinstance(slots.get(key), str) and slots[key].strip().lower() == "null":
                slots[key] = None

        is_complete = all(slots.get(k) not in (None, "") for k in required_slots)
        if not is_complete:
            logger.info("Branch: INCOMPLETE > clarification")
            clarification_ques = invoke_with_usage_tracking(
                chains["clarification_chain"],
                {"slots_json": slots_out, "user_input": user_input},
                self._request_context,
                model_name=self.llm_model,
            )
            clar_msg = clarification_ques.get("clarification_msg", "").strip()
            logger.debug("Clarification chain result: %s", clarification_ques)
            conversation_state["awaiting_clarification"] = True
            return RouterResponse(type="clarification", content=f"Clarification Required. {clar_msg}")

        rewrite_result = invoke_with_usage_tracking(
            chains["rewrite_chain"],
            {"user_input": user_input, "chat_history": chat_history, "slots_json": slots_out},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Rewrite chain result: %s", rewrite_result)
        rewritten = rewrite_result.get("rewritten_question", "").strip()

        if not rewritten:
            return RouterResponse(type="rewrite", content="I couldn't understand your request. Could you rephrase?")
        
        intent_result = invoke_with_usage_tracking(
            chains["intent_chain"],
            {"user_input": rewritten, "chat_history": chat_history},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Intent chain result: %s", intent_result)
        intent = intent_result.get("intent", "")

        component_result = invoke_with_usage_tracking(
            chains["component_chain"],
            {"user_input": rewritten, "chat_history": chat_history},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Component chain result: %s", component_result)
        components = component_result.get("components", "")

        logger.info("Intent extraction completed", extra={"intent": intent})
        logger.info("Component extraction completed", extra={"components": components})

        tbl_select = invoke_with_usage_tracking(
            chains["table_selector_chain"],
            {"user_input": rewritten, "chat_history": chat_history, "components": components},
            self._request_context,
            model_name=self.llm_model,
        ).get("tbl_select", "").strip().lower()

        # Validate the table name
        #valid_tables = ["agg_merch_monthly_mv", "agg_merch_quarterly_mv", "non_agg_merch_monthly_mv", "non_agg_merch_quarterly_mv"]
        valid_tables = tables.split(",")
        if tbl_select not in valid_tables:
            print(f"Warning: Invalid table name returned: {tbl_select}. Falling back to default.")
            tbl_select = "agg_merch_monthly_mv"  # Or handle appropriately
        logger.info("Table extraction completed", extra={"tbl_select": tbl_select})
        # Retrieve schema-linking-specific context
        #context_retriever = ContextRetriever(self.delta_index, self.delta_index_entity)
        sch_link_context = context[0]
        sch_link = invoke_with_usage_tracking(
            chains["schema_linker_chain"],
            {"user_input": rewritten, "chat_history": chat_history, "components": components, "context": sch_link_context, "tbl_select": tbl_select},
            self._request_context,
            model_name=self.llm_model,
        ).get("sch_link", "").strip()
        logger.info("Schema extraction completed", extra={"sch_link": sch_link})

        sql_gen_start = time.time()
        sql_context = context[1]
        sql_gen_result = invoke_with_usage_tracking(
            chains["sql_gen_chain"],
            {"user_input": rewritten, "chat_history": chat_history, "context": sql_context, "intent": intent, "components": components, "tbl_select": tbl_select, "sch_link": sch_link},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("SQL generation chain result: %s", sql_gen_result)
        sql_query = sql_gen_result.get("sql_query", "")
        if sql_query.upper() == "N/A":
            return RouterResponse(type="error", content="I cannot generate a SQL query from the schema provided.")

        sql_validation = SQLValidation(chains["sql_validation_chain"],chains["sql_check_chain"])
        sql_check = sql_validation.is_sql_check({"input": sql_query})
        logger.debug("SQL check result: %s", sql_check)

        if sql_check != "SQL":
            return RouterResponse(type="SQL Generation Error", content=f""" Could not generate an SQL. {sql_query}""")

        validation_ = sql_validation.validate(sql_query=sql_query)
        logger.debug("SQL validation result: %s", validation_)
        if not validation_:
            return RouterResponse(type="validation", content="Your question is not a permitted query. Please ask read-only questions.")

        sql_query_correction = SQLQueryCorrection(chains["sql_correction_chain"])
        query_sql = sql_query_correction.query_correction(sql_query)
        logger.debug("SQL correction result: %s", query_sql)
     
        conversation_state["awaiting_clarification"] = False
        # query_sql = self.sql_hallucinated_col_fixer.fix(query_sql)
        logger.info("SQL query generation completed", extra={"duration_seconds": round(time.time() - sql_gen_start, 2)})
        return RouterResponse(type="sql", content=query_sql)
        # Commenting out the is_valid condition since it fails for the case with select cp union select cnp
        #if self.sql_hallucinated_col_fixer.is_valid_select_sql(query_sql):
            #query_sql = self.sql_hallucinated_col_fixer.fix(query_sql)
            #return RouterResponse(type="sql", content=query_sql)
        #else:
            #return RouterResponse(type="validation", content="Your question is not a permitted select query.")

    @mlflow.trace(span_type="ROUTER2")
    def router(self, user_input, chat_history, label, context, tables, conversation_state):

        logger.info("Router step started", extra={"label": label})
        if label == "lookup":
            return self.lookup_func(user_input, chat_history, context)
        elif label == "sql_query":
            sql_func_op = self.sql_func(user_input, chat_history, context, tables, conversation_state)
            if sql_func_op.is_sql():
                sql_query = sql_func_op.content
                res = self.sql_query_processor.process_result(error_message_chain=self.chains["error_message_chain"], query=sql_query)
                return f"```sql\n{sql_query}\n```\n\n**Output:**\n\n{res}"
            return sql_func_op.content
        else:
            return self.fallback_chain()

    @mlflow.trace(span_type="ROUTER1")
    def chatbot(self, user_input, chat_history, conversation_state):
        logger.info("Chatbot chain started", extra={"user_input": user_input})
        chains = self.chains

        # 1. Greeting check
        greeting_result = invoke_with_usage_tracking(
            chains["greeting_chain"],
            {"user_input": user_input},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Greeting chain result: %s", greeting_result)
        greeting = greeting_result.get("greeting", "").strip().lower()
        if greeting == "yes":
            return "Hello! How can I help you today?"
        # 2. Relevance check
        #if not conversation_state.get("awaiting_clarification", False):
        relevance_result = invoke_with_usage_tracking(
            chains["relevance_chain"],
            {"user_input": user_input, "chat_history": chat_history,
             "awaiting_clarification": conversation_state["awaiting_clarification"]},
            self._request_context,
            model_name=self.llm_model,
        )
        logger.debug("Relevance chain result: %s", relevance_result)
        is_relevant = relevance_result.get("is_relevant", "").strip().lower()
        if is_relevant == "no":
            response = "Sorry, your question is out of scope. Please rephrase."
            return response
        #else:
            #print("Skipping relevance because we are awaiting clarification")

        # 3. Classification and context retrieval
        label, context = self.classify(user_input, chat_history, 5, conversation_state)

        return self.router(user_input, chat_history, label, context, self.tables, conversation_state)

    @mlflow.trace(span_type="CHAIN")
    def _process_request(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        """Core chat logic called by AgentOPSBase.predict().

        AgentOPSBase owns: pre/post-LLM guardrail checks, token tracking via
        emit_standard_trace_metrics() (which reads self._request_context populated
        by invoke_with_usage_tracking calls below), session load/save, and all
        agentops.* trace tags. This method implements domain-specific chat logic only.
        """
        history_messages = []
        for item in request.input:
            role = getattr(item, "role", None)
            if role not in (ROLE_USER, ROLE_ASSISTANT):
                continue
            if isinstance(item.content, str):
                text = item.content
            elif role == ROLE_USER:
                text = extract_request_text({"input": [item.model_dump(exclude_none=True)]})
            else:
                text = extract_response_text({"output": [item.model_dump(exclude_none=True)]})
            history_messages.append({"role": role, "content": text})
        user_input = extract_request_text(request)

        # Structured logging context (Python logging — separate from MLflow tracing)
        user_info = self._get_current_user_info()
        initialize_logger_context(
            request_id=self._get_request_id(),
            user_email=user_info.get("user_name", UNKNOWN),
        )
        logger.info("_process_request started")
        logger.debug("_process_request — user: %s, input: %s",
                     user_info.get("user_name"), user_input[:120])

        conversation_state = {"awaiting_clarification": False}
        memory = self.build_memory_from_messages(history_messages, k=10)
        last_ai = self.get_last_assistant_message(history_messages)
        conversation_state["awaiting_clarification"] = bool(
            last_ai and last_ai.startswith("Clarification Required")
        )
        chat_history = memory.load_memory_variables({}).get("chat_history", "")

        response_text = self.chatbot(user_input, chat_history, conversation_state)

        logger.info("_process_request completed")
        return ResponsesAgentResponse(
            output=[self.create_text_output_item(text=response_text or "", id=self._get_request_id())]
        )


mlflow.models.set_model(model=ChatModel())
