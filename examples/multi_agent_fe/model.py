"""MLflow model-from-code entry point."""
import mlflow
from fe_multi_agent.agent import PlatformEngineeringAgent
from fe_multi_agent.usage import install_native_cost_processor

install_native_cost_processor()
mlflow.langchain.autolog()
mlflow.models.set_model(PlatformEngineeringAgent())
