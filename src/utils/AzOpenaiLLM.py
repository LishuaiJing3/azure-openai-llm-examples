import os
import dotenv

dotenv.load_dotenv()


class AzureOpenAIChatModel:
    """Azure OpenAI model: GPT-4."""

    def __init__(
        self, base_url="BASE_URL", api_key="API_KEY", deployment="DEPLOYMENT_GPT_4"
    ):
        """Initializes the Azure OpenAI model."""

        from langchain.chat_models import AzureChatOpenAI

        # Get base URL, deployment name, and API key from .env file.
        """
        base_url = os.getenv(base_url)
        deployment = os.getenv(deployment)
        api_key = os.getenv(api_key)
        """
        # Create an AzureChatOpenAI instance.
        self.llm = AzureChatOpenAI(
            openai_api_base=base_url,
            openai_api_version="2023-05-15",
            deployment_name=deployment,
            openai_api_key=api_key,
            openai_api_type="azure",
        )

    def get_llm(self):
        """Returns the AzureChatOpenAI instance."""

        return self.llm


class AzureOpenAIModel:
    """Azure OpenAI model: GPT-3."""

    def __init__(
        self,
        base_url="BASE_URL",
        api_key="API_KEY",
        deployment="DEPLOYMENT_GPT_3",
        model_name="gpt-35-turbo",
    ):
        """Initializes the Azure OpenAI GPT 3 model."""

        from langchain.llms import AzureOpenAI

        """
        base_url = os.getenv(base_url)
        api_key = os.getenv(api_key)
        deployment = os.getenv(deployment)
        """
        self.llm = AzureOpenAI(
            openai_api_base=base_url,
            openai_api_version="2023-05-15",
            deployment_name=deployment,
            model=model_name,  ## you need to specify the model, otherwise it will use the default model
            openai_api_key=api_key,
            openai_api_type="azure",
        )

    def get_llm(self):
        """Returns the AzureChatOpenAI instance."""

        return self.llm
