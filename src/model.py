import boto3
import os
from langchain_community.chat_models.bedrock import BedrockChat
from dotenv import load_dotenv, find_dotenv
from langfuse import Langfuse

load_dotenv(override=True)

class LLM():
    def __init__(self):
        pass

    def get_claude_v3_model(self):
        session = boto3.Session(
            aws_access_key_id=os.environ.get("aws_access_key_id"), 
            aws_secret_access_key=os.environ.get("aws_secret_access_key"),
            aws_session_token=os.environ.get("aws_session_token")
        )

        bedrock_client = session.client("bedrock-runtime", region_name='us-east-1')

        claude3 = BedrockChat(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            client=bedrock_client,
            model_kwargs={
                "max_tokens": 10_000,
                "temperature": 0.2,
            },
            streaming=True,
        )

        return claude3
    
    def get_langfuse_handler(self, user_id="test-user"):
        langfuse = Langfuse(
            secret_key=os.environ["LF_SECRET_KEY"],
            public_key=os.environ["LF_PUBLIC_KEY"],
            host=os.environ["LF_HOST_URL"],
        )

        # Create trace with tags
        trace = langfuse.trace(
            user_id=user_id,
        )
        handler = trace.get_langchain_handler()

        return handler
    

if __name__ == "__main__":
    llm_class = LLM()
    llm_model = llm_class.get_claude_v3_model()
    lf_handler = llm_class.get_langfuse_handler()

    response = llm_model.invoke("Hi there", config={"callbacks": [lf_handler]})
    print(f"{response.content=}")