import boto3
import os
from langchain_community.chat_models.bedrock import BedrockChat
from dotenv import load_dotenv

load_dotenv()

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
    

if __name__ == "__main__":
    llm_model = LLM().get_claude_v3_model()
    response = llm_model.invoke("Hi there")
    print(f"{response.content=}")