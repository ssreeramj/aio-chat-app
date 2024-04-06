import chainlit as cl
from model import LLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from prompts.docqna_prompt import FULL_DOCUMENT_PROMPT

llm_class = LLM()


@cl.on_chat_start
async def on_chat_start():
    files = None
    model = llm_class.get_claude_v3_model()

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text file to begin!",
            accept=["text/plain"],
        ).send()

    text_file = files[0]

    with open(text_file.path, "r", encoding="utf-8") as f:
        text = f.read()

    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Processing completed: `{text_file.name}`. Chat with your document now!",
        disable_feedback=True,
    )
    await msg.send()

    inp_prompt_template = ChatPromptTemplate.from_template(
        FULL_DOCUMENT_PROMPT,
        partial_variables={"document": text[:100]},
    )

    qa_chain = inp_prompt_template | model | StrOutputParser()
    cl.user_session.set("runnable", qa_chain)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    lf_handler = llm_class.get_langfuse_handler()

    msg = cl.Message(content="", disable_feedback=False)

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(
            callbacks=[
                cl.LangchainCallbackHandler(),
                lf_handler,
            ]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()
