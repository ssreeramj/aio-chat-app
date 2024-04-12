import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from model import LLM

llm_class = LLM()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_document(text_file):
    file_path = text_file.path
    file_format = file_path.split(".")[-1]

    if file_format == "txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

    elif file_format == "pdf":
        loader = PyMuPDFLoader(file_path)
        pdf_data = loader.load()
        text = "".join([doc.page_content for doc in pdf_data])

    return text


@cl.on_chat_start
async def on_chat_start():
    files = None
    model = llm_class.get_claude_v3_model()
    embeddings = llm_class.get_cohere_embedding()

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a text/pdf file to begin!",
            accept=["text/plain", "application/pdf"],
            max_size_mb=20,
        ).send()

    text_file = files[0]
    text = load_document(text_file=text_file)

    # TODO Split the text
    rec_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = rec_text_splitter.split_text(text=text)

    # TODO index the chunks
    vectorstore = await cl.make_async(Chroma.from_texts)(chunks, embeddings)
    retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context. \n\n<context>{context}</context>\n\nHere is the question, {question}"
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Processing completed: `{text_file.name}`. Chat with your document now!",
        disable_feedback=True,
    )
    await msg.send()

    cl.user_session.set("runnable", rag_chain)


@cl.on_message
async def on_message(message: cl.Message):
    rag_chain = cl.user_session.get("runnable")
    lf_handler = llm_class.get_langfuse_handler()

    msg = cl.Message(content="", disable_feedback=False)

    async for chunk in rag_chain.astream(
        message.content,
        config=RunnableConfig(
            callbacks=[
                cl.AsyncLangchainCallbackHandler(),
                lf_handler,
            ]
        ),
    ):
        await msg.stream_token(chunk)

    await msg.send()
