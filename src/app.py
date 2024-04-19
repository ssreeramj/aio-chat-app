from typing import List
import chainlit as cl
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable.config import RunnableConfig
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableParallel
from langchain_text_splitters import RecursiveCharacterTextSplitter
from operator import itemgetter

from model import LLM

llm_class = LLM()
LLM_MODEL = llm_class.get_claude_v3_model()
EMBEDDING = llm_class.get_cohere_embedding()
LF_HANDLER = llm_class.get_langfuse_handler()


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


def format_chat_history(chat_history):
    formatted_history = ""
    for human_message, ai_response in chat_history:
        formatted_history += f"Human: {human_message}\nAI: {ai_response}\n"
    return formatted_history



@cl.on_chat_start
async def on_chat_start():
    files = None

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

    # index the chunks
    vectorstore = await cl.make_async(Chroma.from_texts)(chunks, EMBEDDING)
    retriever = vectorstore.as_retriever( )

    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the context. \n\n<context>{context}</context>\n\nHere is the question, {question}"
    )

    rag_chain_from_docs = (
        {
            "context": itemgetter("source_docs") | RunnableLambda(format_docs),
            "question": itemgetter("standalone_question"),
        }
        | prompt
        | LLM_MODEL
        | StrOutputParser()
        | (lambda resp: resp.strip())
    )

    rag_chain_with_sources = (
        RunnableParallel({
            "standalone_question": itemgetter("standalone_question")
        })
        .assign(source_docs = itemgetter("standalone_question") | retriever)
        .assign(answer = rag_chain_from_docs)
    )
    
    # Let the user know that the system is ready
    msg = cl.Message(
        content=f"Processing completed: `{text_file.name}`. Chat with your document now!",
        disable_feedback=True,
    )
    await msg.send()

    cl.user_session.set("rag_chain_with_sources", rag_chain_with_sources)
    cl.user_session.set("current_chat_history", [])


@cl.on_message
async def on_message(message: cl.Message):
    rag_chain_with_sources = cl.user_session.get("rag_chain_with_sources")
    lf_handler = llm_class.get_langfuse_handler()
    user_question = message.content

    standalone_question_prompt = ChatPromptTemplate.from_template("""Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is. Just return the question, nothing else.
    <chat_history>
    {chat_history}
    </chat_history>

    Latest User question: {question}"""
    )

    standalone_question_chain = (
        {
            "chat_history": itemgetter("chat_history") | RunnableLambda(format_chat_history),
            "question": itemgetter("question")
        }
        | standalone_question_prompt
        | LLM_MODEL
        | StrOutputParser()
        | (lambda resp: resp.strip())
    )

    current_chat_history = cl.user_session.get("current_chat_history")
    standalone_question = await standalone_question_chain.ainvoke({
        "question": user_question, 
        "chat_history": current_chat_history,
    })
    # print(f"{standalone_question=}")

    msg = cl.Message(content="", disable_feedback=False)
    await msg.send()

    text_elements = []  # type: List[cl.Text]

    async for chunk in rag_chain_with_sources.astream(
        {"standalone_question": standalone_question},
        config=RunnableConfig(
            callbacks=[
                cl.AsyncLangchainCallbackHandler(),
                lf_handler,
            ]
        ),
    ):
        if "source_docs" in chunk:
            for source_idx, source_doc in enumerate(chunk["source_docs"]):
                # Create the text element referenced in the message
                text_elements.append(
                    cl.Text(content=source_doc.page_content, name=f"{source_idx}")
                )
        if "answer" in chunk:
            await msg.stream_token(chunk["answer"])

    msg.content += f"\n\nSources: {', '.join([text_el.name for text_el in text_elements])}"
    msg.elements = text_elements
    await msg.update()

    # append converstation to history
    current_chat_history.append((standalone_question, msg.content))
    cl.user_session.set("current_chat_history", current_chat_history)


