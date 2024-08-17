
import os
import chainlit
from openai import OpenAI
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma.vectorstores import Chroma

from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_openai.chat_models.base import ChatOpenAI
from chainlit.types import AskFileResponse


# Use the below codes to load the passowrd in py or ipynb files
from dotenv import load_dotenv, find_dotenv


# Checking if the .env is loaded or not - Returns True
_ = load_dotenv(find_dotenv())

client = OpenAI()

# Setting the Environment Variables
client.api_key  = os.getenv('openai_api_key')

welcome_message = """Welcome to the Chainlit PDF QA demo! To get started:
1. Upload a PDF or text file
2. Ask a question about the file
"""

# Processing texts/contents of files
def process_files(file: AskFileResponse):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100
    )

    if file.type == "text/plain":
        loader = TextLoader
    elif file.type == "application/pdf":
        loader = PyPDFLoader

    file_loader = loader(file.path)
    documents = file_loader.load()
    docs = text_splitter.split_documents(documents)
    for idx, doc in enumerate(docs):
        doc.metadata["source"] = f"source_{idx}"

    return docs


# this method processes the file, generates embeddings, and prepares the document for efficient retrieval
def get_document_search(file: AskFileResponse):
    docs = process_files(file)

    # saving data in the user session
    chainlit.user_session.set("docs", docs)

    embeddings = OpenAIEmbeddings()

    # Create a unique namespace for the file
    document_search = Chroma.from_documents(docs, embeddings)

    return document_search


# This decorator is used to define a function that runs when a chat session starts
# It initializes the chatbot interaction by setting up necessary components and sending initial messages
@chainlit.on_chat_start
async def start_chat():
    # Sending an image with the local file path
    await chainlit.Message(content="You can now chat with your pdfs.").send()
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await chainlit.AskFileMessage(
            content = welcome_message,
            accept= ["text/plain", "application/pdf"],
            max_size_mb= 20,
            timeout= 180
        ).send()

    file = files[0]
    
    # message is sent to the user indicating that the file is being processed
    message = chainlit.Message(content = f"Processing..`{file.name}`...")
    await message.send()

    # setting up a document search system by converting the function to an asynchronous function
    document_search = await chainlit.make_async(get_document_search)(file)

    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(             # combines document retrieval with question-answering
        ChatOpenAI(temperature=0, streaming=True),                      # model is used to generate answers based on the retrieved information
        chain_type="stuff",                                             
        retriever= document_search.as_retriever(max_tokens_limit=4097), # converts the document search system into a retriever
    )

    message.content = f"`{file.name}` processed. You can now ask questions!"
    await message.update()

    chainlit.user_session.set("qa_chain", qa_chain)


# This decorator is used to define a function that will be executed each time the chatbot receives a message from the user
# The function is asynchronous (async), which allows it to perform non-blocking operations, such as making API calls
@chainlit.on_message
async def main(message):

    # Retrieving the QA Chain
    qa_chain = chainlit.user_session.get("qa_chain")

    # Setting Up the Callback Handler
    callback_handler = chainlit.AsyncLangchainCallbackHandler(
        stream_final_answer= True,
        answer_prefix_tokens= ["FINAL", "ANSWER"]
    )
    callback_handler.answer_reached = True

    # Running the QA Chain. The `chain.acall` method is called to process the user's question (message) and generate a response. 
    # The callbacks=[cb] parameter ensures that the callback handler is used during this process
    result = await qa_chain.acall(message.content, callbacks=[callback_handler])


    # Extracting the answer and sources
    answer = result["answer"]
    sources = result["sources"].strip()
    source_elements = []

    # Handling document metadata
    # Get the documents from the user session
    docs = chainlit.user_session.get("docs")
    metadatas = [doc.metadata for doc in docs]
    all_sources = [md["source"] for md in metadatas]

    if sources:
        found_sources = []

        # Add sources to the message
        for source in sources.split(","):
            source_name = source.strip().replace(".", "")
            try:
                index = all_sources.index(source_name)
            except ValueError:
                continue

            text = docs[index].page_content
            found_sources.append(source_name)
            source_elements.append(chainlit.Text(content= text, name= source_name))

        if found_sources:
            answer += f"\nSources: {', '.join(found_sources)}"
        else:
            answer +="\nNo sources found"

    if callback_handler.has_streamed_final_answer:
        callback_handler.final_stream.elements = source_elements
        await callback_handler.final_stream.update()
    else:
        await chainlit.Message(content=answer,
                               elements= source_elements).send()