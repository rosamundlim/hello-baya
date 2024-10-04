# adapted from https://github.com/langchain-ai/streamlit-agent/blob/main/streamlit_agent/chat_with_documents.py
########
# libs #
########
from dotenv import dotenv_values
import os
import streamlit as st
from langchain_community.chat_models import ChatOpenAI
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, OpenAIEmbeddings
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts.prompt import PromptTemplate
from langchain_community.vectorstores import Chroma
from pathlib import Path
import pickle
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate


################## 
# custom imports #
##################
from chunking import ChunkPipeline
from utils import StreamHandler, PrintRetrievalHandler

##########
# Config #
########## 
# api key - load a config variable instead of environment variable
env_path = Path(__file__).parents[1]/".env"
config = dotenv_values(env_path)

###########
# helpers # 
###########

@st.cache_resource(ttl="1h")
def configure_retriever():

    # with open(config.get("all_splits_filepath"), "rb") as f:
    #     all_splits = pickle.load(f)
    
    # with open(config.get("embedding_function_filepath"), "rb") as f:
    #     embedding_function = pickle.load(f)
    
    # Read documents
    chunk_pipeline = ChunkPipeline(config.get("filepath"))
    cleaned_text = chunk_pipeline.clean_text()
    all_splits = chunk_pipeline.splitter(cleaned_text)

    # Create the embedding function
    embedding_function = OpenAIEmbeddings(
        openai_api_key=config.get('openai_api_key')
    )
     # Initialize Chroma with the documents and embedding function
    db = Chroma.from_documents(
        all_splits,
        embedding_function
    )

    # Create a retriever from vectorstore (db)
    retriever = db.as_retriever(k=4)

    return retriever

#############
# streamlit # 
#############
    
st.title("Hello Bhaiya👋 - we're here to help!")
retriever = configure_retriever()

# Setup memory for contextual conversation
msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    chat_memory=msgs,
    return_messages=True
)

# #  Setup LLM and QA chain
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=config.get("openai_api_key"),
    temperature=0,
    streaming=True,
    max_tokens=100
)

#llm = AzureChatOpenAI(
#    api_key=config.get('api_key'),
#    azure_endpoint=config.get('endpoint'),
#    azure_deployment = config.get("deployment_name_model"),
#    api_version="2024-02-15-preview",) 
#

general_system_template = r"""
If no location/context is given, assume that questions are being asked in the context of Singapore.
Give an easily understandable answer to the question.
Use simple vocabulary.
Give answers in bullet points.
Provide addresses, websites, and contact information when mentioning locations/entities if available.
If the user does not understand, offer to translate into Bengali
-----------
{context}
-----------
"""

general_user_template = "Question:```{question}```"
messages = [
            SystemMessagePromptTemplate.from_template(general_system_template),
            HumanMessagePromptTemplate.from_template(general_user_template)
]


qa_prompt = ChatPromptTemplate.from_messages( messages )

qa_chain = ConversationalRetrievalChain.from_llm(
    llm, 
    retriever=retriever, 
    memory=memory, 
    verbose=True,
    chain_type = "stuff",
    combine_docs_chain_kwargs={'prompt': qa_prompt}#,
    #condense_question_prompt=condense_question_prompt
)

if len(msgs.messages) == 0 or st.sidebar.button("Clear message history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")

avatars = {"human": "user", "ai": "assistant"}
for msg in msgs.messages:
    st.chat_message(avatars[msg.type]).write(msg.content)

if user_query := st.chat_input(placeholder="How can we help?"):
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        retrieval_handler = PrintRetrievalHandler(st.container())
        stream_handler = StreamHandler(st.empty())
        response = qa_chain.run(
            user_query,
            callbacks=[retrieval_handler, stream_handler]
        )