from langchain.chains import RetrievalQA

def init_chroma():
    """Initialize Chroma client"""
    # Initialize Chroma client here
    # chroma_client = ChromaClient(api_key=os.environ.get("CHROMA_API_KEY"))
    logger.info("Initializing Chroma...")


def load_chain() -> RetrievalQA:
    """Logic for loading the chain you want to use should go here."""
    llm = ChatOpenAI(temperature=0.5, model="gpt-3.5-turbo")
    
    # Use Chroma as the vector store
    chroma_client = init_chroma()
    vectorstore = chroma_client.load_vector_store(vectorstore_name=VECTORSTORE_NAME)

    prompt_template = """
    # Your prompt template
    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type_kwargs = {"prompt": PROMPT}
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    return qa
