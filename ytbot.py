from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from urllib.parse import urlparse, parse_qs
from langchain_classic.retrievers.multi_query import MultiQueryRetriever


def generate_vector(link):
    """Takes a YouTube link, parses video_id from it, and generates vectorstore from the transcript of the video."""
    try:
        # Parse the YouTube link to extract the video ID
        parsed_url = urlparse(link)
        if parsed_url.hostname == 'www.youtube.com':
            query = parse_qs(parsed_url.query)
            video_id = query['v'][0] if 'v' in query else None
        elif parsed_url.hostname == 'youtu.be':
            video_id = parsed_url.path[1:]  # Extract video ID from shortened URL
        else:
            raise ValueError("Invalid YouTube link format")

        if not video_id:
            raise ValueError("Could not extract video ID from link")

        # Fetch transcript
        transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=['en'])
        transcript = " ".join(snippet.text for snippet in transcript_list.snippets)

    except TranscriptsDisabled as e:
        raise e
    except Exception as e:
        raise e

    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])

    # Generate embeddings and create vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)

    return vector_store



def ask_ytbot(messages, vector_store):
    """Retrieves relevant context and generates an answer to the given question, using message history as context."""

    # Extract the last user message as the question
    question = messages[-1]["content"] if messages else ""
    history = messages[:-1] if len(messages) > 1 else []

    if vector_store is None:
        return "Please provide a YouTube link first."

    """STEP-2 Retrieval"""
    llm = ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite')
    retriever = MultiQueryRetriever.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5}), # set to False to reduce logging
    )
    """## Step 3 - Augmentation"""
   

    prompt_template = """
        You are a helpful youtube video assistant
        Try to answer from the provided transcript of a youtube video context.
        Answer precisely and do not be unnecessarily long in your answer.
        If you think the Question was not discussed in the youtube video just say \"Irrelevant from the video\"\n        Chat History: {history}
        Context: {context}
        Question: {question}
        """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question", "history"]
    )

    def format_docs(retrieved_docs):
        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
        return context_text

    retrieved_docs = retriever.invoke(question)

    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    history_text = "\n".join([f'{m["role"]}: {m["content"]} ' for m in messages[:-1]])

    chain = {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough(), "history": RunnableLambda(lambda x: history_text)} | prompt | llm | StrOutputParser()

    return chain.stream(question)