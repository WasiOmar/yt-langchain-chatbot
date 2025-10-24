# Langchain Chatbot for YouTube Videos

## Description

A Streamlit application that uses Langchain and the YouTube Transcript API to create a chatbot that can answer questions about YouTube videos.

## Features

*   Takes a YouTube link as input.
*   Fetches the transcript of the video.
*   Generates a vectorstore from the transcript.
*   Allows users to ask questions about the video content.
*   Uses Langchain to retrieve relevant context and generate answers.

## Key Technologies and Components

*   **YouTube Transcript API:** Used to fetch the transcript of YouTube videos.
*   **RecursiveCharacterTextSplitter:** Used to split the transcript into smaller chunks for efficient processing.
*   **FAISS:** Used as the vectorstore to store and retrieve the transcript chunks based on semantic similarity.
*   **HuggingFace Embeddings:** Used to generate embeddings for the transcript chunks, enabling semantic similarity search.
*   **Langchain RunnableParallel, RunnablePassthrough, RunnableLambda:** Used to create a flexible and modular pipeline for processing user queries and generating responses.
*   **MultiQueryRetriever:** Used to generate multiple search queries from the user's input to improve retrieval of relevant context.
*   **Google Gemini (optional):** Used as the language model to generate answers based on the retrieved context.

## Dependencies

*   streamlit
*   python-dotenv
*   youtube-transcript-api
*   langchain
*   langchain_text_splitters
*   langchain_google_genai
*   langchain_community
*   langchain_huggingface
*   urllib3

## Setup Instructions

1.  Clone the repository.
2.  Create a virtual environment (optional but recommended).
3.  Install the dependencies using `pip install -r requirements.txt`.
4.  Create a `.env` file and add your Google Gemini API key (if using the Google Gemini model): `GOOGLE_API_KEY=YOUR_API_KEY`.
5.  Run the Streamlit app using `streamlit run app.py`.

## Usage

1.  Enter a YouTube link in the text input field.
2.  Ask questions about the video in the chat input field.

## Error Handling

*   The app includes error handling for common issues, such as blocked requests from the YouTube Transcript API.
*   If you encounter a "RequestBlocked" or "IPBlocked" error, try using proxies as described in the `youtube-transcript-api` documentation: <https://github.com/jdepoix/youtube-transcript-api?tab=readme-ov-file#working-around-ip-bans-requestblocked-or-ipblocked-exception>.

## Limitations

*   The YouTube Transcript API may be blocked due to excessive requests or IP restrictions.
*   The quality of the chatbot's answers depends on the accuracy and completeness of the YouTube video transcript.

## Contributing

(Optional) Information on how others can contribute to the project.

## License

(Optional) Information about the project's license.
