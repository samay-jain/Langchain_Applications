# ----------------------------------------
# Required Installations:
# ----------------------------------------
# pip install yt_dlp
# pip install -q git+https://github.com/openai/whisper.git
# pip install langchain langchain-openai langchain-community chromadb python-dotenv

# FFmpeg is required by Whisper. Installation depends on your OS:
# macOS:     brew install ffmpeg
# Ubuntu:    sudo apt install ffmpeg
# Windows:   https://www.hostinger.com/tutorials/how-to-install-ffmpeg
# ----------------------------------------

import os
from typing import List, Dict

# External libraries for video/audio download, transcription, and processing
import yt_dlp
import whisper
from dotenv import load_dotenv

# LangChain utilities
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain_core.documents import Document

# Load environment variables (e.g. OPENAI_API_KEY)
load_dotenv()


class EmbeddingModel:
    """
    A wrapper for loading different types of embedding models.
    Supported types: "openai", "chroma", "nomic"
    """

    def __init__(self, model_type="openai"):
        self.model_type = model_type

        if model_type == "openai":
            # OpenAI Embeddings (requires API key)
            self.embedding_fn = OpenAIEmbeddings(
                model="text-embedding-3-small",
                openai_api_key=os.getenv("OPENAI_API_KEY"),
            )
        elif model_type == "chroma":
            # Local HuggingFace embedding
            from langchain_huggingface import HuggingFaceEmbeddings
            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "nomic":
            # Local Ollama embedding
            from langchain_ollama import OllamaEmbeddings
            self.embedding_fn = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")


class LLMModel:
    """
    A wrapper for loading different types of language models (LLMs).
    Supported types: "openai", "ollama"
    """

    def __init__(self, model_type="openai", model_name="gpt-4"):
        self.model_type = model_type
        self.model_name = model_name

        if model_type == "openai":
            # OpenAI LLM (GPT-3.5/4)
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI API key is required for OpenAI models")
            self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        elif model_type == "ollama":
            # Ollama-hosted local model (e.g. llama3)
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported LLM type: {model_type}")


class YoutubeVideoSummarizer:
    """
    End-to-end system to download a YouTube video, transcribe it,
    summarize the content, and set up a Q&A system.
    """

    def __init__(self, llm_type="openai", llm_model_name="gpt-4", embedding_type="openai"):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)
        self.whisper_model = whisper.load_model("base")  # Whisper ASR model

    def get_model_info(self) -> Dict:
        """Return a dictionary of the current model configuration."""
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str, str]:
        """
        Download a YouTube video and extract audio as MP3.
        Returns a tuple of (audio_path, video_title).
        """
        print("Downloading video...")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe the audio file using Whisper and return the full transcript text."""
        print("Transcribing audio...")
        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> List[Document]:
        """Split transcript into smaller chunks and wrap them in LangChain Document objects."""
        print("Creating documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        texts = text_splitter.split_text(text)
        return [Document(page_content=chunk, metadata={"source": video_title}) for chunk in texts]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Generate a Chroma vector store from documents for retrieval-based QA."""
        print(f"Creating vector store using {self.embedding_model.model_type} embeddings...")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(self, documents: List[Document]) -> str:
        """
        Generate a concise and detailed summary using LangChain's map-reduce summarize chain.
        """
        print("Generating summary...")

        # Prompt for individual chunks
        map_prompt = ChatPromptTemplate.from_template(
            """Write a concise summary of the following transcript section:\n"{text}"\nCONCISE SUMMARY:"""
        )

        # Prompt to combine into detailed summary
        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript sections:\n"{text}"\n
            Include:\n- Main topics and key points\n- Important details and examples\n- Any conclusions or call to action\n\nDETAILED SUMMARY:"""
        )

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )
        return summary_chain.invoke(documents)

    def setup_qa_chain(self, vector_store: Chroma):
        """Create a conversational retrieval-based QA chain using the provided vector store."""
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True,
        )

    def process_video(self, url: str) -> Dict:
        """
        Full pipeline for downloading, transcribing, summarizing a video, and initializing Q&A.
        Returns a dictionary with title, summary, full transcript, and the QA chain.
        """
        try:
            os.makedirs("downloads", exist_ok=True)
            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)
            os.remove(audio_path)  # Clean up temporary audio file

            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None


def main():
    """
    CLI for interacting with the summarizer. Allows user to choose models,
    enter a YouTube video URL, get a summary, and ask questions interactively.
    """
    urls = [
        "https://www.youtube.com/watch?v=v48gJFQvE1Y&ab_channel=BrockMesarich%7CAIforNonTechies",
        "https://www.youtube.com/watch?v=XwZkNaTYBQI&ab_channel=TheGadgetGameShow%3AWhatTheHeckIsThat%3F%21",
    ]

    # Choose model types
    print("\nAvailable LLM Models:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama3.2")
    llm_choice = input("Choose LLM model (1/2): ").strip()

    print("\nAvailable Embeddings:")
    print("1. OpenAI")
    print("2. Chroma Default")
    print("3. Nomic (via Ollama)")
    embedding_choice = input("Choose embeddings (1/2/3): ").strip()

    llm_type = "openai" if llm_choice == "1" else "ollama"
    llm_model_name = "gpt-4" if llm_choice == "1" else "llama3.2"

    embedding_type = (
        "openai" if embedding_choice == "1"
        else "chroma" if embedding_choice == "2"
        else "nomic"
    )

    try:
        # Initialize the summarizer system
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type,
        )

        # Show selected model configuration
        model_info = summarizer.get_model_info()
        print("\nCurrent Configuration:")
        print(f"LLM: {model_info['llm_type']} ({model_info['llm_model']})")
        print(f"Embeddings: {model_info['embedding_type']}")

        # Get video URL and process it
        url = input("\nEnter YouTube URL: ")
        print("\nProcessing video...")
        result = summarizer.process_video(url)

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")
            print(result["summary"])

            # Enable Q&A
            print("\nYou can now ask questions about the video (type 'quit' to exit)")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    response = result["qa_chain"].invoke({"question": query})
                    print("\nAnswer:", response["answer"])

            # Show full transcript
            if input("\nWant to see the full transcript? (y/n): ").lower() == "y":
                print("\nFull Transcript:")
                print(result["full_transcript"])

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure required models and APIs are properly configured.")


if __name__ == "__main__":
    main()
