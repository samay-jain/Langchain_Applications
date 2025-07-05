from yt_video_summarizer import YoutubeVideoSummarizer

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
