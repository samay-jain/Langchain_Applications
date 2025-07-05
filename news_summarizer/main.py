from news_summarizer import NewsArticleSummarizer

def main():
    # Example of using both models
    url = "https://www.moneycontrol.com/news/business/economy/india-s-rice-buffers-overflowing-as-another-record-harvest-looms-13191968.html"

    # # Initialize both summarizers
    # openai_summarizer = NewsArticleSummarizer(
    #     api_key=os.getenv("OPENAI_API_KEY"),
    #     model_type="openai",
    #     model_name="gpt-4o-mini"
    # )

    ollama_summarizer = NewsArticleSummarizer(
        model_type="ollama",
        model_name="llama3.2"
    )

    # # Get summaries from both models
    # print("\nGenerating OpenAI Summary...")
    # openai_summary = openai_summarizer.summarize(url, summary_type="detailed")

    print("\nGenerating Llama Summary...")
    llama_summary = ollama_summarizer.summarize(url, summary_type="detailed")

    # Print results
    for summary, model in [(llama_summary, "Llama")]:
        print(f"\n{model} Summary:")
        print("-"*50)
        print(f"Title: {summary['title']}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        # Print first document content
        print("\nFirst Document Content:")
        print(summary["summary"]["input_documents"][0].page_content)

if __name__ == "__main__":
    main()