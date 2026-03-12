#!/usr/bin/env python3
"""
Sales Call Chatbot CLI
Interactive command-line interface for querying sales call transcripts
"""

import os
import traceback

from src.config import (
    DB_PATH,
    FAISS_INDEX_PATH,
    LLM_PROVIDER,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_BASE_URL,
    SENTENCE_TRANSFORMER_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    TRANSCRIPTS_DIR,
    MAX_HISTORY_TURNS,
)
from src.storage.database import DatabaseManager
from src.storage.embeddings import get_embedding_provider
from src.storage.vector_store import FAISSVectorStore
from src.llm.providers import get_llm_provider
from src.ingestion.service import IngestionService
from src.retrieval.service import RetrievalService

try:
    from colorama import init, Fore, Style

    init()
    COLORS_ENABLED = True
except ImportError:
    COLORS_ENABLED = False


class SalesChatbot:
    """Main chatbot interface"""

    def __init__(self):
        self.initialized = False
        self.db_manager = None
        self.vector_store = None
        self.embedding_provider = None
        self.llm_provider = None
        self.ingestion_service = None
        self.retrieval_service = None
        self._history = []

    def initialize(self):
        """Initialize all components"""
        if self.initialized:
            return

        print("🚀 Initializing Sales Call Chatbot...")
        print(f"   LLM Provider: {LLM_PROVIDER}")
        print(f"   Embedding Provider: {EMBEDDING_PROVIDER}")
        print()

        # Initialize database
        self.db_manager = DatabaseManager(str(DB_PATH))

        # Initialize embedding provider
        if EMBEDDING_PROVIDER == "sentence-transformers":
            self.embedding_provider = get_embedding_provider(
                "sentence-transformers", model_name=SENTENCE_TRANSFORMER_MODEL
            )
        elif EMBEDDING_PROVIDER == "gemini":
            self.embedding_provider = get_embedding_provider(
                "gemini", api_key=GEMINI_API_KEY
            )

        # Initialize vector store
        self.vector_store = FAISSVectorStore(
            dimension=self.embedding_provider.get_dimension(),
            index_path=str(FAISS_INDEX_PATH),
        )

        # Initialize LLM provider
        if LLM_PROVIDER == "ollama":
            self.llm_provider = get_llm_provider(
                "ollama", model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL
            )
        elif LLM_PROVIDER == "gemini":
            self.llm_provider = get_llm_provider("gemini", api_key=GEMINI_API_KEY)

        # Initialize services
        self.ingestion_service = IngestionService(
            db_manager=self.db_manager,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        self.retrieval_service = RetrievalService(
            db_manager=self.db_manager,
            vector_store=self.vector_store,
            embedding_provider=self.embedding_provider,
            llm_provider=self.llm_provider,
            top_k=5,
        )

        self.initialized = True
        print("✅ Initialization complete!\n")

    def print_colored(self, text, color=None):
        """Print colored text if colors are enabled"""
        if COLORS_ENABLED and color:
            print(f"{color}{text}{Style.RESET_ALL}")
        else:
            print(text)

    def print_help(self):
        """Print help message"""
        help_text = """
╔══════════════════════════════════════════════════════════════╗
║              SALES CALL CHATBOT - COMMANDS                   ║
╚══════════════════════════════════════════════════════════════╝

📋 CALL MANAGEMENT:
   list calls                    - List all ingested call IDs
   list my call ids             - Same as 'list calls'
   
📥 INGESTION:
   ingest <file_path>           - Ingest a new call transcript
   ingest all                   - Ingest all files from data/transcripts/
   
📝 SUMMARIZATION:
   summarize <call_id>          - Summarize a specific call
   summarize the last call      - Summarize the most recent call
   
❓ QUESTIONS:
   Any natural language question about the calls, e.g.:
   - What objections were raised in the demo call?
   - Give me all negative comments when pricing was mentioned
   - What were the main security concerns?
   - What next steps were agreed upon?
   
🔧 UTILITY:
   reset                        - Clear conversation history
   help                         - Show this help message
   clear                        - Clear the screen
   exit / quit                  - Exit the chatbot
   
💡 TIP: Ask questions naturally! The bot will find relevant segments
        and cite the exact call and timestamp.

"""
        self.print_colored(help_text, Fore.CYAN)

    def handle_list_calls(self):
        """Handle listing all calls"""
        calls = self.retrieval_service.list_calls()

        if not calls:
            self.print_colored("\n⚠️  No calls found in the database.", Fore.YELLOW)
            self.print_colored(
                "   Use 'ingest <file_path>' to add calls.\n", Fore.YELLOW
            )
            return

        self.print_colored("\n📞 Available Calls:", Fore.GREEN)
        self.print_colored("=" * 70, Fore.GREEN)
        for i, call in enumerate(calls, 1):
            print(f"{i}. {call['call_id']}")
            print(f"   File: {call['filename']}")
            print(f"   Chunks: {call['total_chunks']}")
            print(f"   Ingested: {call['ingestion_date']}")
            print()

    def handle_ingest(self, args):
        """Handle ingestion commands"""
        if not args:
            self.print_colored("❌ Usage: ingest <file_path> or ingest all", Fore.RED)
            return

        if args[0].lower() == "all":
            # Ingest all from transcripts directory
            self.ingestion_service.ingest_directory(str(TRANSCRIPTS_DIR))
        else:
            # Ingest single file
            file_path = " ".join(args)
            if not os.path.exists(file_path):
                self.print_colored(f"❌ File not found: {file_path}", Fore.RED)
                return

            self.ingestion_service.ingest_transcript(file_path)

    def handle_summarize(self, args):
        """Handle summarization commands"""
        if not args:
            self.print_colored(
                "❌ Usage: summarize <call_id> or summarize the last call", Fore.RED
            )
            return

        # Check if "the last call" or "last call"
        input_text = " ".join(args).lower()

        # Handle various forms of "last call"
        if "last call" in input_text or input_text == "the call":
            # Get the most recent call
            calls = self.retrieval_service.list_calls()
            if not calls:
                self.print_colored("❌ No calls found in database", Fore.RED)
                return

            # Sort by ingestion date and get latest
            calls.sort(key=lambda x: x["ingestion_date"], reverse=True)
            call_id = calls[0]["call_id"]
            self.print_colored(
                f"\n📝 Summarizing most recent call: {call_id}", Fore.CYAN
            )
        # Handle generic "the call" or "this call" - show list
        elif input_text in ["the call", "this call", "call"]:
            self.print_colored("\n⚠️  Which call? Available calls:", Fore.YELLOW)
            calls = self.retrieval_service.list_calls()
            for i, call in enumerate(calls, 1):
                print(f"  {i}. {call['call_id']}")
            print("\nUsage: summarize <call_id>")
            return
        else:
            # Remove common words that might interfere
            clean_args = [
                arg
                for arg in args
                if arg.lower() not in ["the", "a", "an", "this", "that"]
            ]
            if not clean_args:
                self.print_colored(
                    "❌ Please specify a call_id. Use 'list calls' to see available calls.",
                    Fore.RED,
                )
                return
            call_id = clean_args[0]

        print(f"\n⏳ Generating summary for '{call_id}'...\n")

        result = self.retrieval_service.summarize_call(call_id)

        if "error" in result:
            self.print_colored(f"❌ {result['error']}", Fore.RED)
            self.print_colored(
                "\n💡 Tip: Use 'list calls' to see available call IDs", Fore.YELLOW
            )
            return

        self.print_colored(f"\n📊 SUMMARY: {result['call_id']}", Fore.GREEN)
        self.print_colored("=" * 70, Fore.GREEN)
        print(result["summary"])
        print()
        self.print_colored(
            f"Chunks: {result['total_chunks']} | Ingested: {result['ingestion_date']}",
            Fore.CYAN,
        )
        print()

    def handle_query(self, query):
        """Handle natural language queries"""
        print("\n⏳ Searching and generating answer...\n")

        result = self.retrieval_service.query(query, history=self._history)

        # Show if query was rewritten
        if result.get("query") and result["query"] != result.get("original_question"):
            self.print_colored(f"🔄 (Searched for: '{result['query']}')", Fore.BLACK + Style.BRIGHT)

        # Print answer
        self.print_colored("\n💡 ANSWER:", Fore.GREEN)
        self.print_colored("=" * 70, Fore.GREEN)
        print(result["answer"])
        print()

        # Update history
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": result["answer"]})

        # Cap history to MAX_HISTORY_TURNS (each turn is 2 messages)
        if len(self._history) > MAX_HISTORY_TURNS * 2:
            self._history = self._history[-(MAX_HISTORY_TURNS * 2):]

        # Print sources
        if result["sources"]:
            self.print_colored(f"📚 SOURCES ({result['num_sources']}):", Fore.CYAN)
            self.print_colored("=" * 70, Fore.CYAN)
            for i, source in enumerate(result["sources"], 1):
                print(f"{i}. Call: {source['call_id']}")
                print(f"   Time: {source['timestamp_range']}")
                print(f"   Relevance: {source['similarity_score']:.3f}")
                print(f"   Snippet: {source['snippet']}")
                print()

    def process_command(self, user_input):
        """Process user commands"""
        user_input = user_input.strip()

        if not user_input:
            return True

        # Normalize input
        lower_input = user_input.lower()

        # Handle commands
        if lower_input in ["exit", "quit", "q"]:
            self.print_colored("\n👋 Goodbye!\n", Fore.YELLOW)
            return False

        if lower_input in ["help", "h", "?"]:
            self.print_help()

        elif lower_input == "clear":
            os.system("clear" if os.name == "posix" else "cls")

        elif lower_input == "reset":
            self._history.clear()
            self.print_colored("\n🧹 Conversation history cleared.\n", Fore.GREEN)

        elif lower_input in ["list calls", "list my call ids", "list"]:
            self.handle_list_calls()

        elif lower_input.startswith("ingest"):
            args = user_input.split()[1:]
            self.handle_ingest(args)

        elif lower_input.startswith("summarize") or lower_input.startswith("summarise"):
            args = user_input.split()[1:]
            self.handle_summarize(args)

        else:
            # Treat as a natural language query
            self.handle_query(user_input)

        return True

    def run(self):
        """Main REPL loop"""
        self.initialize()

        # Print welcome message
        self.print_colored(
            """
╔══════════════════════════════════════════════════════════════╗
║          WELCOME TO SALES CALL CHATBOT 🤖                    ║
║                                                              ║
║  Ask questions about your sales calls naturally!             ║
║  Type 'help' for commands or 'exit' to quit                  ║
╚══════════════════════════════════════════════════════════════╝
""",
            Fore.CYAN,
        )

        # Check if any calls are loaded
        calls = self.retrieval_service.list_calls()
        if not calls:
            self.print_colored("⚠️  No calls loaded yet!", Fore.YELLOW)
            self.print_colored(
                "   Try: 'ingest all' to load sample transcripts\n", Fore.YELLOW
            )
        else:
            self.print_colored(
                f"✅ {len(calls)} call(s) loaded and ready!\n", Fore.GREEN
            )

        # Main loop
        while True:
            try:
                if COLORS_ENABLED:
                    user_input = input(f"{Fore.BLUE}You > {Style.RESET_ALL}")
                else:
                    user_input = input("You > ")

                if not self.process_command(user_input):
                    break

            except KeyboardInterrupt:
                self.print_colored("\n\n👋 Goodbye!\n", Fore.YELLOW)
                break
            except Exception as e:
                self.print_colored(f"\n❌ Error: {str(e)}\n", Fore.RED)
                traceback.print_exc()


def main():
    """Entry point"""
    chatbot = SalesChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
