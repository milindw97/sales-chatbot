#!/usr/bin/env python3
"""
Setup helper script for Sales Chatbot
"""

import sys
from pathlib import Path
import shutil


def print_header(text):
    """Print a formatted header"""
    print("\n" + "=" * 60)
    print(text)
    print("=" * 60)


def check_python_version():
    """Check Python version"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9+ required")
        return False

    print("✅ Python version OK")
    return True


def create_env_file():
    """Create .env file from .env.example if it doesn't exist"""
    print_header("Setting up .env file")

    if Path(".env").exists():
        print("⚠️  .env already exists, skipping")
        return True

    if not Path(".env.example").exists():
        print("❌ .env.example not found")
        return False

    shutil.copy(".env.example", ".env")
    print("✅ Created .env from .env.example")
    print("\n📝 Please edit .env and configure:")
    print("   - LLM_PROVIDER (ollama or gemini)")
    print("   - GEMINI_API_KEY (if using Gemini)")
    print("   - EMBEDDING_PROVIDER (sentence-transformers or gemini)")

    return True


def check_directories():
    """Ensure required directories exist"""
    print_header("Checking Directories")

    dirs = [
        "data/transcripts",
        "data/db",
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_path}")

    return True


def check_transcripts():
    """Check if transcript files are present"""
    print_header("Checking Transcript Files")

    transcript_dir = Path("data/transcripts")
    transcripts = list(transcript_dir.glob("*.txt"))

    if not transcripts:
        print("⚠️  No transcript files found in data/transcripts/")
        print("   Please add .txt transcript files to data/transcripts/")
        return False

    print(f"✅ Found {len(transcripts)} transcript file(s):")
    for t in transcripts:
        print(f"   - {t.name}")

    return True


def test_imports():
    """Test if all dependencies are installed"""
    print_header("Testing Dependencies")

    required_modules = [
        ("dotenv", "python-dotenv"),
        ("sqlalchemy", "sqlalchemy"),
        ("sentence_transformers", "sentence-transformers"),
        ("faiss", "faiss-cpu"),
        ("numpy", "numpy"),
        ("ollama", "ollama"),
        ("google.generativeai", "google-generativeai"),
    ]

    missing = []
    for module_name, package_name in required_modules:
        try:
            __import__(module_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} (missing)")
            missing.append(package_name)

    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Run: pip install -r requirements.txt")
        return False

    return True


def check_ollama():
    """Check if Ollama is available"""
    print_header("Checking Ollama (Optional)")

    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        print("✅ Ollama is running")
        print(f"   Available models: {len(models.get('models', []))}")
        return True
    except Exception as e:
        print(f"⚠️  Ollama not available: {e}")
        print("   If using Ollama, ensure it's running: ollama serve")
        return False


def main():
    """Run all setup checks"""
    print("""
╔══════════════════════════════════════════════════════════════╗
║          SALES CHATBOT - SETUP CHECKER                       ║
╚══════════════════════════════════════════════════════════════╝
""")

    checks = [
        ("Python Version", check_python_version, True),
        (".env Configuration", create_env_file, True),
        ("Directories", check_directories, True),
        ("Dependencies", test_imports, True),
        ("Transcript Files", check_transcripts, False),  # Warning only
        ("Ollama", check_ollama, False),  # Optional
    ]

    results = []
    for name, check_func, required in checks:
        try:
            success = check_func()
            results.append((name, success, required))
        except Exception as e:
            print(f"❌ Error in {name}: {e}")
            results.append((name, False, required))

    # Summary
    print_header("SETUP SUMMARY")

    required_passed = all(success for _, success, required in results if required)

    for name, success, required in results:
        req_text = "REQUIRED" if required else "OPTIONAL"
        status = "✅ PASS" if success else ("❌ FAIL" if required else "⚠️  SKIP")
        print(f"{name:25} {req_text:10} {status}")

    print()

    if required_passed:
        print("🎉 Setup complete! You can now run the chatbot:")
        print("   python cli.py")
        print()
        print("💡 Quick start:")
        print("   1. Edit .env if needed (configure LLM provider)")
        print("   2. Add transcript files to data/transcripts/")
        print("   3. Run: python cli.py")
        print("   4. Type: ingest all")
        print("   5. Start asking questions!")
    else:
        print("❌ Setup incomplete. Please fix the errors above.")
        print("   Run: pip install -r requirements.txt")

    return required_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
