
import sys


def check_gpu():
    import torch

    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")
    if cuda_ok:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: No CUDA GPU detected. Running on CPU.")
    return cuda_ok


def check_imports():
    import pandas
    import sentence_transformers
    import chromadb
    import langgraph
    import spacy

    spacy.load("en_core_web_sm")
    print("Imports OK: pandas, sentence-transformers, chromadb, langgraph, spaCy")


def main():
    print(f"Python version: {sys.version.split()[0]}")
    check_imports()
    check_gpu()
    print("Environment check finished.")


if __name__ == "__main__":
    main()