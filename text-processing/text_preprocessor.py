import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize

# Download necessary NLTK resources
#nltk.download('')

def normalize_text(text):
    """Basic text normalization"""
    # Convert to lowercase
    text = text.lower()
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Standardize quotes
    text = re.sub(r'["""]', '"', text)
    # Standardize apostrophes
    text = re.sub(r'[''`]', "'", text)
    return text.strip()

def segment_by_paragraphs(text):
    """Segment text by paragraphs"""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    return paragraphs

def segment_by_sentences(text):
    """Segment text by sentences"""
    sentences = sent_tokenize(text)
    return sentences

def load_story(file_path):
    """Load story from text file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Example usage
if __name__ == "__main__":
    # Load sample text
    sample_text = """Once upon a time in a far away land.
    
    There lived a princess. She was very brave.
    
    One day, she decided to explore the dark forest nearby."""

    # Normalize
    normalized = normalize_text(sample_text)
    print("Normalized text:", normalized[:50], "...\n")

    # Segment
    paragraphs = segment_by_paragraphs(sample_text)
    print(f"Found {len(paragraphs)} paragraphs. First paragraph: {paragraphs[0]}\n")

    sentences = segment_by_sentences(sample_text)
    print(f"Found {len(sentences)} sentences. First sentence: {sentences[0]}")