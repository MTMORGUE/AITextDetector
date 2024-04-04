import os
import sys
import docx
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from collections import Counter
import math
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


def read_docx(file_path):
    """
    Reads a .docx file and returns the text content as a string.

    Args:
        file_path (str): The path to the .docx file.

    Returns:
        str: The text content of the .docx file.
    """
    doc = docx.Document(file_path)
    text = " ".join([para.text for para in doc.paragraphs])
    return text


def read_pdf(file_path):
    """
    Reads a .pdf file and returns the text content as a string.

    Args:
        file_path (str): The path to the .pdf file.

    Returns:
        str: The text content of the .pdf file.
    """
    with open(file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = " ".join([page.extract_text() for page in reader.pages])
    return text


def load_ai_wordlist(file_path):
    """
    Loads the AI wordlist from a file and returns a list of lowercase words.

    Args:
        file_path (str): The path to the AI wordlist file.

    Returns:
        list: A list of lowercase words associated with AI-generated text.
    """
    with open(file_path, "r") as file:
        ai_words = [word.strip().lower() for word in file.readlines()]
    return ai_words


def preprocess_text(text):
    """
    Preprocesses the input text by tokenizing, lowercasing, and removing stopwords.

    Args:
        text (str): The input text to be preprocessed.

    Returns:
        list: A list of preprocessed tokens.
    """
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    return filtered_tokens


def extract_adjectives(tokens):
    """
    Extracts adjectives from a list of tokenized words using part-of-speech tagging.

    Args:
        tokens (list): A list of tokenized words.

    Returns:
        list: A list of adjectives extracted from the tokens.
    """
    tagged_tokens = pos_tag(tokens)
    adjectives = [word for word, pos in tagged_tokens if pos.startswith("JJ")]
    return adjectives


def count_compound_sentences(text):
    """
    Counts the number of compound sentences in the input text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of compound sentences in the text.
    """
    sentences = sent_tokenize(text)
    compound_sentences = [sent for sent in sentences if "," in sent or ";" in sent]
    return len(compound_sentences)


def count_run_on_sentences(text):
    """
    Counts the number of run-on sentences in the input text.

    Args:
        text (str): The input text.

    Returns:
        int: The number of run-on sentences in the text.
    """
    sentences = sent_tokenize(text)
    run_on_sentences = []
    for sent in sentences:
        if len(word_tokenize(sent)) > 25:
            run_on_sentences.append(sent)
    return len(run_on_sentences)


def estimate_probabilities(human_text, ai_text, ai_words):
    """
    Estimates the word probabilities for human-written and AI-generated text based on the AI wordlist.

    Args:
        human_text (str): The human-written reference text.
        ai_text (str): The AI-generated reference text.
        ai_words (list): The list of words associated with AI-generated text.

    Returns:
        tuple: A tuple containing the human word probabilities and AI word probabilities.
    """
    human_adjectives = extract_adjectives(preprocess_text(human_text))
    ai_adjectives = extract_adjectives(preprocess_text(ai_text))

    human_word_counts = Counter(human_adjectives)
    ai_word_counts = Counter(ai_adjectives)

    total_human_words = sum(human_word_counts.values())
    total_ai_words = sum(ai_word_counts.values())

    human_probs = {word: count / total_human_words for word, count in human_word_counts.items() if word in ai_words}
    ai_probs = {word: count / total_ai_words for word, count in ai_word_counts.items() if word in ai_words}

    return human_probs, ai_probs


def analyze_syntax_variance(text, human_text, ai_text):
    """
    Analyzes the syntax variance between the input text and the human/AI reference texts using Kullback-Leibler divergence.

    Args:
        text (str): The input text to be analyzed.
        human_text (str): The human-written reference text.
        ai_text (str): The AI-generated reference text.

    Returns:
        tuple: A tuple containing the syntax variance, syntax variance percentage, and part-of-speech tag probabilities for the input text.
    """
    human_pos_tags = [tag for _, tag in pos_tag(word_tokenize(human_text))]
    ai_pos_tags = [tag for _, tag in pos_tag(word_tokenize(ai_text))]
    text_pos_tags = [tag for _, tag in pos_tag(word_tokenize(text))]

    human_pos_counts = Counter(human_pos_tags)
    ai_pos_counts = Counter(ai_pos_tags)
    text_pos_counts = Counter(text_pos_tags)

    total_human_tags = sum(human_pos_counts.values())
    total_ai_tags = sum(ai_pos_counts.values())
    total_text_tags = sum(text_pos_counts.values())

    human_pos_probs = {tag: count / total_human_tags for tag, count in human_pos_counts.items()}
    ai_pos_probs = {tag: count / total_ai_tags for tag, count in ai_pos_counts.items()}
    text_pos_probs = {tag: count / total_text_tags for tag, count in text_pos_counts.items()}

    kl_divergence_human = sum(
        text_pos_probs[tag] * math.log(text_pos_probs[tag] / human_pos_probs.get(tag, 1e-10)) for tag in text_pos_probs)
    kl_divergence_ai = sum(
        text_pos_probs[tag] * math.log(text_pos_probs[tag] / ai_pos_probs.get(tag, 1e-10)) for tag in text_pos_probs)

    syntax_variance = kl_divergence_human - kl_divergence_ai
    syntax_variance_percentage = syntax_variance / (kl_divergence_human + kl_divergence_ai) * 100

    return syntax_variance, syntax_variance_percentage, text_pos_probs


def estimate_ai_probability(text, human_probs, ai_probs, compound_sentences, run_on_sentences, ai_words,
                            syntax_variance, weight_factor=2.0):
    """
    Estimates the probability of the input text being AI-generated based on various linguistic features and statistical methods.

    Args:
        text (str): The input text to be analyzed.
        human_probs (dict): The word probabilities for human-written text.
        ai_probs (dict): The word probabilities for AI-generated text.
        compound_sentences (int): The number of compound sentences in the input text.
        run_on_sentences (int): The number of run-on sentences in the input text.
        ai_words (list): The list of words associated with AI-generated text.
        syntax_variance (float): The syntax variance between the input text and the human/AI reference texts.
        weight_factor (float, optional): The weight factor for AI words in the log-likelihood calculation. Defaults to 2.0.

    Returns:
        tuple: A tuple containing the estimated probability of the input text being AI-generated and the word frequencies in the input text.
    """
    adjectives = extract_adjectives(preprocess_text(text))
    word_counts = Counter(adjectives)

    log_likelihood = 0
    for word, count in word_counts.items():
        if word in human_probs and word in ai_probs:
            human_prob = human_probs[word]
            ai_prob = ai_probs[word]
            if word in ai_words:
                log_likelihood += weight_factor * count * (math.log(ai_prob) - math.log(human_prob))
            else:
                log_likelihood += count * (math.log(ai_prob) - math.log(human_prob))

    compound_sentence_ratio = compound_sentences / len(sent_tokenize(text))
    log_likelihood += math.log(compound_sentence_ratio + 1)

    run_on_sentence_ratio = run_on_sentences / len(sent_tokenize(text))
    log_likelihood -= math.log(run_on_sentence_ratio + 1)

    log_likelihood += syntax_variance

    ai_probability = 1 / (1 + math.exp(-log_likelihood))
    return ai_probability, word_counts


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_file> <ai_wordlist> [<human_text>] [<ai_text>]")
        sys.exit(1)

    input_file = sys.argv[1]  # The path to the input text file (.docx or .pdf) to be analyzed.
    ai_wordlist = sys.argv[2]  # The path to the file containing a list of words associated with AI-generated text.
    human_text_file = sys.argv[3] if len(
        sys.argv) > 3 else None  # The path to the file containing human-written reference text (optional).
    ai_text_file = sys.argv[4] if len(
        sys.argv) > 4 else None  # The path to the file containing AI-generated reference text (optional).

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    if not os.path.isfile(ai_wordlist):
        print(f"Error: AI wordlist file '{ai_wordlist}' does not exist.")
        sys.exit(1)

    if human_text_file and not os.path.isfile(human_text_file):
        print(f"Error: Human text file '{human_text_file}' does not exist.")
        sys.exit(1)

    if ai_text_file and not os.path.isfile(ai_text_file):
        print(f"Error: AI text file '{ai_text_file}' does not exist.")
        sys.exit(1)

    if input_file.endswith(".docx"):
        text = read_docx(input_file)
    elif input_file.endswith(".pdf"):
        text = read_pdf(input_file)
    else:
        print("Error: Unsupported file format. Only .docx and .pdf files are supported.")
        sys.exit(1)

    human_text = read_docx(human_text_file) if human_text_file and human_text_file.endswith(".docx") else read_pdf(
        human_text_file) if human_text_file else ""
    ai_text = read_docx(ai_text_file) if ai_text_file and ai_text_file.endswith(".docx") else read_pdf(
        ai_text_file) if ai_text_file else ""

    ai_words = load_ai_wordlist(ai_wordlist)
    human_probs, ai_probs = estimate_probabilities(human_text, ai_text, ai_words)

    compound_sentences = count_compound_sentences(text)
    run_on_sentences = count_run_on_sentences(text)
    syntax_variance, syntax_variance_percentage, text_pos_probs = analyze_syntax_variance(text, human_text, ai_text)
    ai_probability, word_counts = estimate_ai_probability(text, human_probs, ai_probs, compound_sentences,
                                                          run_on_sentences, ai_words, syntax_variance,
                                                          weight_factor=2.0)

    print("\nAI word probabilities:")
    for word, prob in ai_probs.items():
        print(f"{word}: {prob:.4f}")

    # print("\nHuman word probabilities:")
    # for word, prob in human_probs.items():
    #     print(f"{word}: {prob:.4f}")

    total_words = len(preprocess_text(text))
    adjective_count = len(extract_adjectives(preprocess_text(text)))
    print(f"\nTotal words: {total_words}")
    print(f"Total adjectives: {adjective_count}")
    print(f"Adjective ratio: {adjective_count / total_words:.2f}")
    print(f"\nCompound sentences: {compound_sentences}")
    print(f"Compound sentence ratio: {compound_sentences / len(sent_tokenize(text)):.2f}")
    print(f"\nRun-on sentences: {run_on_sentences}")
    print(f"Run-on sentence ratio: {run_on_sentences / len(sent_tokenize(text)):.2f}")
    print(f"\nOverall syntax variance percentage: {syntax_variance_percentage:.2f}%")
    print("Top 5 detected syntax variances:")
    for tag, prob in sorted(text_pos_probs.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{tag} (Part-of-Speech tag): {prob:.4f}")

    print(f"\nEstimated probability of the text being AI-generated: {ai_probability:.2%}")