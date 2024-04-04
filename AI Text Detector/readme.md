# AI Text Detector

The AI Text Detector is a Python script that analyzes a given text file (.docx or .pdf) and estimates the probability of the text being AI-generated. It uses various linguistic features and statistical methods to compare the input text with human-written and AI-generated reference texts.

This script is based on the principles and methodology described in the research paper "Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews" by Weixin Liang, Zachary Izzo, Yaohui Zhang, et al. The design and implementation of this script are inspired by the findings and techniques presented in the paper.

The AI wordlist used in this script is derived from the wordlist provided in the research paper, which contains words that are disproportionately used more frequently by AI-generated text compared to human-written text.

## Initial Setup

To use the AI Text Detector, you need to have Python installed on your system. The script requires the following dependencies:

- `python-docx`: Library for reading .docx files
- `PyPDF2`: Library for reading .pdf files
- `nltk`: Natural Language Toolkit for text processing and analysis

You can install these dependencies using pip:

```
pip install python-docx PyPDF2 nltk
```

or

```
pip install -r requirements.txt
```

Additionally, you need to download the required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
```

The script also includes code to handle SSL certificate verification issues that may occur when downloading NLTK resources:

```python
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
```

This code creates an unverified HTTPS context to bypass SSL certificate verification errors.

## Usage

To run the AI Text Detector, use the following command:

```
python script.py <input_file> <ai_wordlist> [<human_text>] [<ai_text>]
```

- `<input_file>`: Path to the input text file (.docx or .pdf) to be analyzed.
- `<ai_wordlist>`: Path to the file containing a list of words associated with AI-generated text.
- `<human_text>` (optional): Path to the file containing human-written reference text.
- `<ai_text>` (optional): Path to the file containing AI-generated reference text.

The script will analyze the input text and provide an estimated probability of the text being AI-generated, along with various linguistic features and statistics.

## Code Overview

The AI Text Detector script consists of several functions and methods that work together to analyze the input text and estimate the probability of AI generation. Here's a detailed description of each component:

### File Reading Functions

- `read_docx(file_path)`: Reads a .docx file and returns the text content as a string.
- `read_pdf(file_path)`: Reads a .pdf file and returns the text content as a string.

### Text Processing Functions

- `load_ai_wordlist(file_path)`: Loads the AI wordlist from a file and returns a list of lowercase words.
- `preprocess_text(text)`: Preprocesses the input text by tokenizing, lowercasing, and removing stopwords.
- `extract_adjectives(tokens)`: Extracts adjectives from a list of tokenized words using part-of-speech tagging.
- `count_compound_sentences(text)`: Counts the number of compound sentences in the input text.
- `count_run_on_sentences(text)`: Counts the number of run-on sentences in the input text.

### Probability Estimation Functions

- `estimate_probabilities(human_text, ai_text, ai_words)`: Estimates the word probabilities for human-written and AI-generated text based on the AI wordlist.
- `analyze_syntax_variance(text, human_text, ai_text)`: Analyzes the syntax variance between the input text and the human/AI reference texts using Kullback-Leibler divergence.
- `estimate_ai_probability(text, human_probs, ai_probs, compound_sentences, run_on_sentences, ai_words, syntax_variance, weight_factor)`: Estimates the probability of the input text being AI-generated based on various linguistic features and statistical methods.

### Main Script

The main part of the script performs the following steps:

1. Checks if the required command-line arguments are provided and validates the file paths.
2. Reads the input text file (.docx or .pdf) and the human/AI reference texts (if provided).
3. Loads the AI wordlist.
4. Estimates the word probabilities for human-written and AI-generated text.
5. Counts the number of compound sentences and run-on sentences in the input text.
6. Analyzes the syntax variance between the input text and the human/AI reference texts.
7. Estimates the probability of the input text being AI-generated.
8. Prints the estimated probability, syntax variance percentage, top 5 detected syntax variances, compound sentence ratio, run-on sentence ratio, total words, total adjectives, adjective ratio, and AI word probabilities.

## Variables

- `text`: The input text to be analyzed.
- `human_text`: The human-written reference text.
- `ai_text`: The AI-generated reference text.
- `ai_words`: The list of words associated with AI-generated text.
- `human_probs`: The word probabilities for human-written text.
- `ai_probs`: The word probabilities for AI-generated text.
- `compound_sentences`: The number of compound sentences in the input text.
- `run_on_sentences`: The number of run-on sentences in the input text.
- `syntax_variance`: The syntax variance between the input text and the human/AI reference texts.
- `syntax_variance_percentage`: The overall syntax variance percentage.
- `text_pos_probs`: The part-of-speech tag probabilities for the input text.
- `ai_probability`: The estimated probability of the input text being AI-generated.
- `word_counts`: The word frequencies in the input text.

## Conclusion

The AI Text Detector script provides a comprehensive analysis of a given text file to estimate the probability of it being AI-generated. By leveraging various linguistic features and statistical methods, it compares the input text with human-written and AI-generated reference texts to identify potential indicators of AI generation.

Please note that the effectiveness of the script may depend on the quality and representativeness of the reference texts and the characteristics of the AI-generated text being analyzed. Further validation and refinement may be necessary based on specific use cases and datasets.

## Acknowledgments

We would like to acknowledge the research paper "Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews" by Weixin Liang, Zachary Izzo, Yaohui Zhang, et al. for providing the foundation and inspiration for this script. The methodology and techniques used in this script are based on the findings and principles presented in their paper.

We also thank the authors for providing the AI wordlist used in this script, which contains words that are disproportionately used more frequently by AI-generated text.
