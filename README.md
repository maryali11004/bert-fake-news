# BERT for Fake News Detection

This project uses a pre-trained BERT model to classify news statements as true or not true (binary classification). It includes preprocessing, feature selection, model training, evaluation, and inference.

## Requirements

To run the code, install the following dependencies:

```bash
pip install transformers datasets torch scikit-learn pandas nltk tqdm
```

## Dataset: LIAR Dataset

The dataset used in this project is the **LIAR dataset**, a publicly available dataset for fake news classification. It consists of 12,836 manually labeled short statements from various political contexts. Each statement is accompanied by detailed meta-data, including speaker, context, and justification.

The dataset is divided into three files:
- `train.tsv`
- `test.tsv`
- `valid.tsv`

Each file is tab-separated and structured as follows:
- Column 0: Statement ID
- Column 1: Original multi-class label (e.g., `true`, `false`, `mostly-true`, etc.)
- Column 2: Text of the statement

### Label Mapping

For binary classification, the labels are mapped as follows:
- `true`, `mostly-true` ➔ `1` (True)
- All other labels ➔ `0` (Not True)

## Preprocessing

1. **Text Cleaning**: Converts text to lowercase, removes special characters, multiple spaces, and stopwords.
2. **Feature Selection**: Extracts the most common 500 words using `CountVectorizer` for analysis purposes (not used in model training).

## Model

The model is based on the pre-trained BERT (`bert-base-uncased`) from Hugging Face. It is fine-tuned for binary classification with:
- Input sequence length: 128 tokens
- Learning rate: 2e-5
- Number of epochs: 7

### Optimizer and Scheduler
- Optimizer: AdamW
- Scheduler: StepLR (step size = 2, gamma = 0.1)

## Training and Evaluation

- **DataLoader**: Created for training, validation, and test datasets.
- **Training**: Loss is calculated using the built-in cross-entropy loss in BERT.
- **Evaluation**: Metrics include accuracy and a detailed classification report.

## Inference

The model can classify any given statement as `True` or `Not True`. Example usage is provided in the script.

## How to Use

1. Clone the repository and navigate to the project directory.
2. Place your dataset files (`train.tsv`, `test.tsv`, `valid.tsv`) in the same directory as the script.
3. Run the script:

```bash
python bert_fake_news.py
```

4. To test a custom statement, use the `predict_statement` function:

```python
print(predict_statement("The government will reduce taxes next year."))
```

## Saving and Loading the Model

- The trained model and tokenizer are saved to the `bert-fake-news` directory.
- Load the saved model for inference:

```python
loaded_tokenizer = BertTokenizer.from_pretrained('bert-fake-news')
loaded_model = BertForSequenceClassification.from_pretrained('bert-fake-news')
```

## Results

The script outputs:
- Training loss for each epoch.
- Validation and test set accuracy.
- Classification reports for validation and test sets.

## File Structure

```
.
├── bert_fake_news.py  # Main script
├── train.tsv          # Training dataset
├── test.tsv           # Test dataset
├── valid.tsv          # Validation dataset
├── bert-fake-news/    # Directory for saved model and tokenizer
└── README.md          # This README file
```

## Notes

- Ensure your dataset follows the specified format.
- The script automatically handles GPU acceleration if available.

## References

- [LIAR Dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [NLTK](https://www.nltk.org/)

## License

This project is licensed under the MIT License.

