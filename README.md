# Parts-Of-Speech-Tagger

This is a Parts Of Speech Tagger implementation using Markov Hidden Models(with the seqlearn library)

## Instructions:
1. install dependencies: `pip install -r requirements.txt`
2. run tests: `pytest`
3. Train the tagger: `python pos-tagger/train.py train_file dev_file`
4. Evaluate the trained tagger: `python pos-tagger/eval.py test_file`
* train_file, dev_file and test_file should be in Conll-2000 format
5. Generate tags for new text with the tagger: `python pos-tagger/generate.py test_csv_file`
* test_csv_file should contain one sentence per line and each line should be comma delimited

## Assumptions I made

1. The corpora the library is designed to deal with can fit in memory. In order to treat bigger corpora, the code will need adjustments and a different model implementation should be used.
2. The corpus being trained on is not very big and the compute available for training is limited, thus I should not use compute and data hungry models(LSTM, GRU,  Attention)

## Testing

1. I have implemented unit tests for the data loading and preprocessing steps.
2. I have tested the implementation end-to-end manually, but it should be automated
3. for the eval script, an automated test comparing the results on the test set to the ground truth shoud be implemented, but I did not have time, so it was tested manually on a small example.


