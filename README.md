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
