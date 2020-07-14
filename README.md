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

## Model Choice
When approaching this problem, you can use many kinds of models. You can use a linear or Tree model and generate relevant features yourself or you can choose a sequence model that is designed to model relations within sequences. Obviously texts are sequences. There are examples of linear models with correct generated features that do well on this problem, but I still decided to go with a sequence model that is more natural for this problem. Inside the family of sequence models, there is still a selection. I decided to go with Hidden Markov Models since it is easy and quick(it only needs one pass over the data to train) to train, and it gives a not too bad accuracy. The downside of this model is that it captures only relations between a word and a previous word(because of the Markov Property) and also it cannot utilize other features that can be relevant(for example, a word that begins with a capital letter and is in the middle of the sentence is usually a noun). In order to capture more features, one can use a Conditional Random Field model. In order to campture more relations within the sequence, one can use LSTMs, Bi-Directional LSTMs or GRUs(but they will be more compute and data hungry).
For the level of confidence in my model: In order to give more confidence, I would implement a baseline solution that predicts the most frequent tag for every word and compare it's performance with my model. Out of lack of time I didn't implement the baseline. On top of that, the train accuracy I get is 95 and the test accuracy is 87, so there is a chance of some overfitting to the train set. I would try a bigger corpus to see if I can get a better test accuracy. I have confidence that the model generalizes well since the test and validation accuracies are very close.

## Assumptions I made

1. The corpora the library is designed to deal with can fit in memory. In order to treat bigger corpora, the code will need adjustments and a different model implementation should be used.
2. The corpus being trained on is not very big and the compute available for training is limited, thus I should not use compute and data hungry models(LSTM, GRU,  Attention)

## Testing

1. I have implemented unit tests for the data loading and preprocessing steps.
2. I have tested the implementation end-to-end manually, but it should be automated
3. for the eval script, an automated test comparing the results on the test set to the ground truth shoud be implemented, but I did not have time, so it was tested manually on a small example.


