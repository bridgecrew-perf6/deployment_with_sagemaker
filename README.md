# Deploying a Sentiment Analysis Model with SageMaker

For this project, we used SageMaker to implement a sentiment analysis model for IMDb reviews using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/).

## Preprocessing
As with almost all NLP projects, we do not just use the text data, and we process it beforehand. We remove any html tags from the reviews using ```BeautifulSoup```. We then define and use a regular expression that removes the characters that are not alphanumeric from the data. Finally, we use ```nltk``` to tokenize the reviews.


Before moving on, I would like to emphasize one little thing we perform, removing the **stop words**! Even though I worked on several NLP projects beforehand, this is actually my first time hearing about stop words. According to [Wikipedia](https://en.wikipedia.org/wiki/Stop_word) __"stop words are any words which are filtered out (i.e. stopped) before or after processing of natural language data (text)."__ I always __did__ drop common words before tokenizing the input. However, I did not know that those common words had a name. Just thought you might want to know!

We then create a lookup table that maps words to integers, which we will use during training. Finally, we use this lookup table to convert reviews into groups of integers and pad the reviews so they are all of the same length before giving them input to the network. 

## Network

The network is an LSTM with one embedding layer, one lstm layer and one linear layer with sigmoid activation. We can use ```pygmantize``` command to visualize the piece of code that implements the network in a very pretty way! 
![Alt text](pyg.png?raw=true "Title")

## Training

This part is where SageMaker shines! We create a training job and train the model using the training loop we implemented. Here is the nice part about using SageMaker: we can create the training job on any machine instance that we like and it will run on that machine. This means that our notebook can run on a lightweight (and cheap) instance and we can __outsource__ the training to a large and compute optimized machine. Also, its PyTorch integration makes using custom PyTorch models very easy. Here is the code to create the training job and outsource the training job.
``` 
estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p3.2xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,})
```
