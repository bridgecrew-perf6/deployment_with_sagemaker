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

This part is where SageMaker shines! We create a training job and train the model using the training loop we implemented. Here is the nice part about using SageMaker: we can create the training job on any machine instance that we like and it will run on that machine. This means that our notebook can run on a lightweight (and cheap) instance and we can __outsource__ the training to a large and compute optimized machine and only be billed for the amount of time that the training job is running. Additionally, SM's PyTorch integration makes using custom PyTorch models very easy. Here is the code to create the training job and outsource the training job. 
``` 
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py", #the file that has the training loop
                    source_dir="train", # the function that implements the training loop
                    role=role, # aws role
                    framework_version='0.4.0',
                    train_instance_count=1, #one instance to train the model 
                    train_instance_type='ml.p3.2xlarge', # a large instance with gpu 
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,})

estimator.fit({'training': input_data})

```
## Deployment

After the model is trained, we can serve the model on an endpoint at an instance of our choice so we can very conveniently run inference on the trained model with the following code:
```
redictor = estimator.deploy(initial_instance_count=1, instance_type='ml.p3.2xlarge')
predictor.predict(data)

```

## Serving the model to everyone

Assume that we want to use this endpoint in a sentiment analysis application, where the users input a sentence and we return whether the sentiment is negative or positive. The data points that will come from our users in that case will not be in the format that our neural network expects: they first need to be tokenized! For this job, we use another AWS service called **Lambda**. Lambda's purpose is to implement and serve functions without dealing with setting up a server. Also, we will only be billed when Lambda is invoked, not all the time when it is active. Thus, we implement the tokenization operation in Lambda by just copying our Python code from earlier. As a last step, we create a REST API that will act as a communication point between the users, and the model & our Lambda function using AWS API Gateway. Now, the users can invoke this API and get back the sentiment! 
 ![Alt text](ss.png?raw=true "Title")