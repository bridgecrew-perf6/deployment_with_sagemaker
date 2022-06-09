# Deploying a Sentiment Analysis Model with SageMaker

For this project, we used SageMaker to implement a sentiment analysis model for IMDb reviews using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/).

## Preprocessing
As with almost all NLP projects, we do not just use the text data, and we process it beforehand. We use ```nltk``` and ```BeautifulSoup``` libraries for preprocessing. Specifically, we remove any html tags from the reviews using ```BeautifulSoup```. We then define and use a regular expression that removes the characters that are not alphanumeric from the data. Finally, we use ```nltk``` to tokenize the reviews.


Before moving on, I would like to emphasize one little thing we perform, removing the **stopwords**! Even though I worked on several NLP projects beforehand, this is actually my first time hearing about stop words. According to [Wikipedia](https://en.wikipedia.org/wiki/Stop_word) "Stop words are any word in a stop list (or stoplist or negative dictionary) which are filtered out (i.e. stopped) before or after processing of natural language data (text)." I always __did__ drop common words before tokenizing the input. However, I did not know that those common words had a name. Just thought you might want to know!
