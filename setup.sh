echo "Downloading dataset for Twitter Sentiment140..."
curl -LO http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip
rm trainingandtestdata.zip

echo "Downloading Twitter GloVe Embeddings..."
curl -LO http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip
unzip glove.twitter.27B.zip
rm glove.twitter.27B.zip
rm glove.twitter.27B.100d.txt glove.twitter.27B.50d.txt glove.twitter.27B.25d.txt
