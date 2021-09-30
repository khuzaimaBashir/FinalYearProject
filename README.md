# FinalYearProject
Developed a Data Science & Deep Learning model as Final Year Bachelors Project using state of art python frameworks to convert sign language into English keywords and then into English sentences.

Following are the frameworks used for this project:
1. Pandas, Numpy
2. Pytorch, Tensorflow
3. Jupyter Notebook, Spyder
4. Resnet 3d (CNN & RNN)
5. Resnet (2 + 1) D
6. Seq2Seq (RNN)

The project was done by following a conventional Data Science pipeline phases:
1. Data Gathering, Cleaning, Wrangling
2. Data Preprocessing
3. Data Modeling (Deep Learning)
4. Model Evaluation & Presentation

The project was divided into two parts:
First part was to convert sign language video into pictures and then transformed into English words. This was done by using Open CV & Resnet 2+1 D and 3D tensorflow. This was our first model and the results were passed to the second model.
Second part was to transform English words into English Sentences with correct English grammar. This was done by using Seq2Seq Resnet model on Pytorch.
Finally both the model were linked with each other to form a Multi-model Neural Network.
