# spam-message-detection-customize
ACKNOWLEDGEMENT





The success and final outcome of this project required a lot of guidance and assistance from many people and we are extremely fortunate to have got this all along the completion of our project work. Whatever we have done is only due to such guidance and assistance and we would not forget to thank them.


It is matter of great pleasure for us to submit the project report on “SPAM MESSAGES DETECTION” as a part of our curriculum.


First and foremost, we would like to thank to our Director Dr. Geeta S. Latkar, for giving us an opportunity to do the project work. We would like to thank our HOD, guide Prof. Ankit Anand and teachers for the valuable guidance and advice. They inspired us greatly to work in this project. Their willingness to motivate us contributed tremendously to our project.


And last but not least a special thanks goes to my team members, who helped me to assemble the information and gave suggestions to complete our project. 

INTRODUCTION


                            The goal of the spam message classifier project is to build a machine learning model that can accurately identify whether a given message is spam or not. The model will be trained on a dataset of labeled messages, where each message is classified as either spam or ham (not spam). The Multinomial Naive Bayes (MultinomialNB) algorithm will be used for this project, as it is a popular choice for text classification tasks.

The project will involve several steps, including data preprocessing, feature extraction, model training, and evaluation. The dataset will be preprocessed by removing stop words, punctuation, and any other unnecessary characters. Feature extraction will be done using the Bag of Words (BoW) technique, which involves counting the frequency of each word in a message and representing it as a vector.

The MultinomialNB algorithm will be trained on the BoW vectors, and its performance will be evaluated using metrics such as accuracy, precision, recall, and F1-score. The goal is to achieve high accuracy and minimize false positives (i.e., classifying a ham message as spam).

Overall, this project aims to build a robust and accurate spam message classifier that can help users filter out unwanted messages and protect their inbox from spam
  
.
                              

ABSTRACT




In order to use MultinomialNB for spam message classification, a set of labeled training data is required. This training data should include examples of both spam and non-spam messages. The algorithm can then be trained on this data to learn the statistical patterns that distinguish between spam and non-spam messages
This is a complete machine learning end-to-end project for building a Web App for the detection of Spam messages or the classification of any messages into Spam and Ham with very high accuracy. For this project, NLP Tools (NLTK, Gensim etc.), Bag of words model, MultinomialNB are used. For building the Web App, streamlit library is used 

METHODOLOGY

Here are the detailed steps
1.	Data collection: Collecting a large dataset of messages that are labeled as spam or not spam. This can be done by scraping email data from public sources or by creating a labelled dataset using crowdsourcing platforms.
2.	Data pre-processing: Cleaning and preparing the data for use in the model. This includes removing stop words, punctuations, and other unnecessary characters, as well as stemming or lemmatizing words to reduce dimensionality. The dataset is then split into training and testing sets.
•	Identifying Missing values.
•	Converting all text to lower case.
•	Performing tokenization.
•	Removing Stop words.
•	Labelling classes: ham/spam: {0;1}
•	Splitting Train and Test Data: 80% and 20%.

3.	Feature extraction: Using the Bag of Words (BoW) technique to extract features from the text data. This involves counting the frequency of each word in a message and representing it as a vector. The TF-IDF (Term Frequency-Inverse Document Frequency) technique can also be used to weigh the importance of each word in a message.
4.	Model training: Training a MultinomialNB classifier on the BoW or TF-IDF vectors to classify messages as spam or not. This involves tuning the hyperparameters of the MultinomialNB algorithm, such as the smoothing factor, to improve the classifier's performance.
5.	Model evaluation: Evaluating the performance of the model using various metrics such as accuracy, precision, recall, and F1-score. This is done by comparing the predicted labels to the actual labels in the testing set.
6.	Model deployment: Deploying the model to a production environment, such as a web application or API, to provide real-time classification of incoming messages. This involves setting up the infrastructure and integrating the model into the application or API.




                                  























                                                 Working

Naive Bayes is a powerful algorithm that is used for text data analysis and with problems with multiple classes. To understand Naive Bayes theorem’s working, it is important to understand the Bayes theorem concept first as it is based on the latter.
Bayes theorem, formulated by Thomas Bayes, calculates the probability of an event occurring based on the prior knowledge of conditions related to an event. It is based on the following formula:
P(A|B) = P(A) * P(B|A)/P(B)
Where we are calculating the probability of class A when predictor B is already provided.
P(B) = prior probability of B
P(A) = prior probability of class A
P(B|A) = occurrence of predictor B given class A probability














                                 Software Required

The Software used in making this project include:
o	Python: MultinomialNB is typically implemented in Python, so you will need to have Python installed on your computer. You can download Python from the official website: https://www.python.org/downloads/

o	Scikit-learn: Scikit-learn is a Python library that provides various machine learning algorithms, including MultinomialNB. You can install Scikit-learn using pip, a package installer for Python, by running the following command in your terminal or command prompt: 
	pip install scikit-learn


o	Pandas and NumPy: Pandas and NumPy are Python libraries that are commonly used for data manipulation and analysis. You may need to use these libraries to preprocess your data before using MultinomialNB. You can install them using pip by running the following commands:
	pip install pandas
	pip install numpy
               

o	A spam dataset: To train and test the MultinomialNB algorithm, you will need a dataset of labeled spam and non-spam messages. There are many publicly available spam datasets that you can use for this purpose, such as the Spam-Assassin public corpus.
         https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset










                    
                         Technologies Used Include

•	python :- Python has a rich ecosystem of ML libraries and frameworks, including Scikit-learn, TensorFlow, Keras, PyTorch, and many others. These libraries make it easier to implement various ML algorithms and models

•	streamlit:- Streamlit is a free and open-source framework to rapidly build and share beautiful machine learning and data science web apps. It is a Python-based library specifically designed for machine learning engineers. Data scientists or machine learning engineers are not web developers and they're not interested in spending weeks learning to use these frameworks to build web apps. Instead, they want a tool that is easier to learn and to use, as long as it can display data and collect needed parameters for modeling. Streamlit allows you to create a stunning-looking application with only a few lines of code.

•	 Numpy : NumPy is a Python library used for working with arrays.It also has functions for working in domain of linear algebra, fourier transform, and matrices.

•	Scikit-learn : It’s versatile and integrates well with other Python libraries, such as matplotlib for plotting, numpy for array vectorization, and pandas for dataframes. 

•	Matplotlib : Matplotlib is a cross-platform, data visualization and graphical plotting library for Python and its numerical extension NumPy. As such, it offers a viable open source alternative to MATLAB. Developers can also use matplotlib’s APIs (Application Programming Interfaces) to embed plots in GUI applications


 




OUTPUT AND SNAPSHOTS

 
 
 
 
 





	FEATURES 

1.	Fast and Efficient: MultinomialNB is a fast and efficient algorithm that can quickly classify incoming messages as spam or non-spam. This is because it requires only a small amount of memory and processing power.
2.	Good Performance: MultinomialNB performs well in spam message classification, even when the number of features (i.e., words in the message) is high. It is particularly good at handling text data, where the number of possible features can be very large.
3.	Simple and Easy to Implement: MultinomialNB is a simple and easy-to-implement algorithm that can be used even by beginners in machine learning. It is based on the Bayes' theorem and assumes that the occurrence of each word in the message is independent of the occurrence of other words.
4.	Requires Minimal Training Data: MultinomialNB requires only a small amount of labeled training data to build a classifier. This is particularly useful in cases where obtaining large amounts of labeled data is difficult or expensive.
5.	Handles Missing Data: MultinomialNB can handle missing data and still make accurate predictions. This is because it does not require complete information about all the features in the data.
 
CONCLUSION


•	Spam message classification is a critical task in information security and privacy. It involves identifying whether an incoming message is spam or not, and taking appropriate actions to prevent users from receiving unwanted or potentially harmful messages.
•	There are several machine learning algorithms that can be used for spam message classification, including Multinomial Naive Bayes (MultinomialNB), Support Vector Machines (SVMs), and Random Forests. Each algorithm has its own strengths and weaknesses, and the choice of algorithm depends on the specific requirements of the task.
 
 
References

	Scikit-learn documentation: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
	A comparative study of Naive Bayes classifiers in spam e-mail filtering: https://ieeexplore.ieee.org/document/1194739
	A review of machine learning algorithms for spam emails classification: https://www.sciencedirect.com/science/article/pii/S2405452617310393
	Naive Bayes and Text Classification: https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
	Multinomial Naive Bayes for Text Classification: https://towardsdatascience.com/multinomial-naive-bayes-for-text-classification-3b5b5bee980a



