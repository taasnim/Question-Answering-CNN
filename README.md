# Cloze Form Question Answering on CNN Dataset

*This was a class project for Deep Learning Course at NTU, Singapore.*

Reading comprehension is a form of a question-answering problem that tests the ability of a machine to read and understand documents by providing a model with a document/context from which the model has to find the answer to a given query. The task which we specifically focused on is called **cloze form question answering**. The speciality of cloze form question answering is that the query only has a single word answer, which is present in the document/passage.

For our task, we use the CNN dataset proposed by Hermann et al (2015). It was created by using news articles from CNN as the document/passage, and generating the query by using an abstractive summary of the passage and replacing one of the words in the query with a placeholder. The task is to predict this placeholder. Since it may be possible to solve the task by training a language model on other news articles, the creators anonymise the words (persons/organizations/locations, etc.) which are possible answer candidates by replacing them with unique, numbered placeholders, henceforth referred to as “entities”.
