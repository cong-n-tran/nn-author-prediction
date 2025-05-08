# Khanh Nguyen Cong Tran
# 1002046419

import tensorflow as tf
import numpy as np
import random

def learn_model(train_files):
   
    num_classes = len(train_files)
    max_tokens = 20000

    input, labels = create_dataset_and_labels(train_files=train_files)

    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=max_tokens,
        ngrams=1,
        output_mode='tf-idf'
    )
    
    dataset = tf.data.Dataset.from_tensor_slices((input, labels)) # make the dataset into a tf dataset
    dataset = dataset.shuffle(len(dataset)) # shuffle the dataset around

    text_vectorization.adapt(dataset.map(lambda x, y: x)) # this basically just shoving the input data in 

    model = tf.keras.Sequential([
        text_vectorization, 
        tf.keras.layers.Dense(64, activation='tanh'),
        tf.keras.layers.Dropout(0.75), # add HELLA dropout bc it was overfitting like crazy
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.9), # add EVEN MORE dropout bc it still overfits
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='rmsprop', # found rmsprop to be better 
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.fit(
            dataset.batch(32), # a cool way to do batch size with the tensorflow dataset object
            epochs=10
    )
    
    return model

def create_dataset_and_labels(train_files): 
    all_texts = []
    all_labels = []
    
    for author_idx, author_files in enumerate(train_files):
        # get all the sentenes in a particular author
        sentences = []

        # iterate through all the files of the author
        for file_path in author_files:
            with open(file_path, 'r', encoding='latin-1') as f: # open with latin-1 bc it got some unknown characters bruh

                # replace all '\n' with space
                text = f.read().replace('\n', ' ')

                # split based on period and then remove all trailing spaces
                book_sents = [s.strip() for s in text.split('.') if s.strip()]

                #add to sentences array
                sentences.extend(book_sents)
        
        # no sentences in book -> skip
        if not sentences:
            continue

        # get the number of words per sentence
        word_counts = [len(sent.split()) for sent in sentences]

        # calculate the number of sentences in total
        num_sentences = len(sentences)

        # the sample sentences used to training
        author_samples = []
        for _ in range(1000): #a thousand times per author
            while True:
                start = random.randint(0, num_sentences - 1) # pick a random sentence
                current_words = 0 # current number of words
                chunk = []
                for i in range(start, num_sentences):
                    current_words += word_counts[i] # update current word counter
                    chunk.append(sentences[i]) # create our chunk
                    if current_words >= 40: #check if we reached the 40 word threshold

                        # add the proper punctuations
                        chunk_text = '. '.join(chunk) + '.'

                        # add to sample and break 
                        author_samples.append(chunk_text)
                        break

                # edge case 
                if current_words >= 40:
                    break
        
        all_texts.extend(author_samples)
        all_labels.extend([author_idx] * len(author_samples))

    return (all_texts, all_labels)
