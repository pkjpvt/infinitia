import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.sequence import pad_sequences

try:
    data_dict = pickle.load(open('./data.pickle', 'rb'))

    # Assuming data_dict['data'] is a list of sequences (e.g., lists)
    # You may need to adjust padding options based on your data
    data = pad_sequences(data_dict['data'], dtype='float32', padding='post', truncating='post')

    labels = np.asarray(data_dict['labels'])

    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

    model = RandomForestClassifier()

    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)

    score = accuracy_score(y_test, y_predict)

    print('{}% of samples were classified correctly!'.format(score * 100))

    # Save the model
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
        # Save the model as a dictionary
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': model}, f)


except Exception as e:
    print(f"An error occurred: {str(e)}")
