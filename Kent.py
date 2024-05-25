import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

namesOfClasses = ["Часы работы","код клуба","Заморозка карты","не посещал по состояню здоровья","куда отправлять справку-больничный"
                  ,"диагностика","бесплатная тренировка","расторжение","гостевой входит ли","персональные тренеры","справка для посещения бассейна"
                  ,"разовый визит","хочу пригласить друга","в какое время меньше всего человек","аренда персональных шкафчиков","правила посещения ТЗ"
                  ,"перефоормление карты","не получается записаться на тренировки","приложение не работает","не могу заморозить карту","хочу массаж, есть ли солярий"
                  ,"нужна справка для налогового вычета", "хочу карту маме папе","есть ли только бассейн","есть ли подарочные сертификаты","у меня уже есть карта"
                  ,"кто мой менеджер","почему меня не предупредили о отмене тренировки","со мной не связались","есть ли бонусы за привод друга",
                  "куда можно оставить отзыв","сколько сейчас человек в бассейне"]

data = pd.read_csv('dataset.csv', names=['answer', 'question'], delimiter='`')

max_words = 75
max_len = 50

tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(data['question'])
X = tokenizer.texts_to_sequences(data['question'])
X = pad_sequences(X, maxlen=max_len)
Y = pd.get_dummies(data['answer']).values


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)



model = Sequential()
model.add(Embedding(max_words, 16))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(16, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(Y.shape[1], activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 32
epochs = 35

model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, Y_test), verbose=2)



new_questions = [
    "Могу ли я ожидать возврата какой-либо части оплаты, если я расторгаю договор после истечения определенного периода времени?",
    "Как записаться на бесплатную тестовую тренировку для диагностирования своих умений?",
    "Я болел и не мог работать, мне нужно как-то медицинскую справку куда-то предоставить",
    "Как я могу заморозить свою карту на некоторое время?",
    "А вечером клуб работает? Могу ли я прийти после работы",
    "Есть ли возможность приобрести билет на один визит в клуб?",
    "Можно ли привести друга с собой в клуб?",
    "Есть ли возможность арендовать персональные шкафчики для хранения личных вещей?",
    "Какие причины могут быть, если я не могу записаться на тренировки?"
]

new_sequences = tokenizer.texts_to_sequences(new_questions)
new_sequences = pad_sequences(new_sequences, maxlen=max_len)


predictions = model.predict(new_sequences)

predicted_classes = np.argmax(predictions, axis=1)


for i, question in enumerate(new_questions):
    print(f"Вопрос: {question}")
    print()
    print(f"Предсказанный класс: {predicted_classes[i]+1}, {namesOfClasses[predicted_classes[i]]} ")
    print("-------------------------------------")
from keras.models import load_model
model.save('question_classification_model.h5')