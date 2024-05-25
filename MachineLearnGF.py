import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Загрузка данных
data = pd.read_csv('dataset.csv', names=['answer', 'question'], delimiter='`')
meta_data = pd.read_csv('meta_class.csv',names=['class','class_name'],delimiter='`')

# Подготовка данных
X_train, X_test, y_train, y_test = train_test_split(data['question'], data['answer'], test_size=0.2, random_state=42)

# Преобразование текста в матрицу признаков
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Обучение классификатора наивного Байеса
clf = MultinomialNB()
clf.fit(X_train_counts, y_train)

# Предсказание категорий для новых вопросов
new_questions = [
    "Могу ли я ожидать возврата какой-либо части оплаты, если я расторгаю договор после истечения определенного периода времени?",
    "Как записаться на бесплатную тестовую тренировку для диагностирования своих умений?",
    "у меня есть друг, хочу привести его",
    "приложение не работает`я не понимаю, не могу найти когда вы работаете, очень важно знать"
    # Добавьте остальные вопросы
]

new_questions_counts = vectorizer.transform(new_questions)
predicted_classes = clf.predict(new_questions_counts)

# Добавление информации о классах из meta_class.csv
class_names = {row['class']: row['class_name'] for _, row in meta_data.iterrows()}
predicted_class_names = [class_names[class_id] for class_id in predicted_classes]

# Вывод предсказанных классов
for i, question in enumerate(new_questions):
    print(f"Вопрос: {question}")
    print(f"Предсказанный класс: {predicted_classes[i]} - {predicted_class_names[i]}")
    print("-------------------------------------")