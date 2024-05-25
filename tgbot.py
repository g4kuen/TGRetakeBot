import csv
import logging
from typing import Callable, Dict, List, Union

import numpy as np
import pandas as pd
from telegram import (InlineKeyboardButton, InlineKeyboardMarkup,
                      ReplyKeyboardMarkup, ReplyKeyboardRemove, Update)
from telegram.ext import (CallbackContext, CallbackQueryHandler,
                          CommandHandler, ConversationHandler,
                          Filters, MessageHandler, Updater)

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

import config

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)

logger = logging.getLogger(__name__)

model = load_model('question_classification_model.h5')

max_words = 100
max_len = 50

data = pd.read_csv('dataset.csv', names=['answer', 'question'], delimiter='`')
tokenizer = Tokenizer(num_words=max_words, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
tokenizer.fit_on_texts(data['question'])

with open('meta_class.csv', newline='',encoding='UTF-8') as csvfile:
    reader = csv.reader(csvfile, delimiter='`')
    namesOfClasses = [row[1] for row in reader]
print(namesOfClasses)

def start(update: Update, context: CallbackContext) -> int:
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Привет! Я бот, который может классифицировать вопросы. "
             "Задайте мне вопрос, и я постараюсь отнести его к одному из классов."
    )

    return SELECT_QUESTION

def select_question(update: Update, context: CallbackContext) -> int:
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Введите вопрос:"
    )

    return SELECT_QUESTION


def handle_text_message(update: Update, context: CallbackContext) -> None:
    text = update.message.text

    processed_text = tokenizer.texts_to_sequences([text])
    processed_text = pad_sequences(processed_text, maxlen=max_len)

    processed_text = np.squeeze(processed_text)

    predicted_proba = model.predict(np.array([processed_text]))
    predicted_indices = np.argsort(predicted_proba)[0][-3:]
    predicted_proba = np.round(predicted_proba[0][predicted_indices], 3)

    keyboard = [
        [
            InlineKeyboardButton("Да", callback_data="yes"),
            InlineKeyboardButton("Нет", callback_data="no"),
        ]
    ]

    markup = InlineKeyboardMarkup(keyboard)
    print(text)
    print("Предсказанные классы и их вероятности:")
    for i, proba in zip(predicted_indices, predicted_proba):
        print(f"{i+1}. {namesOfClasses[i]}: {proba}")
    print("-------------------------------")
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Предсказанные классы и их вероятности:\n{'\n'.join([f'{i+1}. {namesOfClasses[i]}: {proba}' for i, proba in zip(predicted_indices, predicted_proba)])}\n{text}",
        reply_markup=markup,
    )

def handle_callback_query(update: Update, context: CallbackContext) -> None:
    query = update.callback_query
    data = query.data

    if data == "yes":
        pass

    if data == "no":
        question_text = query.message.text

        with open("new_info.csv", "a", newline="", encoding="UTF-8") as csvfile:
            writer = csv.writer(csvfile, delimiter="`")
            writer.writerow([question_text])

    query.message.delete()
def change_class(update: Update, context: CallbackContext) -> int:
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Введите пароль:"
    )

    return ENTER_PASSWORD

def enter_password(update: Update, context: CallbackContext) -> int:
    password = update.message.text

    if password == config.ADMIN_PASSWORD:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выберите класс, который хотите изменить:"
        )

        buttons = []
        for idx, name in enumerate(namesOfClasses, start=1):
            name=namesOfClasses[idx-1]
            print(name)
            button = InlineKeyboardButton(text=name, callback_data=str(idx))
            buttons.append(button)

        keyboard = [buttons[i:i + 2] for i in range(0, len(buttons), 2)]



        reply_markup = InlineKeyboardMarkup(keyboard)

        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Выберите класс:",
            reply_markup=reply_markup
        )

        return SELECT_CLASS_ID
    else:
        context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Неверный пароль!"
        )

        return ConversationHandler.END

def select_class_id(update: Update, context: CallbackContext) -> int:
    query = update.callback_query
    query.answer()

    class_id = int(query.data)

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Введите новое название для класса {namesOfClasses[class_id - 1]}:"
    )

    context.user_data['class_id'] = class_id

    return ENTER_NEW_CLASS_NAME

def enter_new_class_name(update: Update, context: CallbackContext) -> int:
    new_class_name = update.message.text

    class_id = context.user_data['class_id']

    with open(config.META_CLASSES_PATH, 'r',encoding='UTF-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='`')
        rows = list(reader)

    rows[class_id - 1][1] = new_class_name

    with open(config.META_CLASSES_PATH, 'w', newline='',encoding='UTF-8') as csvfile:
        writer = csv.writer(csvfile, delimiter='`')
        writer.writerows(rows)

    namesOfClasses[class_id - 1] = new_class_name

    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Класс {class_id} успешно изменен на {new_class_name}!"
    )

    return ConversationHandler.END

SELECT_QUESTION, ENTER_PASSWORD, SELECT_CLASS_ID, ENTER_NEW_CLASS_NAME = range(4)

def main() -> None:
    updater = Updater(token=config.BOT_TOKEN, use_context=True)

    dp = updater.dispatcher

    conversation = ConversationHandler(
        entry_points=[CommandHandler('start', start), CommandHandler('change', change_class)],
        states={
            SELECT_QUESTION: [
                MessageHandler(
                    Filters.text & (~Filters.command),
                    handle_text_message,


                )
            ],
            ENTER_PASSWORD: [
                MessageHandler(
                    Filters.text,
                    enter_password
                )
            ],
            SELECT_CLASS_ID: [
                CallbackQueryHandler(
                    select_class_id
                )
            ],
            ENTER_NEW_CLASS_NAME: [
                MessageHandler(
                    Filters.text,
                    enter_new_class_name
                )
            ],
        },
        fallbacks=[],
        allow_reentry=True,
        per_chat=True,
    )


    dp.add_handler(conversation)
    dp.add_handler(CallbackQueryHandler(handle_callback_query))
    
    updater.start_polling()

    updater.idle()

if __name__ == '__main__':
    main()
