import csv
import pandas as pd


def refacted():
    with open('new_info.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='`')
        for row in reader:
            print(row)

    with open('new_info.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter='`')
        processed_data = [(row[0], row[2]) for row in reader]

    for row in processed_data:
        print(row)

    with open('new_infoRefacted.csv', 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        for row in processed_data:
            writer.writerow(row)
    print("Обработанные данные сохранены в файл 'обработанный_файл.csv'")


def metaClassed():
    namesOfClasses = ["Часы работы", "код клуба", "Заморозка карты", "не посещал по состояню здоровья",
                      "куда отправлять справку-больничный",
                      "диагностика", "бесплатная тренировка", "расторжение", "гостевой входит ли",
                      "персональные тренеры", "справка для посещения бассейна",
                      "разовый визит", "хочу пригласить друга", "в какое время меньше всего человек",
                      "аренда персональных шкафчиков", "правила посещения ТЗ",
                      "перефоормление карты", "не получается записаться на тренировки", "приложение не работает",
                      "не могу заморозить карту", "хочу массаж, есть ли солярий",
                      "нужна справка для налогового вычета", "хочу карту маме папе", "есть ли только бассейн",
                      "есть ли подарочные сертификаты", "у меня уже есть карта",
                      "кто мой менеджер", "почему меня не предупредили о отмене тренировки", "со мной не связались",
                      "есть ли бонусы за привод друга",
                      "куда можно оставить отзыв", "сколько сейчас человек в бассейне"]

    with open('meta_class.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)

        writer.writerow(["Class_ID", "Class_Name"])

        for i, name in enumerate(namesOfClasses, start=1):
            writer.writerow([i, name])
    print("metaClassed")
#refacted()
metaClassed()
#with open('meta_class.csv', 'r', encoding='utf-8') as f:
#    data = f.read()
#print(data)