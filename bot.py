from LRmodel import classification
import csv
import random
from sklearn.feature_extraction.text import CountVectorizer

def load_answers(path):
    answers = dict()
    with open(path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=';')
        for row in readCSV:
            answers[row[0]] = row[1:]
    return answers

def reply(intention, answers):
    import random

    n = len(answers[intention])

    if intention == 'saudacao' or intention == 'despedida':
        return print(answers[intention][random.randrange(n)])
    else:
        return print(answers[intention][random.randrange(n)] + " Posso ajudar com algo mais?")

def main():
    path = 'baseResposta.csv'
    answers = load_answers(path)

    intention = ''

    print("Ola. Seja bem vindo ao canal de atendimento para Portabilidade Vivo.")

    while intention != 'despedida':
        question = input()
        if question in ['não','n','nao']:
            print("Ok, ate breve!")
            break
        else:
            intention = classification(question)
            reply(intention, answers)

main()