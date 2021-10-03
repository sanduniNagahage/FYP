from unidirectional import start_unidirectional
from main import predict_masking
import pandas as pd

masking_indexes = [0, 1, 3, 4, 5, 6, 8, 10, 11,
                   13, 14, 16, 19, 20, 24, 26, 28, 29, 30, 32]


def csv_reader():
    dataframe = pd.read_csv('prepositions_missing.csv')
    return dataframe


def readRow(index, dataframe):
    sentence = dataframe['Sentence'][index]
    prepositions = dataframe['Prepostions'][index]
    row = [sentence, prepositions]

    return row


def sort_by_index(uni, mask):
    sorted_array = []
    for i in range(len(uni)):
        print(uni[i][0])
        sorted_array.insert(uni[i][0], [uni[i][1], uni[i][2], uni[i][3]])

    for j in range(len(mask)):
        sorted_array.insert(mask[j][0], [mask[j][1], mask[j][2], mask[j][3]])

    return sorted_array


def start_final():
    dataframe = csv_reader()
    masking_sentences = []
    uni_directional_sentences = []
    for index in range(len(dataframe)):
        sentence, prepositions = readRow(index, dataframe)
        if index in masking_indexes:
            masking_sentences.append([index, sentence, prepositions])

        else:
            uni_directional_sentences.append([index, sentence, prepositions])
            print([index, sentence, prepositions])

    print(len(uni_directional_sentences))
    print("************************UNI-DIRECTIONAL**************************")
    res_unidirectional = start_unidirectional(uni_directional_sentences)

    print("************************MASKING**********************************")
    res_masking = predict_masking(masking_sentences)

    sorted_sens = sort_by_index(res_unidirectional, res_masking)
    print("Sorted", sorted_sens)
    for x in range(len(sorted_sens)):

        sentence = sorted_sens[x][0]
        predictions = sorted_sens[x][1]
        percentage = sorted_sens[x][2]

        print("Actual Sentence :", sentence)
        print("Predictions :")
        print(predictions)
        print("PREDICTION ACCURACY OF THE SENTENCE :", percentage)

        # for y in predictions:
        #     print("* ", predictions[y][0])
        print()


start_final()
