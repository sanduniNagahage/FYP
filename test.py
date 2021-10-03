import torch
import pandas as pd
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
prep_csv = "prepositions_missing.csv"

MODEL_NAME = "google/electra-large-generator"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()
model.save_pretrained("/m")

model = AutoModelForMaskedLM.from_pretrained("/m")
model.eval()


# dataframe = pd.read_csv('preps.csv')

def masked_lm(text):
    # This text should only have one Masked word. And the Mask must be "[MASK]"
    masked_index = tokenizer.tokenize(text).index("[MASK]") + 1
    # We add +1 because the model ads a [CLS] token at the beginning
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    predictions = model(**inputs)[0]
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])
    # We are just going to replace the Mask with the rpedicted word
    out = text.split(" ")
    indx_mask = text.split(" ").index("[MASK]")
    out[indx_mask] = predicted_token[0]
    return predicted_token, " ".join(out)


# prediction = masked_lm("I will finish [MASK] 30 minutes.")
# print(prediction)

start = 5
end = 10
current = 0


def getRecord():
    print("reading...")


def readCSV():
    df = pd.read_csv(prep_csv)
    return df


def readRow(index, dataframe):

    sentence = dataframe['Sentence'][index]
    prepositions = dataframe['Prepostions'][index]
    row = [sentence, prepositions]

    return row


def tokenizeSentence(sentence):
    tokens = sentence.split(" ")
    return tokens


def rearrangePrepositions(preps):
    prep_sets = preps.split(";")
    preps_list = []
    for prep in prep_sets:
        prep_split = prep.split(",")
        preps_list.append([int(prep_split[0]), prep_split[1]])

    return preps_list


def addMsk(sentence_tokenized, prep):
    index = prep[0]
    sentence_tokenized[index-1] = "[MASK]"
    return sentence_tokenized


def arrangeSentence(masked_tokens):
    sentence = ""
    for token in masked_tokens:
        sentence += token+" "

    return sentence


def checkAccuracy(predicted_sentence, actual_sentence):
    print("predicted sentence = ", predicted_sentence)
    print("actual sentecne = ", actual_sentence)
    pred_tokens = tokenizeSentence(predicted_sentence)
    pred_tokens.pop()
    act_tokens = tokenizeSentence(actual_sentence)
    token_length = len(pred_tokens)
    count = 0
    for i in range(0, token_length):
        if(pred_tokens[i] == act_tokens[i]):
            count += 1
        i += 1
    accuracy = (count/token_length)*100
    if(accuracy != 100):
        return False
    return True


def findAccuracyPercentage(prepLength, states):
    true_state_count = 0
    for state in states:
        if(state == True):
            true_state_count += 1
    return (true_state_count/prepLength)*100


def test():

    dataframe = pd.read_csv(prep_csv)

    for current in range(end):
        print(current)
        if current < start:
            print('skip')
            current = current + 1
            continue

        sentence, prepositions = readRow(current, dataframe)
        print(sentence)
        sentence_tokens = tokenizeSentence(sentence)
        rearranged_prepositions = rearrangePrepositions(prepositions)
        mask_accuracy_state = []
        for prep in rearranged_prepositions:
            masked_tokens = addMsk(sentence_tokens, prep)
            masked_sentence = arrangeSentence(masked_tokens)
            predicted_token, predicted_sentence = masked_lm(masked_sentence)
            mask_accuracy_state.append(
                checkAccuracy(predicted_sentence, sentence))
            sentence_tokens = tokenizeSentence(sentence)

        prediction_accuracy = findAccuracyPercentage(
            len(rearranged_prepositions), mask_accuracy_state)

        print("PREDICTION ACCURACY = ", prediction_accuracy)
        # for token_sentence in masked_token_sentences:
        #     masked_sentence = arrangeSentence(token_sentence)
        #     # predicted_sentence = masked_lm(masked_sentence)
        #     print("sentence :", masked_sentence)

        current = current + 1


test()
# prediction = masked_lm("I will finish [MASK] 30 minutes.")
# print(prediction)
