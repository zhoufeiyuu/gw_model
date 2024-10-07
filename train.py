from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
model_path = "./punctuation_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)
model.eval()
print("输入 结束 程序结束")
while True:
    text = input("输入句子:")
    if text == "结束":
        break


    inputs = tokenizer(list(text), is_split_into_words=True, max_length = 128,return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

# 获取预测结果
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)

    tokens = inputs.tokens()
    id2label = {0: 'O', 1: 'B'}
    predicted_labels = [id2label[pred.item()] for pred in predictions[0]]


    tokens = tokens[1:-1]
    predicted_labels = predicted_labels[1:-1]

    segmented_text = ""
    for token, label in zip(tokens, predicted_labels):
        segmented_text += token
        if label == 'B':
            segmented_text += ' '

# 输出断句后的文本
    print("断句后:",segmented_text)
    #print(segmented_text)
