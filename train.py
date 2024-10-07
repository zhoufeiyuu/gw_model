from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments,AutoModelForMaskedLM
from torch.utils.data import DataLoader
import datasets

datas = datasets.load_dataset("punctuation.py", trust_remote_code=True)

train_data_tokens = datas['train']['tokens'][:-1]  # 取出所有 tokens，去掉最后一个
train_data_segs = datas['train']['seg_tags'][:-1]  # 取出所有标签，去掉最后一个
test_data_tokens = datas['test']['tokens']
test_data_segs = datas['test']['seg_tags']

#由于条件原因，我是把模型本地下载使用，如果跑不动，我可以提供自己本地下载的模型，然后使用我注释的 tokenizer model
#tokenizer = AutoTokenizer.from_pretrained("bert-ancient-chinese")
#model = AutoModelForTokenClassification.from_pretrained("bert-ancient-chinese")  # 'O', 'B'

tokenizer = AutoTokenizer.from_pretrained("Jihuai/bert-ancient-chinese")
model = AutoModelForMaskedLM.from_pretrained("Jihuai/bert-ancient-chinese")

def tokenize_and_align_labels(tokens, seg_tags):
    tokenized_inputs = tokenizer(tokens, truncation=True, padding=True, max_length=128, is_split_into_words=True)

    labels = []
    for i, label in enumerate(seg_tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = []
        previous_word_idx = None
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # 忽略此 token
            elif word_idx != previous_word_idx:
                label_ids.append(int(label[word_idx]))  # 使用断句标签
            else:
                label_ids.append(-100)  # 对子词进行跳过
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# 对训练集和测试集进行tokenize
train_tokenized = tokenize_and_align_labels(train_data_tokens, train_data_segs)
test_tokenized = tokenize_and_align_labels(test_data_tokens, test_data_segs)

# 转换为Dataset格式
train_dataset = datasets.Dataset.from_dict(train_tokenized)
test_dataset = datasets.Dataset.from_dict(test_tokenized)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",  # 每个 epoch 进行评估
    save_strategy="epoch", 
    learning_rate=2e-5,           # 学习率
    per_device_train_batch_size=8,  # 训练批量大小
    per_device_eval_batch_size=8,   # 评估批量大小
    num_train_epochs=1,             # 训练 epoch 数
    weight_decay=0.01,              # 权重衰减
)

# 使用Trainer进行训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()

# 保存训练好的模型
trainer.save_model("./punctuation_model")
tokenizer.save_pretrained("./punctuation_model")
print("over")
