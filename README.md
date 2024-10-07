# gw_model
古文模型
该模型是基于bert-ancient-chinese( https://github.com/Jihuai-wpy/bert-ancient-chinese )预训练模型，以 https://github.com/Ethan-yt/CCLUE/blob/main/data/punctuation 为数据集的模型，以bert-ancient-chinese为基础，目前，只实现了断句功能。未来，我们将推出关系抽取，标点符号添加等功能。  
如果您想使用这个模型，请  
先  python train.py 训练模型  
之后
python use.py 使用模型  
使用示例：  
<img width="334" alt="image" src="https://github.com/user-attachments/assets/c9276ff9-adcf-49b1-883d-b880455b20aa">
