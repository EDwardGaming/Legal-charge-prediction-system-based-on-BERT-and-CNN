import os
import json
import numpy as np
import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from tensorflow.keras import layers, Model

# 配置
MAX_LENGTH = 256
MODEL_SAVE_PATH = "electra_cnn_legal"
LOCAL_MODEL_PATH = "./hfl/chinese-legal-electra-base-disc"

# 加载 tokenizer 和 electra 模型
tokenizer = ElectraTokenizer.from_pretrained(LOCAL_MODEL_PATH)
electra_model = TFElectraModel.from_pretrained(LOCAL_MODEL_PATH)

# 定义模型类
class LegalClassifier(Model):
    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.electra = electra_model
        self.conv1 = layers.Conv1D(128, 3, activation='gelu', kernel_regularizer=tf.keras.regularizers.l2(1e-4))
        self.bn = layers.BatchNormalization()
        self.pool = layers.GlobalMaxPooling1D()
        self.classifier = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        input_ids, attention_mask = inputs
        outputs = self.electra(input_ids, attention_mask=attention_mask)
        x = outputs.last_hidden_state
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pool(x)
        return self.classifier(x)

    def get_config(self):
        return {"num_classes": self.classifier.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# 加载 label 映射字典
with open("accu.txt", encoding="utf-8") as f:
    labels = [line.strip() for line in f.readlines()]
label_to_id = {label: i for i, label in enumerate(labels)}
id_to_label = {i: label for label, i in label_to_id.items()}

# 加载模型
model = tf.keras.models.load_model(
    MODEL_SAVE_PATH,
    custom_objects={"LegalClassifier": LegalClassifier}
)

# 预测函数
def predict_accusation(text):
    if not text.strip():
        return "❗️输入文本为空"

    tokens = tokenizer(
        text,
        max_length=MAX_LENGTH,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    input_ids = tokens["input_ids"]
    attention_mask = tokens["attention_mask"]

    preds = model.predict((input_ids, attention_mask))
    pred_id = np.argmax(preds, axis=1)[0]
    return id_to_label[pred_id]

# 无限循环预测
if __name__ == "__main__":
    print("🔍 犯罪事实罪名预测系统")
    print("输入犯罪事实文本，输入 'Stop' 可退出。")
    while True:
        fact = input("\n请输入犯罪事实：\n>>> ")
        if fact.strip().lower() == "stop":
            print("👋 已退出预测系统。")
            break
        result = predict_accusation(fact)
        print(f"✅ 预测罪名：{result}")
