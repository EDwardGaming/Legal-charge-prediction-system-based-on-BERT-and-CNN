import json
import os
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_ENABLE_CUBLAS_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_ENABLE_CUDNN_TENSOR_OP_MATH_FP32'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import sys
import logging
import numpy as np
import tensorflow as tf
from transformers import ElectraTokenizer, TFElectraModel
from tensorflow.keras import layers, Model
from tensorflow.keras import callbacks as keras_callbacks
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

# ------------------ 配置参数 ------------------
MAX_LENGTH = 256
BATCH_SIZE = 10
EPOCHS = 1
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = "electra_cnn_legal"
DROPOUT_RATE = 0.3
CLASSIFIER_UNITS = 384
LOCAL_MODEL_PATH = "./hfl/chinese-legal-electra-base-disc"

# ------------------ GPU配置 ------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=7 * 1024)]
        )
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        logging.warning(f"GPU配置提示: {str(e)}")

# ------------------ 加载ELECTRA模型 ------------------
try:
    tokenizer = ElectraTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    electra_model = TFElectraModel.from_pretrained(LOCAL_MODEL_PATH)
    logging.info("ELECTRA模型加载成功")
except Exception as e:
    logging.error(f"模型加载失败: {str(e)}")
    sys.exit(1)

# ------------------ 数据生成器 ------------------
class LegalDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, texts, labels, label_map, batch_size=BATCH_SIZE):
        self.texts = texts
        self.labels = np.array([label_map[l] for l in labels], dtype=np.int32)
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.texts) / self.batch_size))

    def __getitem__(self, idx):
        batch_texts = self.texts[idx * self.batch_size:(idx + 1) * self.batch_size]
        tokenized = tokenizer(
            batch_texts,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        return (tokenized["input_ids"], tokenized["attention_mask"]), self.labels[idx * self.batch_size:(idx + 1) * self.batch_size]

# ------------------ 自定义模型 ------------------
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

# ------------------ 加载数据集 ------------------
def load_dataset(data_dirs, label_map, min_samples_per_class=2):
    texts, labels = [], []
    label_counts = {label: 0 for label in label_map}

    for data_dir in data_dirs:
        if not os.path.isdir(data_dir):
            continue
        for root, _, files in os.walk(data_dir):
            for file in files:
                if not file.endswith('.json'):
                    continue
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            accu = data['meta']['accusation'][0]
                            accu_clean = accu.translate(str.maketrans('', '', '[]（）【】'))
                            if accu_clean in label_map and 10 < len(data['fact']) < 1500:
                                texts.append(data['fact'].strip())
                                labels.append(accu_clean)
                                label_counts[accu_clean] += 1
                except Exception as e:
                    logging.error(f"文件处理错误: {str(e)}")

    # 过滤掉样本数不足的类别
    texts = [text for i, text in enumerate(texts) if label_counts[labels[i]] >= min_samples_per_class]
    labels = [label for label in labels if label_counts[label] >= min_samples_per_class]

    logging.info(f"过滤后数据 - 总样本数: {len(texts)}")
    return texts, labels

# ------------------ 主流程 ------------------
def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('training.log'), logging.StreamHandler()]
    )

    data_dirs = ["trainset"]
    try:
        # 第一步：初步读取全部罪名，构建计数器
        all_texts, all_labels = [], []
        label_counter = {}

        for data_dir in data_dirs:
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if not file.endswith('.json'):
                        continue
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            for line in f:
                                data = json.loads(line)
                                fact = data.get("fact", "").strip()
                                if not fact or not (10 < len(fact) < 1500):
                                    continue
                                accu_list = data["meta"].get("accusation", [])
                                if not accu_list:
                                    continue
                                accu = accu_list[0]
                                accu_clean = accu.translate(str.maketrans('', '', '[]（）【】'))
                                all_texts.append(fact)
                                all_labels.append(accu_clean)
                                label_counter[accu_clean] = label_counter.get(accu_clean, 0) + 1
                    except Exception as e:
                        logging.error(f"数据读取错误: {str(e)}")

        # 第二步：筛选出现次数 >= 2 的罪名，构建新的 label_to_id 映射
        valid_labels = {label for label, count in label_counter.items() if count >= 2}
        label_to_id = {label: idx for idx, label in enumerate(sorted(valid_labels))}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        logging.info(f"有效罪名共计: {len(label_to_id)} 类")

        # 保存实际加载的罪名映射标签，方便模型预测时调用。
        with open("accu_indeed.txt", "w", encoding="utf-8") as f:
            for key in label_to_id:
                f.write(key + "\n")
        f.close()

        # 第三步：再次过滤数据，只保留有效罪名的样本
        texts = [t for i, t in enumerate(all_texts) if all_labels[i] in valid_labels]
        labels = [l for l in all_labels if l in valid_labels]

        # 第一层划分：train+val 与 test（测试集不参与任何训练过程）
        train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
            texts, labels, test_size=0.10, stratify=labels, random_state=42
        )

        # 第二层划分：从 train_val 中再划分出验证集
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_val_texts, train_val_labels, test_size=0.15, stratify=train_val_labels, random_state=42
        )

        logging.info(f"训练样本: {len(train_texts)}, 验证样本: {len(val_texts)}, 测试样本: {len(test_texts)}")
    except Exception as e:
        logging.error(f"数据预处理失败: {str(e)}")
        sys.exit(1)

    # 模型构建和训练不变
    model = LegalClassifier(len(label_to_id))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        keras_callbacks.EarlyStopping(patience=3, monitor='val_accuracy'),
        keras_callbacks.ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True, save_format='tf')
    ]

    try:
        model.fit(
            LegalDataGenerator(train_texts, train_labels, label_to_id),
            validation_data=LegalDataGenerator(val_texts, val_labels, label_to_id),
            epochs=EPOCHS,
            callbacks=callbacks,
            verbose=1
        )
        tf.keras.models.save_model(model, MODEL_SAVE_PATH)
    except Exception as e:
        logging.error(f"训练失败: {str(e)}")
        sys.exit(1)

    # 模型评估
    try:
        model = tf.keras.models.load_model(
            MODEL_SAVE_PATH,
            custom_objects={"LegalClassifier": LegalClassifier}
        )
        y_true, y_pred = [], []
        test_gen = LegalDataGenerator(test_texts, test_labels, label_to_id)

        for (inputs, labels) in test_gen:
            preds = model.predict(inputs)
            y_true.extend(labels)
            y_pred.extend(np.argmax(preds, axis=1))

        print(classification_report(
            y_true, y_pred,
            target_names=[id_to_label[i] for i in sorted(set(y_true))],  # ✅ 只评估实际出现过的类别
            digits=4
        ))
    except Exception as e:
        logging.error(f"评估失败: {str(e)}")

if __name__ == "__main__":
    main()
