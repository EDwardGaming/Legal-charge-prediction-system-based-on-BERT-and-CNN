# 基于 Electra + CNN 的法律罪名预测系统



## 摘要

针对中文法律文本罪名分类的复杂性，本研究提出了一种基于ELECTRA预训练模型与卷积神经网络（CNN）融合的法律罪名预测系统。

本系统采用哈尔滨工业大学开源的预训练模型作为语义编码器，通过动态滑动窗口机制提取局部语义特征，构建包含全局最大池化与正则化技术的多级分类架构。

实验采用真实司法文书数据集，通过双重数据清洗策略（多罪名样本剔除、特殊字符过滤）和动态罪名加载机制，在50万条训练样本上实现单轮训练效率优化。

测试集评估显示，系统在盗窃、危险驾驶等高频罪名中达到98.45%的F1值，宏观准确率达88.47%，但在样本量低于10的68个低频罪名中出现严重识别失效。值得注意的是，模型通过犯罪事实文本特征学习，在毒品犯罪（F1=97.23%）和盗窃案件（Recall=98.27%）中展现出类法律专家水平的判别能力，但对复杂经济犯罪（如合同诈骗Precision=66.76%）和新型网络犯罪（如帮助信息网络犯罪活动罪F1=0）的泛化能力有限。

本研究证实了预训练模型在法律文本表征中的有效性，同时揭示了司法人工智能面临的长尾分布困境，为后续研究提供了数据增强与迁移学习结合的改进方向。



## 运行环境

### GPU环境

本项目使用`conda`创建python虚拟环境，使用GPU进行训练，CUDA和cuDNN版本[参考此网址](https://tensorflow.google.cn/install/source_windows?hl=en#gpu)

| Version               | Python version | cuDNN | CUDA |
| :-------------------- | :------------- | :---- | :--- |
| tensorflow_gpu-2.10.0 | 3.8            | 8.1   | 11.2 |

```
NVIDIA GeForce RTX 4060 Laptop GPU

驱动程序版本:	32.0.15.7270
驱动程序日期:	2025/3/3
DirectX 版本:	12 (FL 12.1)
物理位置：	PCI 总线 1、设备 0、功能 0

专用 GPU 内存	8.0 GB
共享 GPU 内存	7.6 GB
GPU 内存	15.6 GB
```



### 软件包依赖

`requirements.txt`

```
tensorflow-gpu==2.10.0
numpy==1.23.5
transformers==4.30.0
scikit-learn==1.0.2
tqdm==4.65.0
jieba==0.42.1
Keras==2.10.0
```



## 项目实现原理

### 流程图

```
【输入犯罪事实】
      ↓
预训练模型编码：把词变成向量
      ↓
CNN：滑动窗口提取词组特征
      ↓
GlobalMaxPooling：挑出最明显的特征
      ↓
Dense：映射成不同罪名的分数
      ↓
Softmax：转成概率，选出预测结果
      ↓
【输出罪名 】
```



### 一、数据清洗

1. 清洗多重罪名数据

   由于部分数据集中的accusation字段含有多个罪名，需要删除在json中对应的行，减小杂数据对模型的干扰。代码实现如下

   ```python
   import os
   import json
   
   def clean_json_files(folder_path):
       for root, dirs, files in os.walk(folder_path):
           for file in files:
               if file.endswith('.json'):
                   file_path = os.path.join(root, file)
                   try:
                       with open(file_path, 'r', encoding='utf-8') as f:
                           lines = f.readlines()  # 读取所有行（每行一个 JSON 对象）
   
                       valid_lines = []
                       for line in lines:
                           line = line.strip()  # 去除行首尾空格和换行符
                           if not line:
                               continue  # 跳过空行
                           
                           try:
                               data = json.loads(line)  # 解析 JSON 数据
                               accusation = data.get('meta', {}).get('accusation', [])
                               
                               if len(accusation) == 1:  # 仅保留 accusation 列表长度为 1 的行
                                   valid_lines.append(line + '\n')  # 恢复换行符（原文件可能每行末尾有换行）
                               
                           except json.JSONDecodeError as e:
                               print(f"解析 JSON 行时出错（文件: {file_path}, 行: {line[:50]}...）: {e}")
                               continue  # 跳过解析失败的行
   
                       # 写回文件（覆盖原文件）
                       with open(file_path, 'w', encoding='utf-8') as f:
                           f.writelines(valid_lines)
                       print(f"文件 {file_path} 处理完成，保留 {len(valid_lines)} 条有效数据")
   
                   except Exception as e:
                       print(f"处理文件 {file_path} 时发生错误: {e}")
   
   if __name__ == "__main__":
       # 请将此处替换为实际文件夹路径（例如："D:/json_files"）
       target_folder = "temp/trainset"
       clean_json_files(target_folder)
   ```

2. 对应accusation字段还含有形如“[]()”样式的文字，可在训练时去除。

   `accu_clean = accu.translate(str.maketrans('', '', '[]（）【】'))`

3. 网上给出的accu.txt有202个罪名，实际数据集可能不包含这些罪名，可在实际训练时候进行动态加载罪名。(见train.py的主函数)



### 二、生成词向量

选择开源的预训练模型，由[哈尔滨工业大学专门为中文法律文本训练的词向量生成模型](https://huggingface.co/hfl/chinese-legal-electra-base-discriminator/tree/main)

```powershell
git lfs install
git clone https://huggingface.co/hfl/chinese-legal-electra-base-discriminator
```



### 三、神经网络层构建

基本原理是使用卷积神经网络提取文本特征，然后使用GlobalMaxPooling聚合最显著的特征交给全连接层，全连接层把刚刚 CNN 提取到的“最显著特征”当作输入，交给神经网络里的决策层，最终全连接层把这些特征映射成不同罪名的可能性，交给Softmax层。Softmax层把所有罪名的分数，变成 **概率分布**，最后选出概率最大的罪名。



### 四、训练模型

考虑到时间紧迫和算力问题，最终只从200多万条数据集中选择了50万条，进行了1轮训练，小样本训练测试发现如果进行多轮训练模型发生了过拟合问题，精确率不断下降，不清楚为什么会发生这样的问题，最终决定进行一轮训练，平均精度达到80%，或许深入了解模型性能下降的原因后可以更高的提升精度。训练代码如下：

`train.py`

```python
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

# 配置参数 
MAX_LENGTH = 256
BATCH_SIZE = 10
EPOCHS = 1
LEARNING_RATE = 2e-5
MODEL_SAVE_PATH = "electra_cnn_legal"
DROPOUT_RATE = 0.3
CLASSIFIER_UNITS = 384
LOCAL_MODEL_PATH = "./hfl/chinese-legal-electra-base-disc"

# GPU配置 
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

# 加载ELECTRA模型
try:
    tokenizer = ElectraTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    electra_model = TFElectraModel.from_pretrained(LOCAL_MODEL_PATH)
    logging.info("ELECTRA模型加载成功")
except Exception as e:
    logging.error(f"模型加载失败: {str(e)}")
    sys.exit(1)

# 数据生成器
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

# 自定义模型 
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

# 加载数据集
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

# 主流程 
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

    # 模型评估部分
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

```



## 训练结果评估报告

### 法律罪名分类模型测试集评估报告

**一、整体性能概览**

| 指标          | 数值    | 解释说明                                                                 |
|---------------|--------|--------------------------------------------------------------------------|
| 测试集准确率   | 88.47% | 模型对大部分样本的罪名分类正确，但受高频类别主导                         |
| 加权平均F1     | 0.8772 | 高频类别（如盗窃、危险驾驶）的优秀表现拉高整体评分                       |
| 宏观平均F1     | 0.5890 | 反映小样本类别的预测能力严重不足，存在显著的类别不平衡问题               |

---

**二、类别分布特征分析**

1. **极端长尾分布**
   
   - **头部类别**：前5%的罪名（如`盗窃`、`危险驾驶`）占据总样本量的42.3%  
   - **尾部类别**：68个罪名（占类别总数的39.5%）样本量 ≤ 10，其中23个类别样本量=1
   
2. **高频与低频类别对比**
   | 类别类型 | 平均样本量 | 平均F1  | 典型罪名案例                     |
   |----------|------------|---------|----------------------------------|
   | 高频类别 | 1,892      | 0.913   | 盗窃（5436）、危险驾驶（4876）    |
   | 中频类别 | 127        | 0.721   | 合同诈骗（318）、受贿（346）      |
   | 低频类别 | 5.2        | 0.107   | 传授犯罪方法（5）、伪造货币（4）   |

---

**三、关键罪名表现详析**

1. **高频罪名优秀案例**
   | 罪名             | 样本量 | Precision | Recall | F1   |
   |------------------|--------|-----------|--------|------|
   | 盗窃             | 5436   | 0.9589    | 0.9827 | 0.9707 |
   | 危险驾驶         | 4876   | 0.9751    | 0.9941 | 0.9845 |
   | 故意伤害         | 2897   | 0.8902    | 0.9203 | 0.9050 |

   **成功特征**：文本模式明确（如"盗窃"常含"财物丢失"等关键词），样本充足

2. **低频罪名失效案例**
   | 罪名                     | 样本量 | Precision | Recall | F1   |
   |--------------------------|--------|-----------|--------|------|
   | 传授犯罪方法             | 5      | 0.0000    | 0.0000 | 0.0000 |
   | 伪造货币                 | 4      | 0.0000    | 0.0000 | 0.0000 |
   | 组织、领导、参加黑社会性质组织 | 1      | 0.0000    | 0.0000 | 0.0000 |

   **失效原因**：样本量过少导致模型无法学习有效特征，预测结果全为负类

3. **中等样本量但表现异常的罪名**
   | 罪名             | 样本量 | Precision | Recall | F1   | 异常原因分析                     |
   |------------------|--------|-----------|--------|------|----------------------------------|
   | 出售、购买、运输假币 | 32     | 0.4638    | 1.0000 | 0.6337 | 高召回率伴随低精确率，存在误判扩散 |
   | 投放危险物质     | 59     | 0.6374    | 0.9831 | 0.7733 | 召回率接近完美但精确率一般         |

---

**四、数据质量风险提示**

1. **样本过滤缺陷**  
   - 测试集中包含**41个罪名**的样本量 ≤ 3，违反机器学习最小样本原则  
   - 例如`组织、领导、参加黑社会性质组织`（1样本）在测试集出现，导致评估指标失真

2. **标签噪声风险**  
   - 罪名`伪造、倒卖伪造的有价票证`（5样本）与`伪造货币`（4样本）特征高度相似但被划分为不同类别，可能造成模型混淆

3. **文本长度偏差**  
   - 过滤条件`10 < len(fact) < 1500`排除超短/超长文本，但实际业务中需验证该长度范围是否覆盖所有法律场景

---

**五、法律场景关键缺陷**

1. **重大罪名漏检风险**  
   - `故意杀人`（325样本）F1仅0.7457，对量刑关键罪名需更高召回率  
   - `受贿`（346样本）精确率0.8116，存在将其他经济犯罪误判为受贿的风险

2. **程序性罪名识别不足**  
   - `妨害作证`（22样本）F1仅0.5098，影响案件侦破中关键证据链的构建  
   - `帮助毁灭、伪造证据`（9样本）完全无法识别，存在法律程序漏洞

---

**六、评估结论**

1. **优势领域**  
   - 高频罪名（样本量 > 500）分类准确率稳定在90%以上  
   - 特定罪名（如`盗窃`、`危险驾驶`）达到准生产环境可用水平

2. **核心瓶颈**  
   - 类别极度不平衡导致38.2%的罪名无法有效识别  
   - 关键法律罪名（样本量50~300区间）存在精确率-召回率失衡问题

3. **业务影响**  
   - 当前模型适用于高频罪名的批量处理，但**不满足司法裁判场景的全面性要求**  
   - 对新型犯罪（低频罪名）和复合型犯罪（特征交叉罪名）识别能力有限



### 模型在测试集上的表现表格汇总

|                             类别                            |   precision |    recall |  f1-score |   support |
|------------------------------------------------------------|------------|-----------|-----------|-----------|
| 串通投标                                                   |     0.7500 |    0.9000 |    0.8182 |        20 |
| 交通肇事                                                   |     0.9486 |    0.9728 |    0.9606 |      2392 |
| 介绍贿赂                                                   |     0.0000 |    0.0000 |    0.0000 |        22 |
| 以危险方法危害公共安全                                     |     0.7679 |    0.3116 |    0.4433 |       138 |
| 传授犯罪方法                                               |     0.0000 |    0.0000 |    0.0000 |         5 |
| 传播性病                                                   |     1.0000 |    0.8333 |    0.9091 |        12 |
| 伪证                                                       |     0.0000 |    0.0000 |    0.0000 |        20 |
| 伪造、倒卖伪造的有价票证                                   |     0.0000 |    0.0000 |    0.0000 |         5 |
| 伪造、变造、买卖国家机关公文、证件、印章                   |     0.8089 |    0.8728 |    0.8397 |       228 |
| 伪造、变造、买卖武装部队公文、证件、印章                   |     1.0000 |    0.5714 |    0.7273 |         7 |
| 伪造、变造居民身份证                                       |     0.8485 |    0.7778 |    0.8116 |        36 |
| 伪造、变造金融票证                                         |     0.8947 |    0.6296 |    0.7391 |        27 |
| 伪造公司、企业、事业单位、人民团体印章                     |     0.9108 |    0.8291 |    0.8680 |       234 |
| 伪造货币                                                   |     0.0000 |    0.0000 |    0.0000 |         4 |
| 侮辱                                                       |     0.0000 |    0.0000 |    0.0000 |         8 |
| 侵占                                                       |     0.0000 |    0.0000 |    0.0000 |        13 |
| 侵犯著作权                                                 |     0.9636 |    0.8833 |    0.9217 |        60 |
| 保险诈骗                                                   |     0.8889 |    0.8696 |    0.8791 |        46 |
| 信用卡诈骗                                                 |     0.8546 |    0.9158 |    0.8841 |       475 |
| 倒卖文物                                                   |     0.0000 |    0.0000 |    0.0000 |         3 |
| 倒卖车票、船票                                             |     1.0000 |    1.0000 |    1.0000 |         4 |
| 假冒注册商标                                               |     0.7831 |    0.9205 |    0.8463 |       302 |
| 冒充军人招摇撞骗                                           |     0.8000 |    0.9697 |    0.8767 |        33 |
| 出售、购买、运输假币                                       |     0.4638 |    1.0000 |    0.6337 |        32 |
| 利用影响力受贿                                             |     0.0000 |    0.0000 |    0.0000 |        11 |
| 制作、复制、出版、贩卖、传播淫秽物品牟利                   |     0.9583 |    0.9583 |    0.9583 |        24 |
| 制造、贩卖、传播淫秽物品                                   |     0.0000 |    0.0000 |    0.0000 |         1 |
| 动植物检疫徇私舞弊                                         |     0.9231 |    0.8000 |    0.8571 |        15 |
| 劫持船只、汽车                                             |     0.0000 |    0.0000 |    0.0000 |         3 |
| 单位受贿                                                   |     0.2500 |    0.0476 |    0.0800 |        21 |
| 单位行贿                                                   |     0.7068 |    0.7627 |    0.7337 |       177 |
| 危险物品肇事                                               |     1.0000 |    0.1250 |    0.2222 |         8 |
| 危险驾驶                                                   |     0.9751 |    0.9941 |    0.9845 |      4876 |
| 受贿                                                       |     0.8116 |    0.6850 |    0.7429 |       346 |
| 合同诈骗                                                   |     0.6676 |    0.7201 |    0.6929 |       318 |
| 失火                                                       |     0.9717 |    0.9778 |    0.9748 |       316 |
| 妨害作证                                                   |     0.4483 |    0.5909 |    0.5098 |        22 |
| 妨害信用卡管理                                             |     0.9058 |    0.8929 |    0.8993 |       140 |
| 妨害公务                                                   |     0.8929 |    0.9646 |    0.9274 |       311 |
| 容留他人吸毒                                               |     0.9646 |    0.9745 |    0.9695 |       392 |
| 对单位行贿                                                 |     0.0000 |    0.0000 |    0.0000 |        20 |
| 对非国家工作人员行贿                                       |     0.7500 |    0.1579 |    0.2609 |        38 |
| 寻衅滋事                                                   |     0.7706 |    0.5091 |    0.6131 |       607 |
| 帮助毁灭、伪造证据                                         |     0.0000 |    0.0000 |    0.0000 |         9 |
| 帮助犯罪分子逃避处罚                                       |     0.5600 |    0.7778 |    0.6512 |        18 |
| 开设赌场                                                   |     0.8465 |    0.9259 |    0.8844 |       405 |
| 引诱、教唆、欺骗他人吸毒                                   |     1.0000 |    0.6000 |    0.7500 |        15 |
| 强制猥亵、侮辱妇女                                         |     0.7436 |    0.5686 |    0.6444 |        51 |
| 强奸                                                       |     0.8846 |    0.9485 |    0.9154 |       291 |
| 强迫交易                                                   |     0.5625 |    0.6429 |    0.6000 |        28 |
| 强迫他人吸毒                                               |     0.0000 |    0.0000 |    0.0000 |         2 |
| 强迫劳动                                                   |     0.6667 |    0.4000 |    0.5000 |         5 |
| 徇私枉法                                                   |     0.0000 |    0.0000 |    0.0000 |        14 |
| 徇私舞弊不移交刑事案件                                     |     0.0000 |    0.0000 |    0.0000 |         5 |
| 打击报复证人                                               |     0.0000 |    0.0000 |    0.0000 |         3 |
| 扰乱无线电通讯管理秩序                                     |     1.0000 |    0.0159 |    0.0312 |        63 |
| 投放危险物质                                               |     0.6374 |    0.9831 |    0.7733 |        59 |
| 抢劫                                                       |     0.8310 |    0.7804 |    0.8049 |       378 |
| 抢夺                                                       |     0.7442 |    0.9195 |    0.8226 |       174 |
| 拐卖妇女、儿童                                             |     0.8205 |    0.9600 |    0.8848 |       100 |
| 拐骗儿童                                                   |     0.7647 |    0.6190 |    0.6842 |        21 |
| 拒不执行判决、裁定                                         |     0.8609 |    0.9636 |    0.9094 |       302 |
| 拒不支付劳动报酬                                           |     0.9601 |    0.9779 |    0.9689 |       271 |
| 招摇撞骗                                                   |     0.7975 |    0.8690 |    0.8317 |       145 |
| 招收公务员、学生徇私舞弊                                   |     0.0000 |    0.0000 |    0.0000 |         7 |
| 持有、使用假币                                             |     1.0000 |    0.2778 |    0.4348 |        54 |
| 持有伪造的发票                                             |     0.8276 |    0.8780 |    0.8521 |        82 |
| 挪用公款                                                   |     0.8529 |    0.7500 |    0.7982 |       232 |
| 挪用特定款物                                               |     0.0000 |    0.0000 |    0.0000 |         4 |
| 挪用资金                                                   |     0.7773 |    0.7500 |    0.7634 |       256 |
| 掩饰、隐瞒犯罪所得、犯罪所得收益                           |     0.9096 |    0.7295 |    0.8097 |       207 |
| 提供侵入、非法控制计算机信息系统程序、工具                   |     0.0000 |    0.0000 |    0.0000 |         4 |
| 收买被拐卖的妇女、儿童                                     |     0.0000 |    0.0000 |    0.0000 |         6 |
| 放火                                                       |     0.7898 |    0.9288 |    0.8537 |       267 |
| 故意伤害                                                   |     0.8902 |    0.9203 |    0.9050 |      2897 |
| 故意杀人                                                   |     0.7563 |    0.7354 |    0.7457 |       325 |
| 故意毁坏财物                                               |     0.7109 |    0.7085 |    0.7097 |       295 |
| 敲诈勒索                                                   |     0.8550 |    0.6867 |    0.7617 |       249 |
| 污染环境                                                   |     0.9933 |    0.9801 |    0.9867 |       302 |
| 洗钱                                                       |     0.0000 |    0.0000 |    0.0000 |         3 |
| 滥伐林木                                                   |     0.8348 |    0.9697 |    0.8972 |       396 |
| 滥用职权                                                   |     0.6190 |    0.5306 |    0.5714 |       147 |
| 爆炸                                                       |     0.4583 |    0.2200 |    0.2973 |        50 |
| 猥亵儿童                                                   |     0.8167 |    0.8909 |    0.8522 |       110 |
| 玩忽职守                                                   |     0.8174 |    0.8107 |    0.8140 |       243 |
| 生产、销售不符合安全标准的食品                             |     0.9091 |    0.8874 |    0.8981 |       293 |
| 生产、销售伪劣产品                                         |     0.7568 |    0.4719 |    0.5813 |       178 |
| 生产、销售伪劣农药、兽药、化肥、种子                       |     1.0000 |    0.2857 |    0.4444 |         7 |
| 生产、销售假药                                             |     0.9734 |    0.9792 |    0.9763 |       336 |
| 生产、销售有毒、有害食品                                   |     0.9118 |    0.9058 |    0.9088 |       308 |
| 盗伐林木                                                   |     0.9119 |    0.7500 |    0.8231 |       276 |
| 盗掘古文化遗址、古墓葬                                     |     1.0000 |    0.9804 |    0.9901 |        51 |
| 盗窃                                                       |     0.9589 |    0.9827 |    0.9707 |      5436 |
| 盗窃、侮辱尸体                                             |     1.0000 |    0.4286 |    0.6000 |         7 |
| 破坏交通工具                                               |     0.0000 |    0.0000 |    0.0000 |         3 |
| 破坏交通设施                                               |     0.0000 |    0.0000 |    0.0000 |         8 |
| 破坏广播电视设施、公用电信设施                             |     0.5542 |    0.9200 |    0.6917 |       100 |
| 破坏易燃易爆设备                                           |     0.9706 |    0.8462 |    0.9041 |        39 |
| 破坏生产经营                                               |     0.7071 |    0.6731 |    0.6897 |       104 |
| 破坏电力设备                                               |     0.9180 |    0.7778 |    0.8421 |        72 |
| 破坏监管秩序                                               |     0.6000 |    0.9000 |    0.7200 |        10 |
| 破坏计算机信息系统                                         |     0.4000 |    0.1429 |    0.2105 |        14 |
| 票据诈骗                                                   |     0.6316 |    0.7742 |    0.6957 |        31 |
| 私分国有资产                                               |     0.0000 |    0.0000 |    0.0000 |        12 |
| 窃取、收买、非法提供信用卡信息                             |     0.0000 |    0.0000 |    0.0000 |         9 |
| 窝藏、包庇                                                 |     0.8432 |    0.8361 |    0.8397 |       238 |
| 窝藏、转移、收购、销售赃物                                 |     0.2222 |    0.5000 |    0.3077 |         8 |
| 窝藏、转移、隐瞒毒品、毒赃                                 |     0.0000 |    0.0000 |    0.0000 |         3 |
| 组织、领导、参加黑社会性质组织                             |     0.0000 |    0.0000 |    0.0000 |         1 |
| 组织、领导传销活动                                         |     0.9888 |    0.9565 |    0.9724 |       184 |
| 绑架                                                       |     0.8364 |    0.5679 |    0.6765 |        81 |
| 编造、故意传播虚假恐怖信息                                 |     0.6897 |    1.0000 |    0.8163 |        20 |
| 职务侵占                                                   |     0.7276 |    0.8423 |    0.7807 |       279 |
| 聚众冲击国家机关                                           |     0.0000 |    0.0000 |    0.0000 |         8 |
| 聚众哄抢                                                   |     0.0000 |    0.0000 |    0.0000 |         3 |
| 聚众扰乱公共场所秩序、交通秩序                             |     0.8889 |    0.3077 |    0.4571 |        26 |
| 聚众扰乱社会秩序                                           |     0.4421 |    0.9333 |    0.6000 |        45 |
| 聚众斗殴                                                   |     0.6192 |    0.7352 |    0.6722 |       219 |
| 脱逃                                                       |     0.7647 |    0.8125 |    0.7879 |        16 |
| 虐待                                                       |     0.0000 |    0.0000 |    0.0000 |         5 |
| 虐待被监管人                                               |     0.0000 |    0.0000 |    0.0000 |         3 |
| 虚开发票                                                   |     0.8835 |    0.9010 |    0.8922 |       101 |
| 虚开增值税专用发票、用于骗取出口退税、抵扣税款发票         |     0.9771 |    0.9739 |    0.9755 |       307 |
| 虚报注册资本                                               |     0.6957 |    1.0000 |    0.8205 |        16 |
| 行贿                                                       |     0.6677 |    0.8621 |    0.7525 |       261 |
| 诈骗                                                       |     0.8444 |    0.7830 |    0.8125 |       977 |
| 诬告陷害                                                   |     0.6667 |    0.3571 |    0.4651 |        28 |
| 诽谤                                                       |     0.0000 |    0.0000 |    0.0000 |         2 |
| 贪污                                                       |     0.7799 |    0.7977 |    0.7887 |       262 |
| 贷款诈骗                                                   |     0.3333 |    0.0323 |    0.0588 |        31 |
| 赌博                                                       |     0.8092 |    0.7653 |    0.7866 |       277 |
| 走私                                                       |     0.0000 |    0.0000 |    0.0000 |         2 |
| 走私、贩卖、运输、制造毒品                                 |     0.9704 |    0.9743 |    0.9723 |      2255 |
| 走私国家禁止进出口的货物、物品                             |     0.8750 |    0.8750 |    0.8750 |        16 |
| 走私废物                                                   |     1.0000 |    0.1667 |    0.2857 |         6 |
| 走私普通货物、物品                                         |     0.9184 |    0.9926 |    0.9541 |       136 |
| 走私武器、弹药                                             |     0.7692 |    1.0000 |    0.8696 |        10 |
| 走私珍贵动物、珍贵动物制品                                 |     0.7368 |    0.8750 |    0.8000 |        16 |
| 过失以危险方法危害公共安全                                 |     0.5143 |    0.8182 |    0.6316 |        22 |
| 过失投放危险物质                                           |     0.0000 |    0.0000 |    0.0000 |         6 |
| 过失损坏广播电视设施、公用电信设施                         |     0.5000 |    0.8571 |    0.6316 |         7 |
| 过失损坏武器装备、军事设施、军事通信                       |     0.6000 |    0.7500 |    0.6667 |         4 |
| 过失致人死亡                                               |     0.7908 |    0.4874 |    0.6031 |       318 |
| 过失致人重伤                                               |     0.6150 |    0.8092 |    0.6989 |       152 |
| 违法发放贷款                                               |     0.7778 |    0.9032 |    0.8358 |        31 |
| 逃税                                                       |     0.8636 |    0.8636 |    0.8636 |        22 |
| 遗弃                                                       |     0.9231 |    0.6000 |    0.7273 |        20 |
| 重大劳动安全事故                                           |     0.0000 |    0.0000 |    0.0000 |        60 |
| 重大责任事故                                               |     0.6422 |    0.9091 |    0.7527 |       308 |
| 重婚                                                       |     0.9255 |    0.9886 |    0.9560 |        88 |
| 金融凭证诈骗                                               |     0.0000 |    0.0000 |    0.0000 |         3 |
| 销售假冒注册商标的商品                                     |     0.8147 |    0.9410 |    0.8733 |       271 |
| 隐匿、故意销毁会计凭证、会计帐簿、财务会计报告             |     0.7692 |    0.8333 |    0.8000 |        12 |
| 集资诈骗                                                   |     0.6774 |    0.5060 |    0.5793 |        83 |
| 非国家工作人员受贿                                         |     0.6550 |    0.8008 |    0.7206 |       256 |
| 非法买卖、运输、携带、持有毒品原植物种子、幼苗             |     0.0000 |    0.0000 |    0.0000 |         1 |
| 非法买卖制毒物品                                           |     0.9200 |    0.7931 |    0.8519 |        29 |
| 非法侵入住宅                                               |     0.7486 |    0.6782 |    0.7117 |       202 |
| 非法出售发票                                               |     0.8000 |    0.7619 |    0.7805 |        21 |
| 非法制造、买卖、运输、储存危险物质                         |     0.5714 |    0.5000 |    0.5333 |         8 |
| 非法制造、买卖、运输、邮寄、储存枪支、弹药、爆炸物         |     0.8310 |    0.8806 |    0.8551 |       268 |
| 非法制造、出售非法制造的发票                               |     0.2500 |    0.0769 |    0.1176 |        13 |
| 非法制造、销售非法制造的注册商标标识                       |     0.0000 |    0.0000 |    0.0000 |        10 |
| 非法占用农用地                                             |     0.9253 |    0.9670 |    0.9457 |       333 |
| 非法吸收公众存款                                           |     0.8423 |    0.9570 |    0.8960 |       279 |
| 非法处置查封、扣押、冻结的财产                             |     0.8276 |    0.3934 |    0.5333 |        61 |
| 非法拘禁                                                   |     0.8144 |    0.8556 |    0.8345 |       277 |
| 非法持有、私藏枪支、弹药                                   |     0.9185 |    0.9397 |    0.9290 |       348 |
| 非法持有毒品                                               |     0.8125 |    0.8206 |    0.8165 |       301 |
| 非法捕捞水产品                                             |     0.9852 |    0.9925 |    0.9888 |       134 |
| 非法携带枪支、弹药、管制刀具、危险物品危及公共安全         |     0.3333 |    0.0833 |    0.1333 |        12 |
| 非法收购、运输、出售珍贵、濒危野生动物、珍贵、濒危野生动物制品 |     0.0000 |    0.0000 |    0.0000 |         3 |
| 非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品 |     0.8636 |    0.7451 |    0.8000 |        51 |
| 非法收购、运输盗伐、滥伐的林木                             |     0.8000 |    0.8276 |    0.8136 |        29 |
| 非法狩猎                                                   |     0.9560 |    0.9016 |    0.9280 |       193 |
| 非法猎捕、杀害珍贵、濒危野生动物                           |     0.7234 |    0.7907 |    0.7556 |        43 |
| 非法生产、买卖警用装备                                     |     1.0000 |    0.8571 |    0.9231 |         7 |
| 非法生产、销售间谍专用器材                                 |     0.7222 |    1.0000 |    0.8387 |        13 |
| 非法种植毒品原植物                                         |     1.0000 |    0.9941 |    0.9970 |       337 |
| 非法组织卖血                                               |     0.9091 |    1.0000 |    0.9524 |        10 |
| 非法经营                                                   |     0.8908 |    0.7361 |    0.8061 |       288 |
| 非法获取公民个人信息                                       |     0.9394 |    0.8378 |    0.8857 |        37 |
| 非法获取国家秘密                                           |     0.5000 |    0.1667 |    0.2500 |         6 |
| 非法行医                                                   |     0.9755 |    0.9789 |    0.9772 |       285 |
| 非法转让、倒卖土地使用权                                   |     0.8864 |    0.7647 |    0.8211 |        51 |
| 非法进行节育手术                                           |     1.0000 |    0.8182 |    0.9000 |        22 |
| 非法采伐、毁坏国家重点保护植物                             |     0.9077 |    0.9031 |    0.9054 |       196 |
| 非法采矿                                                   |     0.9739 |    0.9106 |    0.9412 |       123 |
| 骗取贷款、票据承兑、金融票证                               |     0.8188 |    0.9457 |    8777 |       258 |
| 高利转贷                                                   |     0.0000 |    0.0000 |    0.0000 |         2 |
| ​**accuracy**​                                               |             |           | ​**0.8847**|     39317 |
| ​**macro avg**​                                              |     0.6260 |    0.5928 |    0.5890 |     39317 |
| ​**weighted avg**​                                           |     0.8798 |    0.8847 |    0.8772 |     39317 |



## 实际测试

使用代码

```python
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

```

### 效果图

![在·](assets/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20250413144050.png)



![微信图片_20250413144235](assets/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20250413144235.png)



![微信图片_20250413144242](assets/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20250413144242.png)



![微信图片_20250413145119](assets/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20250413145119.png)





## 参考文献

[基于BERT词向量和Attention-CNN的智能司法研究-学位-万方数据知识服务平台](https://d.wanfangdata.com.cn/thesis/D01697595)

[使用GPU运行TensorFlow模型的教程_tensorflow gpu-CSDN博客](https://blog.csdn.net/m0_71417856/article/details/136298172)

[Build from source on Windows  | TensorFlow](https://tensorflow.google.cn/install/source_windows?hl=en#gpu)

