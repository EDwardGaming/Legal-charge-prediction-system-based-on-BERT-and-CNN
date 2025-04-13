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