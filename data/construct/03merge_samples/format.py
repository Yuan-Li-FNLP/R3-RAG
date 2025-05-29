# import shutil

# def convert_to_standard_jsonl(input_file, output_file):
#     no_change_flag=True
#     current_lines = []
#     with open(input_file, 'r', encoding='utf-8') as f, \
#          open(output_file, 'w', encoding='utf-8') as out_f:
#         for line in f:
#             # import pdb
#             # pdb.set_trace()
#             if line == '{\n':  # 开始新的样例
#                 current_lines = ['{']
#                 no_change_flag=False
#             elif line == '}\n':  # 样例结束
#                 current_lines.append('}')
#                 # 把所有行合并成一行，去掉中间的换行和多余空格
#                 one_line = ''.join(line.strip() for line in current_lines)
#                 out_f.write(one_line + '\n')
#                 current_lines = []
#             else:
#                 if current_lines:  # 如果已经开始收集
#                     current_lines.append(line.strip())
#     if no_change_flag:
#         shutil.copy(input_file, output_file)

# # 使用示例
# input_file = '/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/2wikimultihopqa/train/merge.jsonl'
# output_file = '/remote-home1/yli/Workspace/R3RAG/data/construct/mainv9/merge/2wikimultihopqa/train/merge1.jsonl'
# convert_to_standard_jsonl(input_file, output_file)

import shutil
import sys

def convert_to_standard_jsonl(input_file, output_file):
    no_change_flag = True
    current_lines = []

    # 打开输入文件进行读取，输出文件进行写入
    with open(input_file, 'r', encoding='utf-8') as f, open(output_file, 'w', encoding='utf-8') as out_f:
        for line in f:
            if line == '{\n':  # 开始新的样例
                current_lines = ['{']
                no_change_flag = False
            elif line == '}\n':  # 样例结束
                current_lines.append('}')
                # 把所有行合并成一行，去掉中间的换行和多余空格
                one_line = ''.join(line.strip() for line in current_lines)
                out_f.write(one_line + '\n')
                current_lines = []
            else:
                if current_lines:  # 如果已经开始收集
                    current_lines.append(line.strip())

        # 如果没有做任何更改，直接复制原始文件
        if no_change_flag:
            shutil.copy(input_file, output_file)

if __name__ == "__main__":
    # 获取命令行参数：输入文件路径和输出文件路径
    if len(sys.argv) != 3:
        print("Usage: python format_jsonl.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    # 调用格式化函数
    convert_to_standard_jsonl(input_file, output_file)
