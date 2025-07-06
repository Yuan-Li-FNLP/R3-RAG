from transformers import AutoTokenizer, AutoModelForCausalLM
import pdb

def tokens_to_text():
    # 1. 指定模型名称或本地路径
    model_name_or_path = "/remote-home1/yli/Model/Generator/Qwen2.5/7B/base"  # 替换为实际的模型名称或本地路径
    
    # 2. 加载分词器和模型
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
    
    # 3. 准备示例数据（仅取出其中的 input_ids 进行演示）
    example_data = [
        {
            'input_ids': [151644, 8948, 198, 2610, 525, 264, 10950, 17847, 13, 151645, 198, 151644, 872, 198, 31115, 264,
                          1140, 315, 4236, 2155, 6467, 911, 8038, 13, 151645, 198, 151644, 77091, 198, 16, 13, 362,
                          36518, 11099, 315, 4120, 553, 18095, 12611, 10566, 198, 17, 13, 576, 10115, 812, 24022, 553,
                          11867, 36751, 11704, 198, 18, 13, 576, 32305, 11278, 64, 27636, 4337, 25, 9965, 438, 264,
                          72897, 304, 279, 12258, 553, 21998, 328, 18345, 198, 19, 13, 576, 28596, 315, 37030, 10107,
                          20201, 553, 11108, 328, 13, 730, 92164, 198, 20, 13, 576, 7093, 15806, 941, 25, 362, 19207,
                          8615, 315, 279, 38093, 315, 279, 28596, 315, 15552, 553, 7801, 422, 13, 31480, 13, 151645,
                          198],
            'attention_mask': [1]*109,  # 这里示例化，真实情况同长度即可
            'labels': [-100]*29 + [16, 13, 362, 36518, 11099, 315, 4120, 553, 18095, 12611, 10566, 198, 17, 13, 576,
                       10115, 812, 24022, 553, 11867, 36751, 11704, 198, 18, 13, 576, 32305, 11278, 64, 27636, 4337,
                       25, 9965, 438, 264, 72897, 304, 279, 12258, 553, 21998, 328, 18345, 198, 19, 13, 576, 28596,
                       315, 37030, 10107, 20201, 553, 11108, 328, 13, 730, 92164, 198, 20, 13, 576, 7093, 15806, 941,
                       25, 362, 19207, 8615, 315, 279, 38093, 315, 279, 28596, 315, 15552, 553, 7801, 422, 13, 31480,
                       13, 151645, 198],
            'images': None,
            'videos': None,
        }
    ]
    texts = ""
    # 4. 取出 input_ids 并使用 tokenizer.decode 转换为可读文本
    input_ids = example_data[0]['input_ids']
    decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)
    
    print("解码后的文本：")
    print(decoded_text)
    print(tokens)
    pdb.set_trace()

if __name__ == "__main__":
    tokens_to_text()
