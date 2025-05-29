import gradio as gr
import argparse
import re
import requests
import random
import json
import socket
from typing import List, Dict, Any
from vllm import LLM, SamplingParams
from openai import OpenAI

def get_local_ip():
    """获取本机IP地址"""
    try:
        # 创建一个socket连接来获取本地IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        try:
            # 备用方法
            hostname = socket.gethostname()
            ip = socket.gethostbyname(hostname)
            return ip
        except Exception:
            return "127.0.0.1"

# 将您原有的函数都保留，这里只显示主要部分
def mystrip(one_str):
    one_str = one_str.strip()
    one_str = re.sub(r'^(\\n)+', '', one_str)
    one_str = re.sub(r'(\\n)+$', '', one_str)  
    one_str = one_str.strip("#")  
    return one_str

def extract_substring2(text, start_str, stop_strs):
    start_index = text.find(start_str)
    if start_index == -1:
        return None
    start = start_index + len(start_str)
    end = len(text)
    
    for stop_str in stop_strs:
        temp_index = text.find(stop_str, start)
        if temp_index != -1 and temp_index < end:
            end = temp_index
    
    if start < end:
        if start_str=="The retrieval query:" and end < len(text):
            if text[end]=='?':
                return mystrip(text[start:(end+1)])
            else:
                return mystrip(text[start:(end)])
        else:
            return mystrip(text[start:(end)])
    else:
        return None

def split_response(response):
    mydict = {"original": response}
    str_analysis = "The problem analysis:"
    str_query = "The retrieval query:"
    str_answer = "The final answer:"
    stop_strs = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####"]
    stop_strs_query = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep", "?"]
    stop_strs_answer = [str_analysis, str_query, str_answer, "The retrieval documents:", "###", "####", "\nStep"]
    
    start_index = response.find(str_analysis)
    if start_index==-1:    
        mydict['analysis']=None
        return mydict
    else:
        mydict["analysis"]=extract_substring2(response, str_analysis, stop_strs)
    
    start_index_query = response.find(str_query, start_index+len(str_analysis))
    start_index_answer = response.find(str_answer, start_index+len(str_analysis))
    
    if start_index_query==-1 and start_index_answer==-1:
        mydict['analysis']=None
        return mydict
    elif start_index_query!=-1 and start_index_answer!=-1:
        if start_index_query<start_index_answer:
            mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
        else:
            mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    elif start_index_query!=-1:
        mydict['query']=extract_substring2(response[start_index_query:], str_query, stop_strs_query)
    elif start_index_answer!=-1:
        mydict['answer']=extract_substring2(response[start_index_answer:], str_answer, stop_strs_answer)
    
    return mydict

def GetRetrieval(query, retriver_url):
    payload = {"query": query}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(retriver_url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Retrieval error {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"Retrieval request failed: {e}")
        return None

def GetStepbystepRetrieval(mydict, retrieved_ids, documents, config):
    doc = []
    attempts_retrieval = 3
    for one_attempt_retrieval in range(attempts_retrieval):
        retrieval_results = GetRetrieval(mydict.get("query"), config['retriver_url'])
        if retrieval_results is not None:
            break
    
    if retrieval_results is None:
        mydict["doc"] = ""
        return
    
    for i, result in enumerate(retrieval_results):
        if len(retrieved_ids) >= config['max_num_passages']:
            break
        if i >= config['num_passages_one_retrieval']:
            break
        if result['id'] not in retrieved_ids:
            retrieved_ids.append(result['id'])
            documents.append(result['contents'].split('\n')[-1])
            doc.append(result['contents'])
    
    mydict["doc"] = '\n'.join(doc)

class VisualizationChatSession:
    def __init__(self, llm, sampling_params, config):
        self.llm = llm
        self.sampling_params = sampling_params
        self.config = config
        self.reset_session()

    def reset_session(self):
        self.input = ""
        self.output = ""
        self.search_chain = []
        self.retrieved_ids = []
        self.documents = []
        self.step_history = []

    def get_answer_with_visualization(self, question, search_chain_length=10):
        self.reset_session()
        self.input = f"The question: {question}"
        self.output = ""
        length = 0
        
        # 用于存储每一步的可视化数据
        visualization_steps = []
        
        while True:
            if length >= search_chain_length:
                break
            else:
                self.input = self.input + self.output
                conversation = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": self.input}
                ]
                
                outputs = self.llm.chat(
                    messages=conversation,
                    sampling_params=self.sampling_params
                )
                
                generated_text = outputs[0].outputs[0].text
                mydict = split_response(generated_text)
                
                # 创建当前步骤的可视化数据
                step_data = {
                    "step": length + 1,
                    "analysis": mydict.get('analysis', ''),
                    "query": mydict.get('query', ''),
                    "retrieved_docs": '',
                    "answer": mydict.get('answer', ''),
                    "is_final": False
                }
                
                if mydict.get('answer'):
                    step_data["is_final"] = True
                    step_data["answer"] = mydict.get('answer')
                    visualization_steps.append(step_data)
                    self.search_chain.append(mydict)
                    length += 1
                    self.output = self.output + generated_text
                    break
                    
                elif mydict.get('query'):
                    GetStepbystepRetrieval(mydict, self.retrieved_ids, self.documents, self.config)
                    step_data["retrieved_docs"] = mydict.get('doc', '')
                    visualization_steps.append(step_data)
                    
                    self.search_chain.append(mydict)
                    length += 1
                    self.output = self.output + generated_text + "\n" + f"The retrieval documents: {mydict['doc']}" + "\n"
        
        return visualization_steps, self.output

# def create_step_html(step_data):
#     """创建单个步骤的HTML显示"""
#     step_num = step_data["step"]
#     analysis = step_data["analysis"]
#     query = step_data["query"]
#     docs = step_data["retrieved_docs"]
#     answer = step_data["answer"]
#     is_final = step_data["is_final"]
    
#     if is_final:
#         html = f"""
#         <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #d4edda; font-family: 'SimHei', 'Times New Roman', serif;">
#             <h3 style="color: #155724; margin-top: 0; font-family: 'SimHei', Arial, sans-serif;">🎯 Step {step_num} - Final Answer</h3>
#             {f'<div style="margin: 10px 0;"><strong style="font-family: \'SimHei\', Arial, sans-serif;">Analysis:</strong><br><p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: \'Times New Roman\', serif;">{analysis}</p></div>' if analysis else ''}
#             <div style="margin: 10px 0;">
#                 <strong style="font-family: 'SimHei', Arial, sans-serif;">Final Answer:</strong>
#                 <p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: bold; font-family: 'Times New Roman', serif;">{answer}</p>
#             </div>
#         </div>
#         """
#     else:
#         html = f"""
#         <div style="border: 2px solid #007bff; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #e3f2fd; font-family: 'SimHei', 'Times New Roman', serif;">
#             <h3 style="color: #0056b3; margin-top: 0; font-family: 'SimHei', Arial, sans-serif;">🔍 Step {step_num}</h3>
#             {f'<div style="margin: 10px 0;"><strong style="font-family: \'SimHei\', Arial, sans-serif;">Analysis:</strong><br><p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; font-family: \'Times New Roman\', serif;">{analysis}</p></div>' if analysis else ''}
#             {f'<div style="margin: 10px 0;"><strong style="font-family: \'SimHei\', Arial, sans-serif;">Retrieval Query:</strong><br><p style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0; font-style: italic; font-family: \'Times New Roman\', serif;">{query}</p></div>' if query else ''}
#             {f'<div style="margin: 10px 0;"><strong style="font-family: \'SimHei\', Arial, sans-serif;">Retrieved Documents:</strong><br><div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; max-height: 200px; overflow-y: auto; border-left: 4px solid #6c757d;"><pre style="white-space: pre-wrap; margin: 0; font-family: \'Times New Roman\', serif;">{docs}</pre></div></div>' if docs else ''}
#         </div>
#         """
    
#     return html

def create_step_html(step_data):
    """创建单个步骤的HTML显示"""
    step_num = step_data["step"]
    analysis = step_data["analysis"]
    query = step_data["query"]
    docs = step_data["retrieved_docs"]
    answer = step_data["answer"]
    is_final = step_data["is_final"]
    
    # 字体样式常量
    font_heading = "font-family: 'SimHei', Arial, sans-serif;"
    font_content = "font-family: 'Times New Roman', serif;"
    font_combined = "font-family: 'SimHei', 'Times New Roman', serif;"
    
    if is_final:
        # 构建analysis部分
        analysis_html = ""
        if analysis:
            analysis_html = f"""
            <div style="margin: 10px 0;">
                <strong style="{font_heading}">Analysis:</strong><br>
                <p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; {font_content}">{analysis}</p>
            </div>
            """
        
        html = f"""
        <div style="border: 2px solid #28a745; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #d4edda; {font_combined}">
            <h3 style="color: #155724; margin-top: 0; {font_heading}">🎯 Step {step_num} - Final Answer</h3>
            {analysis_html}
            <div style="margin: 10px 0;">
                <strong style="{font_heading}">Final Answer:</strong>
                <p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; font-weight: bold; {font_content}">{answer}</p>
            </div>
        </div>
        """
    else:
        # 构建各个部分
        analysis_html = ""
        if analysis:
            analysis_html = f"""
            <div style="margin: 10px 0;">
                <strong style="{font_heading}">Analysis:</strong><br>
                <p style="background-color: #fff; padding: 10px; border-radius: 5px; margin: 5px 0; {font_content}">{analysis}</p>
            </div>
            """
        
        query_html = ""
        if query:
            query_html = f"""
            <div style="margin: 10px 0;">
                <strong style="{font_heading}">Retrieval Query:</strong><br>
                <p style="background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 5px 0; font-style: italic; {font_content}">{query}</p>
            </div>
            """
        
        docs_html = ""
        if docs:
            docs_html = f"""
            <div style="margin: 10px 0;">
                <strong style="{font_heading}">Retrieved Documents:</strong><br>
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin: 5px 0; max-height: 200px; overflow-y: auto; border-left: 4px solid #6c757d;">
                    <pre style="white-space: pre-wrap; margin: 0; {font_content}">{docs}</pre>
                </div>
            </div>
            """
        
        html = f"""
        <div style="border: 2px solid #007bff; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: #e3f2fd; {font_combined}">
            <h3 style="color: #0056b3; margin-top: 0; {font_heading}">🔍 Step {step_num}</h3>
            {analysis_html}
            {query_html}
            {docs_html}
        </div>
        """
    
    return html
# 使用类来管理全局状态，避免全局变量问题
class ModelManager:
    def __init__(self):
        self.session = None
        self.is_initialized = False
    
    def initialize_model(self, model_path, retriver_url, api_url, api_key, tool_model_name, stop_token):
        """初始化模型"""
        try:
            # 初始化OpenAI客户端
            client = OpenAI(
                api_key=api_key if api_key else "EMPTY",
                base_url=api_url + "/v1",
            )
            
            # 配置字典
            config = {
                'num_passages_one_retrieval': 3,
                'num_passages_one_split_retrieval': 5,
                'max_num_passages': 80,
                'num_search_one_attempt': 10,
                'api_try_counter': 3,
                'retriver_url': retriver_url,
                'model_name': tool_model_name,
                'api_url': api_url,
                'openai_suffix': "/v1",
                'client': client,
            }
            
            # 初始化LLM
            llm = LLM(model=model_path, trust_remote_code=True)
            
            # 解析stop_token
            stop_token_ids = []
            if stop_token.strip():
                try:
                    stop_token_ids = [int(stop_token.strip())]
                except ValueError:
                    return "❌ Error: Stop token must be a valid integer"
            
            sampling_params = SamplingParams(
                temperature=0.3,
                max_tokens=512,
                stop_token_ids=stop_token_ids if stop_token_ids else [128009]
            )
            
            self.session = VisualizationChatSession(llm, sampling_params, config)
            self.is_initialized = True
            
            return "✅ Model initialized successfully!"
        except Exception as e:
            self.is_initialized = False
            return f"❌ Error initializing model: {str(e)}"
    
    def process_question(self, question, max_steps):
        """处理问题并返回可视化结果"""
        if not self.is_initialized or self.session is None:
            return "❌ Please initialize the model first!", ""
        
        if not question.strip():
            return "❌ Please enter a question!", ""
        
        try:
            visualization_steps, full_output = self.session.get_answer_with_visualization(
                question, search_chain_length=max_steps
            )
            
            # 创建完整的可视化HTML
            html_content = f"""
            <div style="font-family: 'SimHei', 'Times New Roman', serif;">
                <h2 style="color: #333; border-bottom: 2px solid #007bff; padding-bottom: 10px; font-family: 'SimHei', Arial, sans-serif;">
                    🤖 R3-RAG Step-by-Step Reasoning Process
                </h2>
                <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin: 10px 0;">
                    <strong style="font-family: 'SimHei', Arial, sans-serif;">Question:</strong> 
                    <span style="font-family: 'Times New Roman', serif;">{question}</span>
                </div>
            """
            
            for step_data in visualization_steps:
                html_content += create_step_html(step_data)
            
            html_content += "</div>"
            
            return html_content, full_output
            
        except Exception as e:
            return f"❌ Error processing question: {str(e)}", ""

def create_visualization_interface():
    """创建Gradio界面"""
    
    # 创建模型管理器实例
    model_manager = ModelManager()
    
    # 获取本机IP
    local_ip = get_local_ip()
    
    # 自定义CSS样式
    custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Times+New+Roman&display=swap');
    
    * {
        font-family: 'SimHei', 'Times New Roman', serif !important;
    }
    
    .gradio-container {
        font-family: 'SimHei', 'Times New Roman', serif !important;
    }
    
    .gr-textbox input, .gr-textbox textarea {
        font-family: 'SimHei', 'Times New Roman', serif !important;
        font-size: 14px !important;
    }
    
    .gr-button {
        font-family: 'SimHei', Arial, sans-serif !important;
        font-weight: bold !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'SimHei', Arial, sans-serif !important;
    }
    
    .markdown {
        font-family: 'SimHei', 'Times New Roman', serif !important;
    }
    """
    
    # 创建Gradio界面
    with gr.Blocks(title="R3-RAG Visualization", theme=gr.themes.Soft(), css=custom_css) as demo:
        # 论文标题和介绍
        paper_title = "R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning"
        authors = "Yuan Li, Qi Luo, Xiaonan Li, Bufan Li, Qinyuan Cheng, Bo Wang, Yining Zheng, Yuxin Wang, Zhangyue Yin, Xipeng Qiu"
        affiliation = "School of Computer Science, Fudan University"
        github_url = "https://github.com/Yuan-Li-FNLP/R3-RAG"
        
        gr.Markdown(f"""
        # {paper_title}
        
        **Authors:** {authors}  
        **Affiliation:** {affiliation}  
        **GitHub:** [{github_url}]({github_url})  
        **Server Address:** `{local_ip}:7860`
        
        ---
        
        ## 📖 About R3-RAG
        
        R3-RAG is a novel framework that uses **R**einforcement learning to teach LLMs how to **R**eason and **R**etrieve step-by-step. Unlike traditional RAG systems that rely on fixed retrieval patterns, R3-RAG learns optimal reasoning and retrieval strategies through reinforcement learning with both outcome rewards (answer correctness) and process rewards (relevance-based document verification).
        
        ### 🎯 Key Innovations:
        - **Reinforcement Learning**: Uses RL to optimize reasoning and retrieval strategies
        - **Step-by-Step Process**: Breaks down complex questions into manageable reasoning steps
        - **Dynamic Retrieval**: Retrieves relevant documents at each reasoning step based on current analysis
        - **Dual Reward System**: Combines outcome rewards (correctness) and process rewards (relevance)
        
        ### 🔧 Instructions:
        1. **Configure Model**: Set up your R3-RAG model and retrieval service
        2. **Ask Questions**: Enter questions requiring multi-step reasoning
        3. **Watch Reasoning**: Observe the step-by-step reasoning and retrieval process
        """)
        
        with gr.Tab("🔧 Model Configuration"):
            gr.Markdown("### Configure R3-RAG Model Parameters")
            
            with gr.Row():
                model_path = gr.Textbox(
                    label="Model Path",
                    value="/remote-home1/yli/Workspace/R3RAG/train/Models/SFT/llama",
                    placeholder="Path to your trained R3-RAG model",
                    info="Local path to the trained R3-RAG model"
                )
                retriver_url = gr.Textbox(
                    label="Retriever URL",
                    value="http://10.176.58.103:8001/search",
                    placeholder="URL of your retrieval service",
                    info="HTTP endpoint for document retrieval service"
                )
            
            with gr.Row():
                api_url = gr.Textbox(
                    label="API URL",
                    value="http://10.176.58.103:8002",
                    placeholder="Base URL for vLLM-compatible API",
                    info="Base URL for the vLLM server API"
                )
                api_key = gr.Textbox(
                    label="API Key",
                    value="EMPTY",
                    placeholder="API key (if required)",
                    info="Authentication key for API access"
                )
            
            with gr.Row():
                tool_model_name = gr.Textbox(
                    label="Tool Model Name",
                    value="/remote-home1/yli/Model/Generator/Qwen2.5/14B/instruct",
                    placeholder="Model name for tool operations",
                    info="Model used for query processing (driven by vLLM with OpenAI client)"
                )
                stop_token = gr.Textbox(
                    label="Stop Token ID",
                    value="128009",
                    placeholder="Stop token ID for generation",
                    info="Stop token for text generation (Qwen: 151645, Llama: 128009)"
                )
            
            init_btn = gr.Button("🚀 Initialize Model", variant="primary", size="lg")
            init_status = gr.Textbox(
                label="Initialization Status", 
                interactive=False,
                info="Status of model initialization process"
            )
            
            # 绑定模型管理器的方法
            init_btn.click(
                model_manager.initialize_model,
                inputs=[model_path, retriver_url, api_url, api_key, tool_model_name, stop_token],
                outputs=init_status
            )
        
        with gr.Tab("🤖 Question & Answer"):
            gr.Markdown("### Ask Your Question and Watch R3-RAG Reason")
            
            with gr.Row():
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Enter your question here... (e.g., complex multi-hop questions work best)",
                    lines=3,
                    scale=3,
                    info="Questions requiring multi-step reasoning work best with R3-RAG"
                )
                max_steps = gr.Slider(
                    label="Maximum Steps",
                    minimum=1,
                    maximum=20,
                    value=10,
                    step=1,
                    scale=1,
                    info="Maximum number of reasoning steps"
                )
            
            submit_btn = gr.Button("🔍 Start Reasoning Process", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 📊 Step-by-Step Reasoning Visualization")
                    visualization_output = gr.HTML()
                
                with gr.Column():
                    gr.Markdown("### 📝 Complete Model Output")
                    full_output = gr.Textbox(
                        label="Raw Model Response",
                        lines=20,
                        max_lines=30,
                        interactive=False,
                        info="Complete unprocessed output from the model"
                    )
            
            # 绑定模型管理器的方法
            submit_btn.click(
                model_manager.process_question,
                inputs=[question_input, max_steps],
                outputs=[visualization_output, full_output]
            )
            
            # 示例问题
            gr.Markdown("### 💡 Example Questions (Click to Use)")
            
            with gr.Row():
                example1 = gr.Button("📌 What's the place of birth of the former member of The Sunnyboys?", size="sm")
                example2 = gr.Button("📌 Who directed 'Danger: Diabolik' and what year was it released?", size="sm")
            
            with gr.Row():
                example3 = gr.Button("📌 What is the capital of the country where the Eiffel Tower is located?", size="sm")
                example4 = gr.Button("📌 Who won the Nobel Prize in Physics the year iPhone was first released?", size="sm")
            
            # 绑定示例问题
            example1.click(lambda: "What's the place of birth of the former member of The Sunnyboys?", outputs=question_input)
            example2.click(lambda: "Who is the director of 'Danger: Diabolik' and what is the release year of the film?", outputs=question_input)
            example3.click(lambda: "What is the capital of the country where the Eiffel Tower is located?", outputs=question_input)
            example4.click(lambda: "Who won the Nobel Prize in Physics the year iPhone was first released?", outputs=question_input)
        
        with gr.Tab("📚 Technical Details"):
            citation_text = """@article{li2024r3rag,
  title={R3-RAG: Learning Step-by-Step Reasoning and Retrieval for LLMs via Reinforcement Learning},
  author={Li, Yuan and Luo, Qi and Li, Xiaonan and Li, Bufan and Cheng, Qinyuan and Wang, Bo and Zheng, Yining and Wang, Yuxin and Yin, Zhangyue and Qiu, Xipeng},
  journal={arXiv preprint},
  year={2024}
}"""
            
            gr.Markdown(f"""
            ### 🔬 Technical Architecture
            
            **Current Server:** `{local_ip}:7860`
            
            #### R3-RAG Framework Overview:
            
            **Stage 1: Cold Start Learning**
            - Initialize the model with basic reasoning and retrieval patterns
            - Learn to interleave thinking and retrieval operations
            
            **Stage 2: Reinforcement Learning**
            - **Outcome Reward**: Measures answer correctness for the complete trajectory
            - **Process Reward**: Evaluates relevance of retrieved documents at each step
            - Uses these rewards to optimize reasoning and retrieval strategies
            
            #### Key Components:
            
            1. **Step-by-Step Reasoning**: The model analyzes problems incrementally
            2. **Dynamic Retrieval**: Retrieves documents based on current reasoning state
            3. **Reward-Guided Learning**: Uses RL to improve both reasoning quality and retrieval relevance
            4. **Transferability**: Works with different retrieval systems
            
            #### Model Configuration Details:
            
            - **Model Path**: Location of the trained R3-RAG model
            - **Retriever URL**: HTTP endpoint for document search
            - **Tool Model**: Auxiliary model for query processing
            - **Stop Tokens**: 
              - Qwen models: `151645`
              - Llama models: `128009`
            
            #### Performance Benefits:
            
            - Significantly outperforms baseline RAG methods
            - Better handling of multi-hop reasoning questions
            - Improved document relevance through process rewards
            - Transferable across different retrieval systems
            
            ---
            
            **Citation:**
            ```
            {citation_text}
            ```
            """)
    
    return demo

if __name__ == "__main__":
    # 获取本机IP并显示
    local_ip = get_local_ip()
    print(f"🌐 Server will run on: {local_ip}:7860")
    print("🚀 Starting R3-RAG Visualization Interface...")
    print("📝 Make sure your model and retrieval services are running!")
    
    # 创建并启动界面
    demo = create_visualization_interface()
    demo.launch(
        server_name=local_ip,   # 允许外部访问
        server_port=7860,       # 端口号
        share=False,            # 如果想要公共链接，设置为True
        debug=True              # 调试模式
    )