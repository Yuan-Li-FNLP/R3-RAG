
## 安装
```bash
conda create -n r3rag_eval python=3.11
conda activate r3rag

pip install -r requirements.txt
```

## 评测
- 确保当前目录是该文件夹
- 在scripts.sh中填写必要参数
- 在**不同的**三个节点上分别运行retriever model、split query model和inference model
	```bash
	bash scripts.sh retriver

	bash scripts.sh split

	bash scripts.sh eval [your_model_path]
	```

## 参数解释
- retriever_path : 检索器模型路径
- split_paths : 改写查询模型路径，文中使用Qwen-2.5-72B-Instruct
- index_path : 处理好的index路径
- corpus_path : 处理好的corpus路径
- retrieve_host : 运行retriever mode的节点的地址
- retrieve_port : 运行retriever mode的节点的任意可用的端口
- split_host : 运行split query mode的节点的地址
- split_port : 运行split query mode的节点的任意可用的端口
