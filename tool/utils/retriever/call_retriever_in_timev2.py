import time
import schedule
import logging
from datetime import datetime
import tqdm, requests
POST_BATCH_SIZE=1024
SSL_RETRY=3
# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scheduler.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
def GetRetrieval(query):
    payload = {
        "query": query
    }
    headers = {
        "Content-Type": "application/json"
    }
    try:
        response = requests.post("http://10.244.69.222:8001/search", json=payload, headers=headers)
        if response.status_code == 200:
            retrieval_results = response.json()
            return retrieval_results
        else:
            print(f"Error {response.status_code}: {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def task_to_execute():
    """
    在这里实现你需要定时执行的任务
    """
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"任务执行时间: {current_time}")
    
    # TODO: 在这里添加你的具体实现
    # 示例:
    logger.info("执行任务...")
    # 你的代码...
    query_list = ["What is the capital of France?"]
    query_list = query_list*1
    doc_list = []
    for query in query_list:
        doc_list.append(GetRetrieval(query)) 
    logger.info("任务完成")

def setup_schedule(interval_type, interval_value=1):
    """
    设置定时任务
    
    参数:
        interval_type: 时间间隔类型，可选值: "seconds", "minutes", "hours", "days", "weeks"
        interval_value: 时间间隔值，默认为1
    """
    if interval_type == "seconds":
        schedule.every(interval_value).seconds.do(task_to_execute)
    elif interval_type == "minutes":
        schedule.every(interval_value).minutes.do(task_to_execute)
    elif interval_type == "hours":
        schedule.every(interval_value).hours.do(task_to_execute)
    elif interval_type == "days":
        schedule.every(interval_value).days.do(task_to_execute)
    elif interval_type == "weeks":
        schedule.every(interval_value).weeks.do(task_to_execute)
    else:
        logger.error(f"不支持的时间间隔类型: {interval_type}")
        return False
    
    logger.info(f"已设置每{interval_value}{interval_type}执行一次任务")
    return True

def run_scheduler():
    """运行调度器"""
    logger.info("调度器启动...")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("调度器被手动停止")
    except Exception as e:
        logger.error(f"调度器出错: {e}")
    
    logger.info("调度器已停止")

if __name__ == "__main__":
    # 设置每小时执行一次
    # 你可以修改为: "seconds", "minutes", "hours", "days", "weeks"
    setup_schedule("seconds", 1)
    
    # 启动调度器
    run_scheduler()