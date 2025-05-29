from huggingface_hub import create_repo
from huggingface_hub import HfApi
api = HfApi(token="your_token")

name = "your_name"
repo_name = "your_repo_name"
create_repo("name/repo_name")

api.upload_folder(
    folder_path="your_model_path",
    repo_id="name/repo_name",    
    repo_type="model"
)