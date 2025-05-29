from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = 'your_token'
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

from modelscope.hub.constants import Licenses, ModelVisibility

owner_name = 'your_name'
model_name = 'your_repo_name'
model_id = f"{owner_name}/{model_name}"


api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
    chinese_name="我的测试模型"
)

api.upload_folder(
    repo_id=f"{owner_name}/{model_name}",
    folder_path='your_model_path',
    commit_message='upload model folder to repo',
)