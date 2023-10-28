from hume import HumeBatchClient
from hume.models.config import FaceConfig
from hume.models.config import ProsodyConfig
from hume.models.config import BurstConfig

def get_emotion_data(video_path):
    client = HumeBatchClient("")
    urls = []
    files = [video_path]
    configs = [FaceConfig(fps_pred=60), ProsodyConfig(granularity="conversational_turn")]
    job = client.submit_job(urls, configs, files=files)
    return job
    
    # print(job)
    # print("Running...")

    # job.await_complete()
    # return job.get_predictions()