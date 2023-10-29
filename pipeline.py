from extract import *
from speech_utils import *
from utils import *
from hume_calls import *

#generate text transcripts for all uploaded videos
def read_transcript(path):
    with open(path) as f:
        lines = f.readlines()
    return

def process(fileselector, speaker_id):
    #process all the given files in a directory
    model_keypoints = []
    for fpath in glob.glob(fileselector):
        kp_associated_data = {
            'speaker_id': speaker_id,
            'transcript_id': fpath[0:fpath.index('.mov')] + '.txt',
            'video_id': fpath,
            'speech_info': get_speech_info(fpath),
            'keypoints': keypoints(fpath)
        }
        model_keypoints.append(kp_associated_data)

    #for each keypoint
    #generate embeddings and emotion data to store in Singlestore
    for keypoint in model_keypoints:
        #find where the keypoint lingual statement maps to audio with Whisper
        time_info = get_sentence_time_segment(keypoint, keypoint['speech_info'])
        clip_path = get_clip(keypoint['video_id'], time_info)
        emotion = get_emotion_data(clip_path)
        embd = get_embedding(keypoint)
        keypoint['model_emotion'] = emotion
        keypoint['vector_embedding'] = embd
    
    return model_keypoints

def update_db(data):
    #at this point, we've populated our model_keypoints with all the necessary data
    #Let's store this somewhere for use -- namely Singlestore
    for keypoint in data:
        add_vector(speaker_id=keypoint['speaker_id'], transcript_id=keypoint['transcript_id'], video_id=keypoint['video_id'], emotion=keypoint['emotion'], vector=keypoint['vector_embedding'])

    #end of populating the data for one speaker given a set of their videos to analyze

#given a new recording, predict the model speaker's intonations and emotions that they'd use in this scenario
def predict(user_recording_filepath, transcript_filepath, model_speaker_id):
    #find new keypoints in this transcript
    #essentially the same steps as those for the speaker so
    user_transcript = read_transcript(transcript_filepath)
    user_keypoints = aggregate_keypoints("./transcript.txt", model_speaker_id)
    print(user_keypoints)
    speech_info = get_speech_info(user_recording_filepath)
    print(speech_info)
    #get emotional and speaking cues + the vector embeddings for search purposes
    for speech in user_keypoints:
        for keypoint in speech["keypoints"]:
        #find where the keypoint lingual statement maps to audio with Whisper
            time_info = get_sentence_time_segment(keypoint, speech_info)
            if sum(time_info) > 0:
                print(time_info)
                clip_path = get_clip(user_recording_filepath, time_info)
                job = get_emotion_data(clip_path)
                job.await_complete()
                user_emotion = job.get_predictions()
                processed_emotion_info = post_process(user_emotion)
                user_embd = get_embedding(keypoint)
                add_vector(model_speaker_id, transcript_filepath, keypoint, processed_emotion_info, user_embd)

        #let's now do similarity search for similar keypoints in our database of the model speakers kps
        model_closest_embedding = search(user_embd)
        
print(predict("test_video.mp4", "transcript.txt", "Joebama"))

