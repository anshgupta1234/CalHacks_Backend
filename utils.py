import subprocess
import json

def get_clip(video_path, time_info):
    start_time, end_time = time_info
    output_path = f"clip_{start_time}_{end_time}.mp4"  # Specify the output file name

    # Run FFmpeg to extract the clip
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-ss", str(start_time),
        "-to", str(end_time),
        "-c:v", "libx264",  # Specify the video codec (libx264)
        "-c:a", "aac",      # Specify the audio codec (AAC)
        "-strict", "experimental",  # Use experimental AAC encoding
        output_path
    ]

    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError as e:
        print("Error:", e)
        return None

def post_process(job_data):

    processed_data = {
        "face_data": {},
        "voice_data": {}
    }

    predictions = job_data[0]["results"]["predictions"][0]

    if (len(predictions["models"]["face"]["grouped_predictions"]) > 0):

        raw_face_data = predictions["models"]["face"]["grouped_predictions"][0]["predictions"]

        num_face_preds = len(raw_face_data)
        total_face_data = {}
        for prediction in raw_face_data:
            for emotion in prediction["emotions"]:
                try:
                    total_face_data[emotion["name"]] += emotion["score"]
                except:
                    total_face_data[emotion["name"]] = emotion["score"]

        processed_data["face_data"] = { k: v / num_face_preds for k, v in total_face_data.items() }

        voice_emotions = predictions["models"]["prosody"]["grouped_predictions"][0]["predictions"][0]["emotions"]
        for emotion in voice_emotions:
            processed_data["voice_data"][emotion["name"]] = emotion["score"]

        return processed_data

    else:

        return None

def compare(keypoint_data, keypoints_list, threshold):

    face_data_comparisons = []
    voice_data_comparisons = []
    summary = {}

    for keypoint in keypoints_list:
        expected_face = keypoint_data[keypoint]["expected"]["face_data"]
        expected_voice = keypoint_data[keypoint]["expected"]["voice_data"]

        actual_face = keypoint_data[keypoint]["actual"]["face_data"]
        actual_voice = keypoint_data[keypoint]["actual"]["voice_data"]

        summary[keypoint] = []

        for emotion in expected_face.keys():
            diff = expected_face[emotion] - actual_face[emotion]
            if diff > threshold:
                face_data_comparisons.append({ 
                    "keypoint": keypoint,
                    "emotion": emotion,
                    "difference": diff 
                })

                summary[keypoint].append({ 
                    "emotion": emotion,
                    "difference": diff,
                    "type": "face"
                 })

        for emotion in expected_voice.keys():
            diff = expected_voice[emotion] - actual_voice[emotion]
            if abs(diff) > threshold:
                voice_data_comparisons.append({ 
                    "keypoint": keypoint,
                    "emotion": emotion,
                    "difference": diff 
                })

                summary[keypoint].append({ 
                    "emotion": emotion,
                    "difference": diff,
                    "type": "voice"
                 })

    return face_data_comparisons, voice_data_comparisons, summary