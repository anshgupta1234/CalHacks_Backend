from flask import Flask, request, jsonify
from speech_utils import *
from utils import *
from extract import *
from flask_cors import CORS, cross_origin
from hume_calls import get_emotion_data

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/process_video', methods=['POST'])
@cross_origin()
def process_video():
    # Check if the request contains a file and a speaker string
    if 'video' not in request.files or 'speaker' not in request.form:
        return jsonify({'error': 'Video file and speaker string are required'}), 400

    video_file = request.files['video']
    filename = video_file.filename
    speaker = request.form['speaker']

    video_file.save(filename)

    # Check if the video file has an allowed extension (e.g., .mp4)
    allowed_extensions = ['mp4', 'mov']
    if '.' in video_file.filename and video_file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
        return jsonify({'error': 'Invalid file format. Allowed formats: .mp4, .mov'}), 400

    speech_info = get_speech_info(filename)

    transcript = speech_info["text"]
    text_file = open("./temp/transcript_api.txt", "w+")
    text_file.write(transcript)
    text_file.close()

    keypoint_data = {}
    learner_keypoints = keypoints("./temp/transcript_api.txt")
    for keypoint in learner_keypoints:
        # get model kp data
        keypoint_data[keypoint] = {}
        # get user kp data
        time_info = get_sentence_time_segment(keypoint, speech_info)
        if sum(time_info) > 0:
            clip_path = get_clip(filename, time_info)
            embd = str(get_embedding(keypoint))
            model_kp_details = search(embd)
            keypoint_data[keypoint]["expected"] = model_kp_details["emotion"]
            job = get_emotion_data(clip_path)
            keypoint_data[keypoint]["job"] = job
        else:
            keypoint_data[keypoint]["job"] = None

    for keypoint in learner_keypoints:
        if keypoint_data[keypoint]["job"] != None:
            keypoint_data[keypoint]["job"].await_complete()
            raw_job_data = keypoint_data[keypoint]["job"].get_predictions()
            avg_data = post_process(raw_job_data)
            keypoint_data[keypoint]["actual"] = avg_data

    suggestions = []
    threshold = 0.2
    face_mistakes, voice_mistakes, summary = compare(keypoint_data, learner_keypoints, threshold)
    
    # Process the video file and speaker string (you can add your custom logic here)
    # For demonstration purposes, we'll just return the received data
    return jsonify({
        'face_mistakes': face_mistakes,
        'voice_mistakes': voice_mistakes,
        'summary': summary,
        'filename': video_file.filename
    })

if __name__ == '__main__':
    app.run(debug=True, port=5005)