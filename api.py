from flask import Flask, request, jsonify
from speech_utils import get_speech_info, get_sentence_time_segment
from utils import get_clip, post_process, compare
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
    keypoint_data = {}
    learner_keypoints = get_keypoints(transcript)
    for keypoint in learner_keypoints:
        keypoint_data[keypoint] = {}
        expected_speaker_emotions = expected_emotions_at_keypoint(keypoint, speaker)
        keypoint_data[keypoint]["expected"] = expected_speaker_emotions
        time_info = get_sentence_time_segment(keypoint, speech_info)
        clip_path = get_clip("yo.mov", time_info)
        job = get_emotion_data(clip_path)
        keypoint_data[keypoint]["job"] = job

    for keypoint in learner_keypoints:
        keypoint_data[keypoint]["job"].await_response()
        raw_job_data = keypoint_data[keypoint]["job"].get_predictions()
        avg_data = post_process(raw_job_data)
        keypoint_data[keypoint]["actual"] = avg_data

    suggestions = []
    threshold = 0.3
    face_data_comparisons, voice_data_comparisons = compare(keypoint_data, keypoints, threshold)
    
    # Process the video file and speaker string (you can add your custom logic here)
    # For demonstration purposes, we'll just return the received data
    return jsonify({
        'speech_info': speech_info,
        'filename': video_file.filename
    })

if __name__ == '__main__':
    app.run(debug=True, port=5003)