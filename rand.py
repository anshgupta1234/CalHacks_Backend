def get_emotion_for_keypoint(keypoint):
    keypoint_data[keypoint]["expected"] = expected_speaker_emotions
    time_info = get_sentence_time_segment(keypoint, speech_info)
    clip_path = get_clip("yo.mov", time_info)
    job = get_emotion_data(clip_path)