import singlestoredb as s2

#connection instructions
conn_link = 'https://admin:Calhacks123@host:port/database?local_infile=True'
host = 'svc-237c8144-15b8-45fa-b7f9-0c2f105c6881-dml.aws-virginia-6.svc.singlestore.com'
port = '3306'
user = 'admin'
password = 'Calhacks123'


#add a vector embedding to our database
#add a speech/transcript keypoint to our database--this is called per keypoint per speech per speaker
def add_vector(speaker_id, transcript_id, keypoint, emotion, vector):
    conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='tuple')
    ADD_VECTOR_SQL = f'''
        INSERT INTO speechvectorspace (speaker_id, transcript_id, keypoint, emotion, vector)
        VALUES ("{speaker_id}", "{transcript_id}", "{keypoint}", '{emotion}', JSON_ARRAY_PACK("{vector}"));
    '''
    with conn:
        with conn.cursor() as cur:
            cur.execute(ADD_VECTOR_SQL)
            cur.close()

#given a speaker id and a model embedding, we'd like to return the emotional data
#correlating with the model speaker's embedding
def get(speaker_id, embedding):
    conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='tuple')
    GET_SQL = f''
    with conn:
        with conn.cursor() as cur:
            cur.execute(GET_SQL)
            cur.close()

#get the similar embeddings to the given vector embedding
#we will be matching a given new set of keypoints with the existing keypoints
#After finding the k most smilar keypoints in the model speaker's speech, we'll
#return those similar keypoints. We simply are matching the linguistic scenarios
#here so we can pass those into hume later for the emotional cues and expressions that we expect
def search(embedding):
    SEARCH_SQL = f'select keypoint, dot_product(vector, JSON_ARRAY_PACK("${embedding}")) as score from speechvectorspace order by score desc limit 1;'
    conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='tuple')
    model_closest_keypoint = ''
    with conn:
        with conn.cursor() as cur:
            cur.execute(SEARCH_SQL)
            row = cur.fetchone()
            model_closest_keypoint, n = (row)
            cur.close()
    print('M: ', model_closest_keypoint)
    GET_SQL = f'SELECT * from speechvectorspace WHERE keypoint = "{model_closest_keypoint}"'
    conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='dict')
    keypoint_dictionary = {}
    with conn:
        with conn.cursor() as cur:
            cur.execute(GET_SQL)
            row = cur.fetchone()
            keypoint_dictionary = row
            cur.close()
    return keypoint_dictionary