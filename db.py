import singlestoredb as s2

#connection instructions
conn_link = 'https://admin:Calhacks123@host:port/database?local_infile=True'
host = 'svc-237c8144-15b8-45fa-b7f9-0c2f105c6881-dml.aws-virginia-6.svc.singlestore.com'
port = '3306'
user = 'admin'
password = 'Calhacks123'

conn = s2.connect(host=host, port=port, user=user, password=password, database='openaidb', results_type='tuple')

#add a vector embedding to our database
#add a speech/transcript keypoint to our database--this is called per keypoint per speech per speaker
def add_vector(speaker_id, transcript_id, keypoint, emotion, vector):
    table = 'speakervectors'
    ADD_VECTOR_SQL = f'''INSERT INTO speechvectorspace (speaker_id, transcript_id, keypoint, emotion, vector) VALUES ("{speaker_id}", "{transcript_id}", "{keypoint}", '{"emotion": ${emotion}}', JSON_ARRAY_PACK("{vector}"));'''
    with conn:
        with conn.cursor() as cur:
            cur.execute(ADD_VECTOR_SQL)

#given a speaker id and a model embedding, we'd like to return the emotional data
#correlating with the model speaker's embedding
def get(speaker_id, embedding):
    GET_SQL = f''
    with conn:
        with conn.cursor() as cur:
            cur.execute(GET_SQL)

#get the similar embeddings to the given vector embedding
#we will be matching a given new set of keypoints with the existing keypoints
#After finding the k most smilar keypoints in the model speaker's speech, we'll
#return those similar keypoints. We simply are matching the linguistic scenarios
#here so we can pass those into hume later for the emotional cues and expressions that we expect
def search(embedding):
    SEARCH_SQL = f'select text, dot_product(vector, JSON_ARRAY_PACK("${embedding}")) as score from my speechvectorspace order by score desc limit 1;'
    with conn:
        with conn.cursor() as cur:
            cur.execute(SEARCH_SQL)
            row = cur.fetchone()
            print(row)
            return
