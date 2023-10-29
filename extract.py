from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import openai
import os
import singlestoredb as s2
import glob
from db import add_vector, search

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = "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"

#identify the key points in a transcript
#We describe a key point as a major speech point that the speaker makes in an effort
#to emphasize a certain concept or emotion. We use ChatGPT which does a good job of
#pattern matching and classification to essentially to find keypoints in a speech. Given
#the large amount of media, it's processed--it should be adept with speeches.
def identify_keypoints(filepath):
    loader = TextLoader(filepath)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_documents(texts, embeddings, persist_directory="/tmp/chroma/")

    prompt_template = """Use the following pieces of context to answer the question at the end. 
    {context}

    Question: {question}
    """
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs)
    #Add a confidence level field alongside each field you extract representing how confident you are in your result.
    query = '''
    You are given a speech's transcript here. Your goal here is to identify the key points in this speech. For the purposes of this task,
    you can define a key point as something that makes a new significant argument in the speech.

    Follow these guidelines for a keypoint.
    1. It is a direct quote from the speech word for word.
    2. The maximum size of the keypoint you can choose is one sentence. Do not choose a keypoint consisting of multiple sentences.
    3. All keypoints are meant to be unique and shouldn't overlap with other keypoints.
    4. It should be relevant and hold high significance to the argument.

    Finally, represent your answers as a list of keypoints where the elements of the list are the exact pices of speech that you have identified as
    a keypoint. Use the formatting: ["keypoint1", "keypoint2"].
    Choose a maximum of 5 keypoints.
    '''
    
    result = qa.run(query)
    return result

def keypoints(filepath):
    result = identify_keypoints(filepath)
    #convert the text output into a list
    begin = result.index("[")
    end = result.index("]")
    list_string = result[begin + 1 : end]
    list = list_string.split(",")
    return list

#now that we have our keypoints, we address the entire labeling problem--primary system
#Let's run this method over all the media that we're given to develop a holistic view
#aggregating the necessary keypoints for us to run similarity search and determine labels
#returns a list of dictionary elements that contain an id with a list of keypoints for the file of filename of id
def aggregate_keypoints(fileselector):
    ungrouped_kps = []
    for fpath in glob.glob(fileselector):
        kp_associated_data = {
            'transcript_id': fpath,
            'keypoints': keypoints(fpath)
        }
        ungrouped_kps.append(kp_associated_data)
    for ungrouped_kp in ungrouped_kps:
        print(ungrouped_kp['keypoints'])
        print('\n\n...\n\n')

    return ungrouped_kps

#speakers = aggregate_keypoints('./transcripts/*.txt')

def get_embedding(text):
    return openai.Embedding.create(input = [text], model="text-embedding-ada-002")['data'][0]['embedding']

def group_keypoints(spkr_id, trscpt_id, speakers):
    for speaker in speakers:
        for kp in speaker['keypoints']:
            embd = get_embedding(kp)
            add_vector(spkr_id, trscpt_id, kp, embd)

#Difference function to evaluate the differences in speech and emotional cues of
#two specific keypoint analyses from Hume.AIg 
def difference(model_speech_emos, human_speech_emos):
    return

def score(transcript, speaker_id):
    #identif
    #return json() 
    #keypoints from your transcript
    #{kp:[
    #  your_keypoints,
    #  kp1: {
    #     expected_emotion_data
    #  }
    # 
    # }
    speakers = aggregate_keypoints('./data/model/*.txt')
    #update the database by extracting embedding for each keypoint for speech for each speaker
    group_keypoints(speaker_id, transcript + uuid(), speakers)
    #access hume functions
    customer_kps = aggregate_keypoints('./data/customer/*.txt')
    differences = [{}]
    for customer_kp in customer_kps:
        for customer_keypoint in customer_kp['keypoints']:
            customer_embedding = get_embedding(customer_keypoint)
            most_similar_model_keypoint, transcript_id = search(customer_embedding)
            customer_emotion_at_keypoint = get_human_emotion(customer_keypoint)
            differences.append({
                'source_customer_keypoint': customer_keypoint,
                'nearest_model_keypoint': most_similar_model_keypoint,
                'customer_emotion_data': customer_emotion_at_keypoint,
                'most_similar_model_data': hume_emotion_from_clip(clip_path(get_timestep_for_keypoint(transcript_id, most_similar_model_keypoint))),
                'emotional_difference': difference(most_similar_model_keypoint, customer_emotion_at_keypoint)
            })
    }


print('Monkey\n')
print(get_embedding('monkey'))
'''print('\n\nParent\n\n')
print(get_embedding('parent'))
print('\n\nChildren\n\n')
print(get_embedding('children'))
print('\n\nFuture of our children is doomed.\n\n')
print(get_embedding('Future of our children is doomed.'))
print('\n\n.')
print('\n\nHow we treat our kids determines our future.\n\n')
print(get_embedding('How we treat our kids determines our future.'))
print('\n\n.')
print('\n\nWe must think as parents to best determine the future.\n\n')
print(get_embedding('We must think as parents to best determine the future.'))'''