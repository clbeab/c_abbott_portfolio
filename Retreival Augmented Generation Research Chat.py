import os
from langchain_community.vectorstores.milvus import Milvus
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from openai import OpenAI
import gradio as gr

os.environ['NVIDIA_API_KEY'] = 'ENTER YOUR API KEY HERE'
embedder = NVIDIAEmbeddings(model="nvolveqa_40k")
vector_store = Milvus(embedding_function=embedder, connection_args={"host": "127.0.0.1", "port": "19530"})

### Create the system message given to the LLM as instruction
conversation_history = ''
sys_message = '''You are a research assistant at the University of Colorado Colorado Springs (UCCS).  
Your job is to discuss the user\'s research with them and provide guidance, feedback, and suggestions.
You will be provided with a conversation history and some exerpts from research papers which may be relevant to the conversation by the assistant.'''

def get_chat(user_input, sys_message=sys_message, vector_store=vector_store, return_buffer=True):
    global conversation_history
    query = str(user_input)

    # Appends the current user query to the chat history
    conversation_history += 'USER: ' + query + '\n'
    conversation_history = str(conversation_history)
    sys_message = str(sys_message)

    # Checks the vector database for similar entries and reutrns the top 5
    results = vector_store.similarity_search(query=query,k=5)

    # Hands the LLM the conversation history as well as the vector results for additional context during generation
    context = 'Here are some exerpts from published research papers that may be helpful in answering the user query.  These papers may have been published after the last training date, so do not reject them.\n'
    references = ''
    reference_id = 1
    short_ref = ''
    for result in results:
        short_ref += result.metadata['Title']
        if result.metadata['Title'] not in references:
            references += f'[{reference_id}] '+result.metadata['Title']+'\n'
            # references += f'[{reference_id}] '+result.metadata['Title']+' Published on '+result.metadata['Published']+' by '+result.metadata['Authors']+'"\n'
            reference_id += 1
        context += 'NEW EXERPT: Published on '+result.metadata['Published']+' by '+result.metadata['Authors']+' titled "'+result.metadata['Title']+'"\n'
        context += 'EXERPT: '+result.page_content+'\n\n'

    client = OpenAI(
    base_url = "https://integrate.api.nvidia.com/v1",
    api_key = os.environ['NVIDIA_API_KEY']
    )

    # Asks the LLM to only include the citation if the relevant information was used
    ammended_query = query + ' If information provided by the assistant was used, please cite it.  Otherwise, do not cite sources.'

    if len(conversation_history) > 4*4000:
        conversation_history = conversation_history[-4*4000:]

    completion = client.chat.completions.create(
    model="meta/llama3-70b-instruct",
    messages=[
        {'role':'system','content':sys_message},
        {'role':'assistant','content':context +'Here is a summary of the conversation so far:\n'+conversation_history},
        {"role":"user","content":ammended_query}
        ],
    temperature=0.5,
    top_p=1,
    max_tokens=1024,
    stream=True
    )

    # Get response and append to chat history
    response = ''
    conversation_history += 'AGENT: '
    for chunk in completion:
        if chunk.choices[0].delta.content is not None:
            conversation_history += chunk.choices[0].delta.content
            response += chunk.choices[0].delta.content + ' '
            yield response
    response +=('\n\n')
    response +=('References Provided to Assistant:\n')
    response += (references + '\n')
    response +=('--- Please Verify Any References Included In The Response That Were Not Provided ---'+'\n')
    yield response

# Sample intro message
initial_msg = 'Welcome to the UCCS Physics Chatbot powered by Llama 3'

# Basic GUI for demostration purposes
chatbot = gr.Chatbot(value = [[None, initial_msg]])
demo = gr.ChatInterface(get_chat, chatbot=chatbot).queue()

try:
    demo.launch(debug=True, share=True, show_api=False)
    demo.close()
except Exception as e:
    demo.close()
    print(e)
    raise e
