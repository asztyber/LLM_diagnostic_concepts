import faultdiagnosistoolbox as fdt
import sympy as sym
import json
import re
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

class ThreadsWithAI():


    def __init__(self, prompt, model_name="gpt-4-turbo-preview"):
        """
        Creates and configures the GPT model with an assistant for fault diagnosis.
        
        Args:
            prompt (str): The instructions for the assistant
            model_name (str): The name of the GPT model to use
            
        Returns:
            tuple: (client, assistant_id, model_name)
            
        Raises:
            ValueError: If OPENAI_API_KEY environment variable is not set
        """
        # Check if API key is set

        api_key = os.getenv('OPENAI_API_KEY')

        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
        
        # Initialize the OpenAI client with API key
        client = OpenAI(api_key=api_key)
        

        if model_name == "gpt-4o-mini-2024-07-18":
            assistant = client.beta.assistants.create(
                name="FaultDiagnosis",
                instructions=prompt,
                temperature=0,
                top_p = 0,
                #presence_penalty = 0,
                #frequency_penalty = 0,
                model=model_name,
            )

        else:
            assistant = client.beta.assistants.create(
                name="FaultDiagnosis",
                instructions=prompt,
                #temperature=0,
                model=model_name,
            )
 

        # Create the assistant with specific instructions
        
        thread = client.beta.threads.create()

        self.client = client
        self.assistantID = assistant.id
        self.model_name = model_name
        self.thread = thread




    def _extract_from_message(self,key, message):
        """
        Extracts JSON data from a message using a key.
        
        Args:
            key (str): The key to look for in the JSON data
            message (str): The message containing JSON data
            
        Returns:
            The value associated with the key in the JSON data, or error messages if not found
        """
        json_regex = re.search(r'\{[\s\S]*\}', message)

        if json_regex:
            json_data = json_regex.group(0)
            json_data = json_data.replace("'", '"')
            try:
                data = json.loads(json_data)
                return data.get(key, [])
            except json.JSONDecodeError:
                return ['Decode JSON Error']
        else:
            return ['No JSON']

    def call_GPT(self, thread, input, client, assistant_id):
        """
        Calls the GPT model with additional input information (e. g. model definition or incidence matrices).
        The GPT model is called in a thread where the messages contain the additional input information.
        The same thread is used for all calls to the GPT model within one experiment run.
        
        Args:
            input (list): List of input messages to send to the model
            client: The OpenAI client
            assistant_id: The ID of the assistant to use
            
        Returns:
            List of responses from the assistant
        """
        try:
            # Initialize the thread

            #print("########################################################### Input: " + str(input))
            # Add the additional input information to the thread
            
            client.beta.threads.messages.create(thread_id=thread.id, role="user",content=input)

            #print("Augenkrebs")
            #print("ID: ", assistant_id)
            # Create and poll the run in one step
            run = client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant_id,
                max_completion_tokens = 30000
            )


            if run.status == 'completed':
                # Get all messages from the thread
                messages = client.beta.threads.messages.list(
                    thread_id=thread.id
                    
                )

                return messages.data[0].content[0].text.value
            elif run.status == "failed":
                return [f"OpenAI Error: {run.last_error}"]
            else:
                return ["OpenAI Error: Unknown status"]
                
        except Exception as e:
            return [f"Error in GPT call: {str(e)}"]


def get_IMs_from_model(model_def):
    """
    Extracts the IMs from a given model.
    
    Args:
        model_def: The model definition to analyze
        
    Returns:
        The incidence matrix X from the model
    """
    # Create the diagnosis model
    model = fdt.DiagnosisModel(model_def)
    model.Lint()

    # Get the incidence matrix X
    model_IMs = model.X
    return model_IMs

