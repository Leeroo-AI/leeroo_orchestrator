import os
import openai
from dotenv import load_dotenv
load_dotenv("app/.env")
from app.abstract_classes import BaseServerObject


class OpenaiGptExpert(BaseServerObject):
    """
        Uses openai framework to interact with underlying gpt-* models.
        for reference visit : https://platform.openai.com/docs/models 
    """
    def __init__(self,**kwargs):
        """
        """
        self.config = kwargs
        openai.organization = os.getenv("OPENAI_ORGANIZATION")
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        self.openai_base = self.config.get('base_url')
        self.model_id = self.config.get('model_id')
        self.instance_name = self.model_id
        self.config.update(dict(logs=[]))
        
    def start_server(self):
        instance_state = dict(
            ip_address="ip_address",
            instance_name="openaigpt"
        )
        self.config.update(instance_state)
        self.config['logs'].append(f'Openai gpt server {self.model_id}')
    
    def start_inference_endpoint(self, max_wait_time=120):
        self.config['logs'].append(f'Openai gpt server inference endpoint {self.model_id}')
        
    def stop_server(self):
        pass
    
    def check_servers_state(self):
        return (True, 'running')

    def get_response(self, message, stream=False):
        messages = [{"role": "user", "content": message}]
        if stream:
            return self._generate_stream(messages)
        else:
            return self._generate(messages)

    def _generate(self, messages):
        try:
            openai.api_base = self.openai_base
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=messages
            )
            answer = response['choices'][0]['message']['content']
            return answer
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I couldn't process your request. Too many requests for me to handle!"
    
    def _generate_stream(self, messages):
        try:
            openai.api_base = self.openai_base
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=messages,
                request_timeout=300,
                stream=True
            )
            for chunk in response:
                content = chunk['choices'][0]['delta'].get("content", "")
                yield content
        except Exception as e:
            print(f"An error occurred: {e}")
            yield "Sorry, I couldn't process your request. Too many requests for me to handle!"
            
            
            
if __name__ == "__main__":
    config = {
            "model_id": "gpt-4-1106-preview",
            "backend": "openai_backend",
            "base_url": 'https://api.openai.com/v1'
        }
    
    expert = OpenaiGptExpert(**config)
    response = expert.get_response("hello! ")
    print(response)
    