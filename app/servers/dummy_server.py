import random
from dotenv import load_dotenv
load_dotenv("app/.env")
from app.abstract_classes import BaseServerObject


class DummyLocalOrchestrator(BaseServerObject):
    """
        This is a Test Server used for Debugging Orchestrator flow.
    """
    def __init__(self,**kwargs):
        """
        """
        self.config = kwargs
        self.model_id = self.config.get("model_id")
        self.instance_name = self.config.get("model_id")
        self.config.update(dict(logs=[]))
        
    def start_server(self):
        self.config['logs'].append(f'Dummy Orchestrator server {self.model_id}')
    
    def start_inference_endpoint(self, max_wait_time=120):
        self.config['logs'].append(f'Dummy Orchestrator inference endpoint {self.model_id}')
        
    def stop_server(self):
        return "server stopped"
    
    def check_servers_state(self):
        return (True, 'running')

    def get_response(self, message, stream=False):
        """Since this is a dummy orchestrator for testing, 
        it would return random expert id for each of the input query"""
        return random.randint(0,2)         


class DummyLocalExpert(DummyLocalOrchestrator):
    def start_server(self):
        self.config['logs'].append(f'Dummy Expert server {self.model_id}')
    
    def start_inference_endpoint(self, max_wait_time=120):
        self.config['logs'].append(f'Dummy Expert inference endpoint {self.model_id}')
    
    def get_response(self, message, stream=False):
        """Since this is a dummy Expert for testing, 
        it would return the same respose for every query"""
        return "Hello! This is sample response!"