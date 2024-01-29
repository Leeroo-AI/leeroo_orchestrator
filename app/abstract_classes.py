from abc import ABC, abstractmethod

class BaseServerObject(ABC):
    def __init__(self, **kwargs):
        self.config = kwargs
        self.model_id = self.config.get('model_id')
        self.config.update(dict(logs=[]))

    @abstractmethod
    def start_server(self):
        pass

    @abstractmethod
    def start_inference_endpoint(self):
        pass

    @abstractmethod
    def stop_server(self):
        pass

    @abstractmethod
    def check_servers_state(self):
        pass

    @abstractmethod
    def get_response(self):
        pass
