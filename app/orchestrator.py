from app.server_manager import get_server

class Orchestrator:
    """The Orchestrator functions as the central wrapper object, facilitating the seamless deployment of underlying models. Each individual model can be hosted on a dedicated server for efficient and scalable service.
    Furthermore, multiple models can be hosted on a shared server. Refer to the configurations in the `configs/demo*` directory for illustrative examples.
    """
    def __init__(self, config, verbose=True):
        """init

        Args:
            config (json): The configuration file contains all the necessary information required to initialize a server. Refer to the examples in the `app/config` directory for guidance.
            verbose (bool, optional): Defaults to True.
        """
        self.config = config
        self.verbose = verbose
        self.orchestrator = get_server(config['orchestrator'], "orchestrator")
        self.experts = {expert_config['expert_id']: get_server(expert_config) \
                        for expert_config in self.config['experts']}
        self.config['logs'] = []

    def load_orchestrator_server(self):
        """Loads the Orchestrator Server. Activates the dedicated machine using specifications from the configuration for the orchestrator server.
        """
        if self.orchestrator.config.get('s3_download_path'):
            raise NotImplementedError
        instance_state = self.orchestrator.start_server()
        self.config['logs'].append('starting orchestrator')

    def load_experts_server(self):
        """
        Load Servers for Each Expert
        Activates dedicated machines using specifications from the configuration for each of the Experts' servers.
        """
        for expert_id,expert in self.experts.items():
            instance_state = expert.start_server()
            self.config['logs'].append(f'starting expert {expert_id}')

    def start_inference_endpoints(self, max_wait_time=120):
        """Start Inference Endpoint for Orchestrator and All Experts.
        The inference endpoints for each model can be served using VLLM or AWS Sagemaker.
        Note: For closed-source endpoints like GPT-4, `start_inference_endpoint` is non-functional.
        
        Args:
            max_wait_time (int, optional): Defaults to 120.
        """
        self.orchestrator.start_inference_endpoint(max_wait_time)
        for expert in self.experts.values():
            expert.start_inference_endpoint(max_wait_time)
            
    def check_servers_state(self):
        """Perform a ping test on each server to confirm their operational status.

        Returns:
            status (bool): True/False if the the inference endpoint is running.
        """
        status = []
        state = self.orchestrator.check_servers_state()
        print(f"Model id {self.orchestrator.instance_name} : {state}")
        status.append(state[0])
        for expert in self.experts.values():
            state = expert.check_servers_state()
            print(f"Model id {expert.instance_name} : {state}")
            status.append(state[0])
        return False not in status

    def get_response(self,query):
        """Get the Response for input query.
        The input query undergoes orchestration to determine the most suitable expert model. Subsequently, the query is processed through the chosen expert model, and the resulting response is returned.   

        Args:
            query (str): Input Query.

        Returns:
            response (str): Response.
        """
        expert_id = self.get_orchestrator_response(query)
        response = self.get_expert_response(query,expert_id)
        return response
    
    def get_orchestrator_response(self,query):
        expert_id = self.orchestrator.get_response(query)
        print(f"Selecting Expert Id {expert_id} for current query")
        return expert_id
    
    def get_expert_response(self,query,expert_id):
        expert = self.experts.get(expert_id)
        response = expert.get_response(query)
        return response