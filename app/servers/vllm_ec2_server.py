import openai
from app.abstract_classes import BaseServerObject
from app.utils.ssh_utils import get_ssh_session
from app.utils.aws_ec2_utils import (
    AwsEngine,
    create_or_revive_expert_instance,
    check_vllm_server_running_status,
    run_vllm_server,
    stop_ec2_instance
)


class Ec2VllmExpert(BaseServerObject):
    """
        Serve an Expert Model using VLLM on AWS EC2 Machine
        Expert models operate independently and can be hosted on different or common machines. 
        They may include closed-source models like GPT-4. Refer to (app/configs/demo_orch_ec2_mix.json) for configuration details.
    """
    def __init__(self,**kwargs):
        """
        """
        self.config = kwargs
        self.aws_engine = AwsEngine(self.config)
        self.instance_name = self.config.get('instance_name')
        self.model_id = self.config.get('model_id')
        self.base_url = None
        self.config.update(dict(logs=[]))
        
    def start_server(self):
        """
        Starts a Dedicated EC2 Instance
        The 'instance_name' serves as a unique identifier. If an instance tagged with 'instance_name' is already present in a given region, the operation is aborted. The unique identifier can be edited in app/server_manager.py: get_server.
        """
        instance_state = \
            create_or_revive_expert_instance(
                self.aws_engine,
                instance_name=self.instance_name,
                model_id=self.model_id,
                expert_instance_type=self.config['instance_type'],
                KeyName=self.config['KeyName'],
                ami_id=self.config['ami_id'],
                wait_for_expert_to_start=False,
                key_path = self.config['KeyPath']
            )
        self.config.update(instance_state)
        self.config['logs'].append(f'starting expert server {self.model_id}')

    def _set_base_url(self):
        self.ip_address = self.config.get('ip_address')
        self.port = self.config.get('port', 8000)
        self.transfer_protocol =  self.config.get('transfer_protocol', 'http')
        self.base_url = \
            f"{self.transfer_protocol}://{self.ip_address}:{self.port}/v1"
        self.docs_url = \
            f"{self.transfer_protocol}://{self.ip_address}:{self.port}/docs"
    
    def start_inference_endpoint(self, max_wait_time=120):
        """Starts a new tumx session with name 'tmux_session_name' and activates the environment 'pytorch'.        
        NOTE: we provide an aws ami that has the required conda env and VLLM installed and ready to used. For changing the env refer app/utils/ssh_utils.py : start_vllm_server

        Args:
            max_wait_time (int, optional): Defaults to 120.
        """
        instance_meta = \
            run_vllm_server(
                    self.aws_engine,
                    self.instance_name,
                    self.model_id,
                    max_wait_time = max_wait_time,
                    wait_for_expert_to_start=False,
                    key_path= self.config['KeyPath']
                )
        self.config.update(instance_meta)
        self._set_base_url()
        self.config['logs'].append(f'starting expert inference endpoint {self.model_id}')
        
    def stop_server(self):
        """Stops the Ec2 server. 
        """
        response = stop_ec2_instance(
            self.aws_engine, 
            self.config['InstanceId']
        )
        return response
    
    def check_servers_state(self):
        ssh = get_ssh_session(self.config['ip_address'], key_path=self.config['KeyPath'])
        status = check_vllm_server_running_status(
                ssh, wait=False, verbose=False, docs_url=self.docs_url)
        return status

    def get_response(self, message, stream=False):
        """Generate Text using the expert LLM.

        Args:
            message (str): Input Query
            stream (bool, optional): Get a response stream. Defaults to False.

        Returns:
            response (str): Output Text Generation
        """
        messages = [{"role": "user", "content": message}]
        if stream:
            return self._generate_stream(messages)
        else:
            return self._generate(messages)
    
    def _generate(self, messages, max_new_tokens=1000):
        openai.api_base = self.base_url
        try:
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=messages,
                request_timeout=60
            )
            answer = response['choices'][0]['message']['content']
            return answer
        except Exception as e:
            print(f"An error occurred: {e}")
            return "Sorry, I couldn't process your request. Too many requests for me to handle!"
    
    def _generate_stream(self, messages, max_new_tokens=1000):
        openai.api_base = self.base_url
        try:
            response = openai.ChatCompletion.create(
                model=self.model_id,
                messages=messages,
                request_timeout=60,
                stream=True
            )
            for chunk in response:
                content = chunk['choices'][0]['delta'].get("content", "")
                yield content
        except Exception as e:
            print(f"An error occurred: {e}")
            yield "Sorry, I couldn't process your request. Too many requests for me to handle!"
