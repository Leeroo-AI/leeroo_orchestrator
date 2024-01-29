import os
import json
from dotenv import load_dotenv
load_dotenv("app/.env")
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.utils import name_from_base

from app.abstract_classes import BaseServerObject
from app.utils.aws_sagemaker_utils import (
    AwsSagemakerEngine,
    get_realtime_response_stream,
    parse_response_stream,
    get_hf_image
)

class SagemakerServer(BaseServerObject):
    """
        Serve an Expert Model using AWS Sagemaker Inference.
        Expert models operate independently and can be hosted on different or common machines. 
        They may include closed-source models like GPT-4. Refer to (app/configs/demo_orch_ec2_mix.json) for configuration details.
    """
    def __init__(self,**kwargs):
        """
        """
        self.config = kwargs

        os.environ['AWS_DEFAULT_REGION'] = self.config.get("region", "us-east-1")
        
        self.engine = AwsSagemakerEngine(self.config)
        self.model_id = self.config['model_id']
        self.instance_name = self.model_id
        self._set_inferece_params()
        self.config.update(dict(logs=[]))

    def _set_inferece_params(self):
        self.inference_params = {
            "top_p": self.config.get("top_p", 0.6),
            "top_k": self.config.get("top_k", 50),
            "stop": self.config.get("stop", ["</s>"]),
            "do_sample": self.config.get("do_sample", True),
            "temperature": self.config.get("temperature", 0.9),
            "max_new_tokens": self.config.get("max_new_tokens", 512),
            "return_full_text": self.config.get("return_full_text", False),
            "repetition_penalty": self.config.get("repetition_penalty", 1.03)
        }

    def get_endpoint_name(self):
        """HACK : endpoint names with more than 2 '-' are not supported 28jan2024.
        We use model_id for creating endpoint name and hence trim the later"""
        endpoint_name = "-".join(self.config['model_id'].split("-")[0:2])
        endpoint_name = f"{endpoint_name.split('/')[1]}-tgi-streaming"
        return endpoint_name
        
    def start_server(self):
        """Since aws sagemaker deployment dosent require this step, 'start_inference_endpoint' covers everything.
        """
        self.config['logs'].append(f'Sagemaker server {self.model_id}')
        
    def start_inference_endpoint(self, max_wait_time=600):
        """
        Starts an AWS SageMaker Endpoint using the 'instance_type' from the configuration.
        The 'get_endpoint_name' method returns a unique identifier for the expert. If an inference endpoint with the same name is already present in the provided region, the operation is aborted.
        NOTE: All models available on Hugging Face can be served using AWS SageMaker.

        Args:
            max_wait_time (int, optional): Defaults to 600.
        """
        
        """Check if inference endpoint is already present"""
        is_present = False
        instance_meta = {}
        endpoints = self.engine.sagemaker_client.list_endpoints()
        model_id_trimmed =  self.get_endpoint_name()
        for endpoint in endpoints['Endpoints']:
            if model_id_trimmed in endpoint['EndpointName'] and \
                endpoint['EndpointStatus'] in ['Creating', 'InService']:
                """TODO the inference endpoint matching strategy is using the model_id
                In case of multiple experts with same model_id assign tags during endpoint creation
                and use that as a filter here"""
                is_present = True
                self.endpoint_name = endpoint['EndpointName']
                instance_meta = dict(
                    is_present=is_present,
                    ip_address="sagemaker-endpoint",
                    instance_name=self.endpoint_name,
                    endpoint_name=self.endpoint_name
                )
        self.config.update(instance_meta)
        if is_present:
            return
        
        """Create a new inference endpoint"""
        llm_image = get_hf_image(self.config.get("region", "us-east-1"))
        self.llm_image = llm_image.__str__()
        
        # sagemaker config
        instance_type = self.config['instance_type']
        number_of_gpu = self.config.get("number_of_gpu", 1)

        # Define Model and Endpoint configuration parameters
        config = {
            'HF_MODEL_ID': self.model_id, # model_id from hf.co/models
            'SM_NUM_GPUS': json.dumps(number_of_gpu), # Number of GPU used per replica
            'MAX_INPUT_LENGTH': json.dumps(self.config.get("max_input_length", 2048)),  # Max length of input text
            'MAX_TOTAL_TOKENS': json.dumps(self.config.get("max_total_length",4096)),  # Max length of the generation (including input text)
            'MAX_BATCH_TOTAL_TOKENS': json.dumps(self.config.get("max_batch_total_tokens",8192)),  # Limits the number of tokens that can be processed in parallel during the generation
            'HUGGING_FACE_HUB_TOKEN': os.getenv("HUGGING_FACE_HUB_TOKEN") # Read Access token of your HuggingFace profile https://huggingface.co/settings/tokens
        }

        # create HuggingFaceModel with the image uri
        llm_model = HuggingFaceModel(
            role=self.engine.role,
            image_uri=self.llm_image,
            env=config
        )
        
        endpoint_name = self.get_endpoint_name()
        self.endpoint_name = name_from_base(endpoint_name)
        print(self.endpoint_name)
        
        llm = llm_model.deploy(
            endpoint_name=self.endpoint_name,
            initial_instance_count=1,
            instance_type=instance_type,
            wait=False, # Whether the call should wait until the deployment of this model completes
            container_startup_health_check_timeout=max_wait_time,
        )
        
        instance_meta = dict(
            ip_address="sagemaker-endpoint",
            instance_name=self.endpoint_name,
            endpoint_name=self.endpoint_name
        )
        self.config.update(instance_meta)
        self.config['logs'].append(f'Sagemaker inference endpoint {self.model_id}')
        
    def stop_server(self):
        """Terminate the Sagemaker Endpoint
        """
        endpoint = self.engine.sagemaker_client.describe_endpoint(
            EndpointName=self.endpoint_name)
        endpoint_config_name = endpoint['EndpointConfigName']
        endpoint_config = self.engine.sagemaker_client.describe_endpoint_config(
            EndpointConfigName=endpoint_config_name)
        model_name = endpoint_config['ProductionVariants'][0]['ModelName']

        print(f"""
            About to delete the following sagemaker resources:
            Endpoint: {self.endpoint_name}
            Endpoint Config: {endpoint_config_name}
            Model: {model_name}
            """)    
        # delete endpoint
        self.engine.sagemaker_client.delete_endpoint(EndpointName=self.endpoint_name)
        # delete endpoint config
        self.engine.sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        # delete model
        self.engine.sagemaker_client.delete_model(ModelName=model_name)
    
    def _get_inference_payload(self, message, stream=True):
        payload = {
            "inputs":  message,
            "parameters": self.inference_params,
            "stream": stream
        }
        return payload
    
    def check_servers_state(self):
        payload = self._get_inference_payload("hello!", stream=True)
        try:
            resp = get_realtime_response_stream(
                self.engine.sagemaker_runtime, 
                self.endpoint_name, 
                payload
            )
        except:
            return (False, '')
        # print_response_stream(resp)
        return (True, 'running')

    def get_response(self, message, stream=True, verbose=True):
        payload = self._get_inference_payload(message, stream)
        resp = get_realtime_response_stream(
            self.engine.sagemaker_runtime, 
            self.endpoint_name, 
            payload
        )
        if stream:
            text = parse_response_stream(resp, verbose)
        else: # TODO parse the response generated when streaming is false
            raise NotImplementedError
        return text




if __name__ == "__main__":
    """
    Useful links for sagemaker deployment
    - https://sagemaker.readthedocs.io/en/stable/api/inference/model.html
    - https://aws.amazon.com/blogs/machine-learning/announcing-the-launch-of-new-hugging-face-llm-inference-containers-on-amazon-sagemaker/
    - https://aws.amazon.com/blogs/machine-learning/inference-llama-2-models-with-real-time-response-streaming-using-amazon-sagemaker/
    - https://www.philschmid.de/sagemaker-deploy-mixtral
    - https://github.com/aws-samples/amazon-sagemaker-llama2-response-streaming-recipes/tree/main
    """
    import time
    config = {
        "expert_id": 0,
        "model_id": "mistralai/Mistral-7B-v0.1",
        "instance_type": "ml.g5.xlarge"
    }
    
    expert = SagemakerServer(**config)
    
    expert.start_server()
    expert.start_inference_endpoint()
    started = False
    
    while True:
        state = expert.check_servers_state()
        print(state)
        started = state[0]
        if started: break
        time.sleep(30)
    
    
    print()