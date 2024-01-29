from app.servers.vllm_ec2_server import Ec2VllmExpert
from app.servers.sagemaker_server import SagemakerServer
from app.servers.gpt_server import OpenaiGptExpert
from app.servers.dummy_server import DummyLocalOrchestrator, DummyLocalExpert


server_factory = dict(
    ec2_vllm_backend= Ec2VllmExpert,
    sagemaker_backend= SagemakerServer,
    openai_backend= OpenaiGptExpert,
    dummy_orchestrator= DummyLocalOrchestrator,
    dummy_expert= DummyLocalExpert
)

def get_instance_name(model_id, server_type):
    return f"{server_type}-{model_id}"

def get_server(config, server_type='expert'):
    instance_name = get_instance_name( config['model_id'], server_type)
    config.update(dict(instance_name=instance_name))
    expert_cls = server_factory.get(config['backend'])
    return expert_cls(**config)


    
