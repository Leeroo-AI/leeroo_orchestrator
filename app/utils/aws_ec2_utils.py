import os
import time
import boto3
import paramiko
import requests
from dotenv import load_dotenv
load_dotenv("app/.env")

from app.utils.ssh_utils import (
    get_ssh_session,
    start_vllm_server,
    is_running_vllm_server
)


class AwsEngine:
    """AwsEngine uses boto3 for ec2 connections
    """
    def __init__(self,config):
        self.ec2 = boto3.client(
            'ec2',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=config.get("region", "us-east-1")
        )
        self.ec2_resource = boto3.resource(
            'ec2',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=config.get("region", "us-east-1")
        )
        
        
def get_available_instances(aws_engine):
    """Get the details for all the available instances using aws_engine.
    
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
    """
    response = aws_engine.ec2.describe_instances()
    available_instances = []
    for instances in response['Reservations']:
        for instance in instances['Instances']:
            available_instances.append(
                    instance
                )
    return available_instances


def get_instance(aws_engine,InstanceId):
    """Get Metadata for an Instance

    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        InstanceId (str): Instance ID.

    Returns:
        response (json): Instance metadata
    """
    response = aws_engine.ec2.describe_instances(
        InstanceIds=[InstanceId],
    )
    return response


def is_ec2_instance_present(aws_engine, instance_name):
    """
    check if an ec2 instance is present
    
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        instance_name (str): The tag assigned to instance.
    Returns:
        status (dict): Instance Running Status
    """
    ec2_instances_present = get_available_instances(aws_engine)
    for instance in ec2_instances_present:
        for current_instance_name in instance.get('Tags',[]):
            if current_instance_name['Key'] == "Name":
                if current_instance_name['Value'] == instance_name:
                    if instance['State']['Name'] != 'terminated':
                        return dict(
                            is_present=True, 
                            meta=instance
                            )
    return dict( is_present=False, meta=None )


def create_ec2_instance(
        aws_engine,
        expert_instance_name,
        expert_instance_type='g5.2xlarge',
        KeyName = 'connection-key',
        ami_id = "ami-your-ec2-ami-with-vllm-installed"
    ):
    """
    Created an ec2 instance in the default region.  
    
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        expert_instance_name (str): This will be the instance name
        expert_instance_type (str, optional): Defaults to 'g5.2xlarge'.
        KeyName (str, optional): The pem file key pair to be used. Defaults to 'connection-key'.
        ami_id (str, optional): The ami should have vllm installed. 
                                Defaults to "ami-your-ec2-ami-with-vllm-installed".
                                This is all the required repo's and pakages installed.
                                conda environment to be used is pytorch.
    """
    if not ami_id:
        raise
        
    instance = aws_engine.ec2_resource.create_instances(  
        ImageId = ami_id,
        InstanceType = expert_instance_type,
        
        MinCount=1,
        MaxCount=1,
        
        KeyName=KeyName,
        
        SecurityGroupIds=[
            # This sequrity group has port 8000 open and allows ssh connection
             os.getenv('SECURITY_GROUP_ID'),
        ],
        

        TagSpecifications=[
            {   'ResourceType': 'instance',
                'Tags': [
                    {
                        'Key': 'Name',
                        'Value': expert_instance_name
                    },
                ]
            },
        ],
    )
    return instance


def create_or_revive_expert_instance(
    aws_engine,
    instance_name: str,
    model_id: str,
    expert_instance_type: str='g5.2xlarge',
    KeyName: str = 'connection-key',
    ami_id: str ="ami-your-ec2-ami-with-vllm-installed",
    tmux_session_name: str = "vllm_server",
    wait_for_expert_to_start = False,
    key_path = "app/connection-key.pem"
):
    """
    Check if ec2 instace is already present.
    If alreafy present and is inactive, turn it on
    Else create a new instance.
    Connect to the instance and start vllm server.
    
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        instance_name (str): The tag assigned to instance.
        model_id (str): _description_
        expert_instance_type (str, optional): _description_. Defaults to 'g5.2xlarge'.  
        KeyName (str, optional): _description_. Defaults to 'connection-key'.  
        ami_id (str, optional): _description_. Defaults to "ami-your-ec2-ami-with-vllm-installed".  
        tmux_session_name (str, optional): Name assigned to tmux session running inference endpoints. Defaults to "vllm_server".  
        wait_for_expert_to_start (bool): wait until the expert server is started successfully.
            It might be recommended to turn this off when threading is not used and experts are started in series.  
            This would eliminate the wait time, and hence the expert checking can be handled outside this function.  
        key_path (str) : Path to the pem file required to connect to ec2 instance.  
    """
    # expert_instance_name = modelid_to_instancename(model_id)
    
    instance_meta = is_ec2_instance_present(aws_engine, instance_name )
    if instance_meta['is_present'] and \
        instance_meta['meta']['State']['Name']=='stopped':
        print("Instance is present and is in stopped state, reviving...")
        revive_ec2_instance(aws_engine,instance_meta['meta']['InstanceId'])
        # instance = check_ec2_server_running_status(aws_engine, instance_meta['meta']['InstanceId'])
    elif instance_meta['is_present'] and \
        instance_meta['meta']['State']['Name'] == 'running':
        print("Instance is present and is in running state...")
    elif not instance_meta['is_present']:
        print("Creating new instance...")
        """Create a new ec2 instance"""
        instance = create_ec2_instance(
            aws_engine,
            instance_name,
            expert_instance_type,
            KeyName,
            ami_id
        )
        instance_meta = is_ec2_instance_present(aws_engine, instance_name )
        assert instance_meta['is_present']
    else:
        print("Instance in Initialization mode, please try after sometime.")
        return dict()
        
    """ connect to instance and start vllm server for the expert """
    try:
        ip_address = instance_meta['meta']['PublicIpAddress']
        ssh = get_ssh_session(ip_address, key_path=key_path)
    except Exception as e:
        print("Instance in Initialization mode, please try after sometime.")
        print(str(e))
        return dict()

    return dict(
        ip_address=ip_address,
        instance_name=instance_name
    )
    
    
def run_vllm_server(
    aws_engine,
    instance_name: str,
    model_id: str,
    tmux_session_name: str = "vllm_server",
    max_wait_time = 120,
    wait_for_expert_to_start=False,
    key_path = "app/keys/connection-key.pem"
):
    """
    Start a VLLM server for the provided model_id. Note model_id should be from HuggingFace.  
   
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        instance_name (str): The tag assigned to instance.
        model_id (str): Huggingface model_id to served.
        tmux_session_name (str, optional): Name assigned to tmux session running inference endpoints. Defaults to "vllm_server".  
        max_wait_time (int, optional): Wait time in seconds till the server starts.  
        wait_for_expert_to_start (bool, optional): Pause the Execution until the inference endpoint starts. Defaults to False.  
        key_path (str) : Path to the pem file required to connect to ec2 instance.  
    """
    
    """ connect to instance and start vllm server for the expert """
    start_time = time.time()
    while True:
        try:
            instance_meta = is_ec2_instance_present(aws_engine, instance_name )
            ip_address = instance_meta['meta']['PublicIpAddress']
            ssh = get_ssh_session(ip_address, key_path=key_path)
            break
        except:
            pass
        if time.time() - start_time > max_wait_time:
            print("Unable to ssh to instance, instace might be initializing, please try after some time or increase the max_wait_time")
            return
        else:
            time.sleep(15)

    start_vllm_server(
            ssh = ssh,
            model_id = model_id,
            ip_address = ip_address,
            port =8000,
            conda_env_name = "pytorch",
            tmux_session_name = tmux_session_name
        )
    
    """ keep waiting till expert starts """
    if wait_for_expert_to_start:
        check_vllm_server_running_status( ssh, tmux_session_name )
    
    response = dict(ip_address=ip_address,instance_name=instance_name)
    response.update(instance_meta['meta'])
    return response



def revive_ec2_instance(
        aws_engine,
        InstanceId
):
    """Revive an ec2 instance which is in "Stopped" state.

    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        InstanceId (str): Instance ID.
    """
    response = aws_engine.ec2.start_instances(
        InstanceIds=[
            InstanceId,
        ],
        DryRun=False
    )
    
def stop_ec2_instance(
        aws_engine,
        InstanceId
):
    """Stop an Ec2 Instance.
    NOTE:  This would change the status of ec2 instance to "stopped" and not "Terminate".  

    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        InstanceId (str): Instance ID.
    """
    response = aws_engine.ec2.stop_instances(
        InstanceIds=[
            InstanceId,
        ],
        DryRun=False
    )
    return response


def check_ec2_server_running_status(
    aws_engine,
    InstanceId
):
    """Wait until an ec2 instance is in 'running' state
    Args:
        aws_engine (AwsEngine): aws ec2 connection object.
        InstanceId (str): Instance ID.
    """
    start_time = time.time()
    time_out = 60*5
    instance_state = None
    while True:
        
        instance_state = get_instance(aws_engine, InstanceId)
        current_wait_time = (time.time() - start_time)
        status = \
            instance_state['Reservations'][0]['Instances'][0]['State']['Name'] \
                == "running"
        if status:
            print(f"EC2 Instance Running." )
            break
        elif current_wait_time > time_out:
            print("TimeOutError : EC2 Instance didnt start!")
            break
        else:
            print(f"Current wait time {current_wait_time}.  Max wait time {time_out}")
            time.sleep(10)
   
    return instance_state


def check_vllm_server_running_status(
    ssh: paramiko.SSHClient,
    tmux_session_name: str ="vllm_server",
    wait=True,
    verbose=True,
    docs_url=None
):
    """Check if VLLM insference endpoint has started.  

    Args:
        ssh (paramiko.SSHClient):    
        tmux_session_name (str, optional): Defaults to "vllm_server".
        wait (bool, optional): Wait unitl the VLLM server is in running state. Defaults to True.
        verbose (bool, optional): 
        docs_url (str, optional): This url can to used to check if the endpoint is callable. 
        If this isnt provided, the corresponding tmux pane is used for checks. Defaults to None`
    """
    start_time = time.time()
    time_out = 5#sec
    while True:
        
        current_wait_time = (time.time() - start_time)
        if docs_url:
            try:
                res = requests.get(docs_url)
                if res.status_code == 200:
                    is_running, server_status = True, "running"
            except:
                is_running, server_status = False, "connection error"
        else:
            is_running, server_status = is_running_vllm_server(ssh, tmux_session_name, verbose)
        
        if not wait:
            return is_running, server_status
        elif server_status == "running":
            print(f"VLLM SERVER RUNNING." )
            return server_status
        elif current_wait_time > time_out:
            print("TimeOutError : vllm server didnt started, Manual intervention needed!")
            return server_status
        else:
            print(f"Current wait time {current_wait_time}.  Max wait time {time_out}")
            time.sleep(10)
    