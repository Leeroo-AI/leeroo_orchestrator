import os
import io
import json
import boto3
from dotenv import load_dotenv
load_dotenv("app/.env")

class AwsSagemakerEngine:
    """AwsEngine uses boto3 for sagemaker connections
    """
    def __init__(self,config):
        self.iam = boto3.client(
            'iam',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=config.get("region", "us-east-1")
        )
        self.role = self.iam.get_role(
            RoleName=os.getenv('AWS_SAGEMAKER_ROLE_NAME'))['Role']['Arn']

        self.sagemaker_runtime = boto3.client(
            'sagemaker-runtime',         
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=config.get("region", "us-east-1")
        )
        self.sagemaker_client = boto3.client(
            'sagemaker',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=config.get("region", "us-east-1")
        )

class LineIterator:
    """
    A helper class for parsing the byte stream input. 
    
    The output of the model will be in the following format:
    
        {"outputs": [" a"]}   
        {"outputs": [" challenging"]}   
        {"outputs": [" problem"]}   
        ...
    
    
    While usually each PayloadPart event from the event stream will contain a byte array 
    with a full json, this is not guaranteed and some of the json objects may be split across
    PayloadPart events. For example:
    
        {'PayloadPart': {'Bytes': {"outputs": '}}   
        {'PayloadPart': {'Bytes': [" problem"]}'}}   
    
    
    This class accounts for this by concatenating bytes written via the 'write' function
    and then exposing a method which will return lines (ending with a '\\n' character) within
    the buffer via the 'scan_lines' function. It maintains the position of the last read 
    position to ensure that previous bytes are not exposed again. 
    """
    
    def __init__(self, stream):
        self.byte_iterator = iter(stream)
        self.buffer = io.BytesIO()
        self.read_pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            self.buffer.seek(self.read_pos)
            line = self.buffer.readline()
            if line and line[-1] == ord('\n'):
                self.read_pos += len(line)
                return line[:-1]
            try:
                chunk = next(self.byte_iterator)
            except StopIteration:
                if self.read_pos < self.buffer.getbuffer().nbytes:
                    continue
                raise
            if 'PayloadPart' not in chunk:
                print('Unknown event type:' + chunk)
                continue
            self.buffer.seek(0, io.SEEK_END)
            self.buffer.write(chunk['PayloadPart']['Bytes'])

def get_realtime_response_stream(sagemaker_runtime, endpoint_name, payload):
    """Fetch Streaming Text Generation response from AWS Sagemaker Enpoint.  

    Args:
        sagemaker_runtime (AwsSagemakerEngine.sagemaker_runtime):   
        endpoint_name (str): Endpoint Name.
        payload (json): Request

    Returns:
        _type_: _description_
    """
    response_stream = sagemaker_runtime.invoke_endpoint_with_response_stream(
        EndpointName=endpoint_name,
        Body=json.dumps(payload), 
        ContentType="application/json",
        CustomAttributes='accept_eula=true'
    )
    return response_stream

def parse_response_stream(response_stream, verbose=True):
    """Prints Streaming Text

    Args:
        response_stream : response stream
        verbose (bool, optional): _description_. Defaults to True.

    Returns:
        resonse (str): Generated Text.
    """
    response_text = ""
    event_stream = response_stream['Body']
    start_json = b'{'
    stop_token = '</s>'
    for line in LineIterator(event_stream):
        if line != b'' and start_json in line:
            data = json.loads(line[line.find(start_json):].decode('utf-8'))
            if data['token']['text'] != stop_token:
                response_text += data['token']['text']
                if verbose: print(data['token']['text'],end='')
    return response_text  

def get_hf_image(boto_region_name):
    region_mapping = {
        "af-south-1": "626614931356",
        "il-central-1": "780543022126",
        "ap-east-1": "871362719292",
        "ap-northeast-1": "763104351884",
        "ap-northeast-2": "763104351884",
        "ap-northeast-3": "364406365360",
        "ap-south-1": "763104351884",
        "ap-south-2": "772153158452",
        "ap-southeast-1": "763104351884",
        "ap-southeast-2": "763104351884",
        "ap-southeast-3": "907027046896",
        "ap-southeast-4": "457447274322",
        "ca-central-1": "763104351884",
        "cn-north-1": "727897471807",
        "cn-northwest-1": "727897471807",
        "eu-central-1": "763104351884",
        "eu-central-2": "380420809688",
        "eu-north-1": "763104351884",
        "eu-west-1": "763104351884",
        "eu-west-2": "763104351884",
        "eu-west-3": "763104351884",
        "eu-south-1": "692866216735",
        "eu-south-2": "503227376785",
        "me-south-1": "217643126080",
        "me-central-1": "914824155844",
        "sa-east-1": "763104351884",
        "us-east-1": "763104351884",
        "us-east-2": "763104351884",
        "us-gov-east-1": "446045086412",
        "us-gov-west-1": "442386744353",
        "us-iso-east-1": "886529160074",
        "us-isob-east-1": "094389454867",
        "us-west-1": "763104351884",
        "us-west-2": "763104351884",
    }
    llm_image = f"{region_mapping[boto_region_name]}.dkr.ecr.{boto_region_name}.amazonaws.com/huggingface-pytorch-tgi-inference:2.1.1-tgi1.3.1-gpu-py310-cu121-ubuntu20.04-v1.0"
    # print ecr image uri
    print(f"llm image uri: {llm_image}")
    return llm_image