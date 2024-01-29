# [Leeroo Orchestrator](https:///www.leeroo.com/)




For more details refer the paper [**Leeroo Orchestrator: Elevating LLMs Performance Through Model Integration**](https://arxiv.org/abs/2401.13979).

## Project layout

* `orchestrator.py` - The Orchestrator serves as the central wrapper object, facilitating seamless coordination among expert models.
* `server_manager.py` - Manages the servers implemented in the `./servers/` directory.
* `test_orchestrator.py` - Provides an example of running the orchestrator for testing purposes.
* `abstract_classes.py` - Contains base classes for servers, serving as foundational structures for server implementations.
* `./configs/*` - Holds configuration files used by the orchestrator for various settings.
* `./servers/*` - Houses all server implementations, each dedicated to serving specific serving strategy/machine i.e Ec2 vllm/sagemaker/... etc.
* `./utils/*` - Includes utility functions used by servers to enhance functionality.
* `.env` - Specifies the required environment variables for proper system operation.

```pyhton

AWS_SAGEMAKER_ROLE_NAME = "*****"   # Role is used for sagemaker connections.
SECURITY_GROUP_ID       = "****"    # SecurityGroupId is used to create an instance. make sure to have inference port open for this group.
AWS_ACCESS_KEY_ID       = "*****"   # 
AWS_SECRET_ACCESS_KEY   = "*****"   #
HUGGING_FACE_HUB_TOKEN  = "*****"   # Required for authenticating with HF for downloading checkopints
OPENAI_ORGANIZATION     = "*****"
OPENAI_API_KEY          = "*****"   # If openai models are used as experts
```


## Run Orchestrator server  
```python
import json
import time
from app.orchestrator import Orchestrator


config = json.load(open("app/configs/demo_orch_sagemaker_mix.json", "r"))

# init
leeroo_orchestrator = Orchestrator(config)

# boot the machines
leeroo_orchestrator.load_orchestrator_server()
leeroo_orchestrator.load_experts_server()

# start the inference endpoints
leeroo_orchestrator.start_inference_endpoints(max_wait_time=120)


# Wait until all endpoints are up
status = False
while not status:
    print("Checking server status...")
    status = leeroo_orchestrator.check_servers_state()
    if status:
        print("Servers are running...")
        break
    time.sleep(30)

# Test get_response for all the servers
for expert in leeroo_orchestrator.experts.values():
    print(expert.model_id)
    print(expert.get_response("hello"))

# Test get_response for complete pipeline
response = leeroo_orchestrator.get_response("What is the capital of India?")
print(response)

# turn off the machines
leeroo_orchestrator.orchestrator.stop_server()
for expert_id, expert in leeroo_orchestrator.experts.items():
    res = expert.stop_server()
    print(res)

print("done!")
```

