import json
import time
from app.orchestrator import Orchestrator


config = json.load(open("app/configs/leeroo.json", "r"))

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
