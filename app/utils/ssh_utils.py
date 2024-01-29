import time
import paramiko

def get_ssh_session(
    ip_address: str, 
    timeout:int =15,
    key_path:str ="app/keys/connection-key.pem"
):
    """Get paramiko SSHClient  
     
    Args:
        ip_address (str): ip address
        timeout (int, optional): Defaults to 15.
        key_path (str) : Path to the pem file required to connect to ec2 instance. 
    """
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(ip_address, 
                username="ubuntu", 
                key_filename=key_path,
                timeout=timeout)
    return ssh


def execute_ssh_command(
        ssh: paramiko.SSHClient,
        cmd: str,
        verbose: bool =True
    ):
    """Runs the given command in paramiko ssh client 
    
    Args:
        ssh (paramiko.SSHClient): The SSH client for the connection.
        cmd (str): Command that can run in tmux session terminal.
    """
    if verbose: print( f"ssh : {cmd}" )
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cmd)
    ssh_stdout_str = ssh_stdout.read().decode()
    ssh_stderr_str = ssh_stderr.read().decode()
    if verbose:
        if ssh_stdout_str:
            print("\t stdout : ", ssh_stdout_str)
        else:
            print("\t stderr : ", ssh_stderr_str)
    return ssh_stdin, ssh_stdout_str, ssh_stderr_str


def start_vllm_server(
        ssh: paramiko.SSHClient,
        model_id: str,
        ip_address: str,
        port: int =8000,
        conda_env_name: str = "pytorch",
        tmux_session_name: str = "vllm_server"
    ):
        
    """Starts a VLLM Server for the provided Model ID.

    Args:
        ssh (paramiko.SSHClient):  The SSH client for the connection.
        model_id (str): HF supported model ID. TODO: Support s3 checkpoint path downloading.
        ip_address (str): The IP address for the EC2 instance.
        port (str): VLLM runs a server that listens on this port. Ensure that this port is open for requests.
        conda_env_name (str): The specified environment should have VLLM installed.
        tmux_session_name (str): This is the name of the tmux session that runs the VLLM server.
    """
    
    """check if session is already created"""
    _, ssh_stdout, _ = execute_ssh_command(ssh,"tmux ls")
    if tmux_session_name not in ssh_stdout:
        """start a new tmux session"""
        _,_,_ = execute_ssh_command(ssh,  
                    f"tmux new -d -s {tmux_session_name}" )
        time.sleep(5)
        
        """activate environment"""
        _, _, _ = execute_ssh_command(ssh, 
                    f"tmux send-keys -t {tmux_session_name}.0 'conda activate {conda_env_name}' ENTER")
        time.sleep(5)
    else:
        print("tmux session was found :")
        print(ssh_stdout)
    
    
    """confirm if session is created"""
    _, ssh_stdout, _ = execute_ssh_command(ssh,"tmux ls")
    assert tmux_session_name in ssh_stdout
    print(ssh_stdout)
    
    
    """check if vllm server is already running"""
    _,vllm_status = \
        is_running_vllm_server( ssh,  tmux_session_name)
    if vllm_status not in ['running', 'loading']:
        """start vllm server"""
        print( "Vllm server starting... " )
        start_vllm_server = f"python -m vllm.entrypoints.openai.api_server --model {model_id} --port {port}"
        _, _, _ = execute_ssh_command(ssh,
            f"tmux send-keys -t {tmux_session_name}.0 '{start_vllm_server}' ENTER")
        _,vllm_status = \
            is_running_vllm_server( ssh,  tmux_session_name)
        print( f"vllm status {vllm_status}" )
    else:
        print( f"vllm status {vllm_status}" )


def is_running_vllm_server(
        ssh: paramiko.SSHClient,
        tmux_session_name: str = "vllm_server",
        verbose: bool =True
    ):
    """Check if vllm server has started successfully.  
    
    Args:
        ssh: (paramiko.SSHClient):  The SSH client for the connection.
        tmux_session_name (str, optional): Defaults to "vllm_server".
    """
    ssh_stdin, ssh_stdout, ssh_stderr = capture_tmux_pane(ssh, tmux_session_name, verbose)
    if "Uvicorn running on http" in ssh_stdout:
        return True, "running"
    elif "python -m vllm" in ssh_stdout or \
            "llm_engine.py" in ssh_stdout:
        return False, "loading"
    else:
        return False, ""


def capture_tmux_pane(
    ssh: paramiko.SSHClient,
    tmux_session_name: str = "vllm_server",
    verbose: bool =True
):
    """
    Capture the content of the specified tmux pane.

    Args:
        ssh (paramiko.SSHClient): The SSH client for the connection.
        tmux_session_name (str, optional): The name of the tmux session. Defaults to "vllm_server".
        verbose (bool, optional): If True, display verbose output. Defaults to True.

    Returns:
        tuple: A tuple containing stdin, stdout, and stderr from the SSH command execution.
    """
    ssh_stdin, ssh_stdout, ssh_stderr = execute_ssh_command(ssh,
        f"tmux capture-pane -p -t {tmux_session_name}.0", verbose)
    return ssh_stdin, ssh_stdout, ssh_stderr