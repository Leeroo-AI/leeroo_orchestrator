# **VLLM-PYTORCH2-LEEROO**
# AMI ID: **ami-06ff97c8d51fa0dcd** 
## Version: 0.0.1  
### Release Date: 29 Jan 2024

## Overview
This AWS AMI is specifically designed for serving Hugging Face models. VLLM (Very Large Language Model) is pre-installed for model-serving capabilities. The conda environment : 'pytorch' should be used for running vllm server.  

## AMI Details
- **AMI ID:** *ami-06ff97c8d51fa0dcd*
- **AMI Name:** *VLLM-PYTORCH2-LEEROO*
- **Base AMI Name:** Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.1.0 (Ubuntu 20.04) 20240116
- **Architecture:** x86_64
- **Release Type:** Stable

## Prerequisites
- Users should have an AWS account.
- Prepare a security keypair file and a Security Group using your credentials.

## Usage Instructions  
Please refer : app/test_orchestrator.py  

## Usage Instructions (without leeroo orch)  
1. Launch an EC2 instance using this AMI.
2. Use the security key file created with your credentials for SSH access.
3. Ensure that the security group allows SSH from any IP and has port 8000 open.

## Security Considerations   
- Users must create and use a secure key file for SSH access. [Create EC2 key pairs](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/create-key-pairs.html#having-ec2-create-your-key-pair)
- Security group should allow SSH from any IP and have port 8000 open. [Create EC2 Security Groups for Linux instances](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html)



---

Thank you for choosing our AWS AMI! If you have any questions or feedback, please don't hesitate to contact our support team at www.leeroo.com.
