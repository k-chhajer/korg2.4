# AWS Benchmark Setup

Status on 2026-04-05:

- AWS CLI configured for IAM user access on this machine
- No EC2 instance launched yet
- No billable compute started by this setup

Created resources:

- Key pair name: `arch1-g4dn-20260405`
- Private key path: `/Users/luthiraa/Documents/korg2.4/arch-1/aws/arch1-g4dn-20260405.pem`
- Security group id: `sg-095179c421536bfbb`
- Security group name: `arch1-benchmark-ssh-20260405`
- Launch template id: `lt-0e8580e0885e7c8c6`
- Launch template name: `arch1-g4dn-spot-20260405`

Network and region:

- Region: `us-east-2`
- Default VPC: `vpc-05d225075b7631b74`
- Recommended subnet: `subnet-07dd414026e4885ce` in `us-east-2a`
- SSH ingress is restricted to `52.154.22.53/32`

Launch template details:

- AMI: `ami-06517bc7fad3c6a48`
- AMI name: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403`
- Instance type: `g4dn.xlarge`
- Market: `spot`
- Root volume: `150 GB gp3`, encrypted, delete on termination
- Instance-initiated shutdown behavior: `terminate`

Bootstrap files:

- User data: `/Users/luthiraa/Documents/korg2.4/arch-1/aws/ec2_user_data.sh`
- Launch template data source: `/Users/luthiraa/Documents/korg2.4/arch-1/aws/launch_template_data.json`

Notes:

- This setup avoids compute charges until an instance is actually launched.
- AWS credits usually offset eligible charges, but that is an AWS billing decision and is not guaranteed by the CLI setup itself.
- The IAM access key used during setup should be rotated after provisioning is complete, since it was pasted into chat.

## us-east-1 Setup

Prepared because the GPU quota increase request was submitted in `us-east-1`.

- Region: `us-east-1`
- Default VPC: `vpc-0a4f286c94a9c249b`
- Recommended subnet: `subnet-0f53ea9bb9d1ad583` in `us-east-1a`
- AMI: `ami-057b641f1539dc1c4`
- AMI name: `Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260403`
- Key pair name: `arch1-g4dn-use1-20260405`
- Private key path: `/Users/luthiraa/Documents/korg2.4/arch-1/aws/arch1-g4dn-use1-20260405.pem`
- Security group id: `sg-03c2d49ae13951496`
- Security group name: `arch1-benchmark-ssh-use1-20260405`
- Launch template id: `lt-040c74ea268505a3e`
- Launch template name: `arch1-g4dn-spot-use1-20260405`

Files:

- `/Users/luthiraa/Documents/korg2.4/arch-1/aws/launch_template_data_use1.json`
- `/Users/luthiraa/Documents/korg2.4/arch-1/aws/ec2_user_data.sh`

Current blocker:

- EC2 GPU quota approval is still required before a `g4dn.xlarge` instance can be launched in `us-east-1`.
