# deep_learning

### Managing EC2 instances
Use `aws_manager.py` to start and stop your instance:
```
python aws_manager.py start
python aws_manager.py stop
```
Ensure the proper environment variables are set.

- Use the decorator `ec2_host` on a fabric task to run on the ec2 host.

#### Granting S3 access from the instance
- From the IAM console, create a role to grant EC2 instances permissions to S3
- Assign this role to the instance