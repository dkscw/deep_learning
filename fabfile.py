from fabric.api import env, task, run
import boto3
import os

from aws_manager import ec2_host

@ec2_host
@task
def get_info():
    run("uname -a")
