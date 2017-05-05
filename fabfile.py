from fabric.api import env, task, run
import boto3
import os

from aws_manager import ec2_host

@ec2_host
@task
def get_info():
    run("uname -a")

@ec2_host
@task
def download_from_s3():
    run("mkdir data")
    run("aws s3 cp --recursive s3://danielshenf-deeplearning/data data")

def install_git():
    run("sudo yum install git")

@ec2_host
@task
def pull_repo():
    install_git()  # does nothing If git is already present
    run("git clone https://github.com/dkscw/deep_learning.git")