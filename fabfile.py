from fabric.api import env, task, run, settings
import boto3
import os

from aws_manager import ec2_host

class FabricException(Exception):
    pass


@ec2_host
@task
def get_info():
    run("uname -a")


@ec2_host
@task
def mount_ebs_volume():
    with settings(abort_exception = FabricException):
        try:
            run("sudo mkdir /data")
        except FabricException:
            pass
    run("sudo mount /dev/xvdf /data")


@ec2_host
@task
def set_data_dir():
    run("export KAGGLE_DATA_PATH=/data/Kaggle-planet")
    # os.environ['KAGGLE_DATA_PATH'] = '/data/Kaggle-planet'


def install_git():
    run("sudo yum install git")

def clone_repo():
    run("sudo yum install git")

@ec2_host
@task
def pull_repo():
    with settings(abort_exception = FabricException):
        try:
            install_git()
            print "Installed git"
        except FabricException:
            print "Git already installed"

        try:
            clone_repo()
            print "Cloned repo"
        except FabricException:
            print "Repo already cloned"



@ec2_host
@task
def install_requirements():
    # Note: ec2 AMIs come with an old version of pip and upgrading will
    # screw up the installation directory... see https://stackoverflow.com/questions/34103119/upgrade-pip-in-amazon-linux
    # Need to install gcc separately using the command:
    # sudo yum groupinstall "Development Tools"
    run("sudo pip install -r requirements.txt")


def prepare_new_instance():
    install_git()
    pull_repo()
    install_requirements()

# @ec2_host
# @task
# def download_from_s3():
#     run("mkdir data")
#     run("aws s3 cp --recursive s3://danielshenf-deeplearning/data data")