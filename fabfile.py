from fabric.api import env, task, run
import boto3
import os

EC2_INSTANCE_ID = os.environ['DEFAULT_EC2_INSTANCE_ID']
EC2_USERNAME = os.environ['DEFAULT_EC2_USERNAME']
EC2_KEY_FILENAME = os.environ['DEFAULT_EC2_KEY_FILENAME']

env.user = EC2_USERNAME
env.key_filename = EC2_KEY_FILENAME


class EC2InstanceManager(object):
    """ Simple class to interact with an EC2 instance """

    def __init__(self, instance_id=EC2_INSTANCE_ID):
        self.instance_id = instance_id
        self.instance = boto3.resource('ec2').Instance(self.instance_id)

    def start_instance(self, wait=False):
        """ Start the instance. If wait is True, wait until instance is running. """
        self.instance.start()
        if wait:
            self.instance.wait_until_running()

    def stop_instance(self, wait=False):
        """ Stop the instance. If wait is True, wait until instance is stopped. """
        self.instance.stop()
        if wait:
            self.instance.wait_until_stopped()
    
    def set_host(self, instance_id=EC2_INSTANCE_ID):
        " Set the instance as the fabric host "
        if self.instance.public_ip_address is None:
            print "Starting instance... "
            self.start_instance(wait=True)
            print "Instance running at IP {}".format(self.instance.public_ip_address)

        env.hosts = [self.instance.public_ip_address]

ec2_manager = EC2InstanceManager(EC2_INSTANCE_ID)
ec2_manager.set_host()

@task
def get_info():
    run("uname -a")

@task
def stop_instance():
    ec2_manager.stop_instance(wait=True)
