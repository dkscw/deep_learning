import sys
import os
import boto3
from fabric.api import env

EC2_INSTANCE_ID = os.environ['DEFAULT_EC2_INSTANCE_ID']
EC2_USERNAME = os.environ['DEFAULT_EC2_USERNAME']
EC2_KEY_FILENAME = os.environ['DEFAULT_EC2_KEY_FILENAME']

# Set the ec2 username and key filename for fabric use
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
            print "Instance started, waiting until running... (^C to run in background)"
            try:
                self.instance.wait_until_running()
            except KeyboardInterrupt:
                pass

    def stop_instance(self, wait=False):
        """ Stop the instance. If wait is True, wait until instance is stopped. """
        self.instance.stop()
        if wait:
            print "Instance stopping, waiting until stopped... (^C to run in background)"
            try:
                self.instance.wait_until_stopped()
            except KeyboardInterrupt:
                pass
    
    def set_host(self, instance_id=EC2_INSTANCE_ID):
        " Set the instance as the fabric host "
        if self.instance.public_ip_address is None:
            print "Starting instance... "
            self.start_instance(wait=True)
            print "Instance running at IP {}".format(self.instance.public_ip_address)

        env.hosts = [self.instance.public_ip_address]

ec2_manager = EC2InstanceManager(EC2_INSTANCE_ID)

def ec2_host(task):
    "Decorator that wraps fabric tasks adding an ec2 host to the list of hosts"
    if ec2_manager.instance.public_ip_address is None:
        raise RuntimeError("EC2 instance not running")
    if ec2_manager.instance.public_ip_address not in env.hosts:
        env.hosts.append(ec2_manager.instance.public_ip_address)
    return task

if __name__ == '__main__':
    "Start / stop EC2 instance"
    if sys.argv[1] == 'start':
        ec2_manager.start_instance(wait=True)
    elif sys.argv[1] == 'stop':
        ec2_manager.stop_instance(wait=True)
    else:
        print "Invalid argument. Valid arguments are 'start', 'stop'."

