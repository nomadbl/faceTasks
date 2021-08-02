import time

import boto3
import argparse
import yaml
import paramiko
import pdb
import sys
import subprocess
from datetime import datetime
import glob
import traceback
import os


def execute_commands_on_linux_instances(client, commands, instance_ids):
    """Runs commands on remote linux instances
    :param client: a boto/boto3 ssm client
    :param commands: a list of strings, each one a command to execute on the instances
    :param instance_ids: a list of instance_id strings, of the instances on which to execute the command
    :return: the response from the send_command function (check the boto3 docs for ssm client.send_command() )
    """

    resp = client.send_command(
        DocumentName="AWS-RunShellScript", # One of AWS' preconfigured documents
        Parameters={'commands': commands},
        InstanceIds=instance_ids,
    )
    return resp


def init_instance(params):
    ec2_resource = boto3.resource('ec2')
    instances = ec2_resource.create_instances(**params)
    instances[0].wait_until_running()
    instance = instances[0]
    instance.load()
    pub_dns_name = instance.public_dns_name
    instance_id = instance.instance_id
    return instance_id, pub_dns_name


def get_dataset(s3_path):
    s3_resource = boto3.resource('s3')


def term_instance(instance_id):
    ec2_client = boto3.client('ec2')
    ec2_client.terminate_instances(InstanceIds=[instance_id])


def train(path):
    # get ec2 parameters
    with open(path, 'r') as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # initialize instance
    instance_id, pub_dns_name = init_instance(params)
    key_name = params['KeyName']
    # commands = []
    home_folder = os.getenv("HOME")
    keyfile = f'{home_folder}/.ssh/{key_name}.pem'
    key = paramiko.RSAKey.from_private_key_file(keyfile)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print("waiting for startup...(30 sec)")
    time.sleep(30)
    print(f"connecting to ec2 {pub_dns_name}")

    # now, connect and use paramiko Transport to negotiate SSH2 across the connection
    try:
        client.connect(hostname=pub_dns_name, username='ec2-user', pkey=key)
        sftp = client.open_sftp()
        print("connection established")
        aws_files = glob.glob(f'{home_folder}/.aws/*')
        # 1. create session folder (contains all files for current model except dataset)
        current_time = datetime.today().strftime('%Y-%m-%d-%H:%M')
        session_path = f'session_{current_time}/'
        sftp.mkdir(session_path)
        # 2. grant s3 access
        sftp.mkdir(f"{home_folder}/.aws")
        for file in aws_files:
            sftp.put(file, file)
        # 3. upload pynb file
        nb_path = f'{home_folder}/PycharmProjects/facesTasks/facial-segmentation-unlabeled-AWS.ipynb'
        nb_name = nb_path.split(sep='/')[-1]
        sftp.put(nb_path, f'{session_path}{nb_name}')
        sftp.close()

        # 4. download dataset from s3
        command = 'aws s3 cp s3://nomadblfaces/data data.zip'
        print("Executing {}".format(command))
        stdin, stdout, stderr = client.exec_command(command)
        print(stdout.read())
        print("Errors")
        print(stderr.read())
        # 5. unzip data into images folder (contained in the file)
        command = 'unzip data.zip'
        print("Executing {}".format(command))
        stdin, stdout, stderr = client.exec_command(command)
        print(stdout.read())
        print("Errors")
        print(stderr.read())
        # 6. activate tensorflow
        command = 'source activate tensorflow2_latest_p37'
        print("Executing {}".format(command))
        stdin, stdout, stderr = client.exec_command(command)
        print(stdout.read())
        print("Errors")
        print(stderr.read())
    except Exception as e:
        print("*** Caught exception: %s: %s" % (e.__class__, e))
        traceback.print_exc()
        client.close()
        sys.exit(1)
    finally:
        client.close()

    # 7. connect with ssh
    cmd = f'ssh -L localhost:8888:localhost:8888 -i ~/.ssh/{key_name}.pem ec2-user@{pub_dns_name}'
    print("connect to server using command:")
    print(cmd)

    # # 6. start jupyter notebook
    # commands.extend('jupyter notebook')
    # 7. run notebook
    # 8. download outputs to s3 and local
    # stop instance
    # term_instance(instance_id)


def setup(path):
    with open(path+'/sc2_params.yaml', 'w') as f:
        data = {
            'ImageId': 'ami-0d1981233d9ae1ea5',  # default Deep Learning AMI (Amazon Linux 2) Version 49.0
            'MinCount': 1,
            'MaxCount': 1,
            'InstanceType': 'g4dn.xlarge',
            'KeyName': 'lior',
            # 'IAMrole': 'ec2_s3_full_access',
            'BlockDeviceMappings': [
                {
                    'DeviceName': "/dev/xvda",
                    'Ebs': {
                        'DeleteOnTermination': True,
                        'VolumeSize': 105
                    }
                }
            ],
            'SecurityGroups': ['DL']
        }
        yaml.dump(data, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run training job on EC2 instance and terminate on error or finish')
    subparsers = parser.add_subparsers()
    parser_setup = subparsers.add_parser('setup', help='generate default configuration yml')
    parser_setup.set_defaults(func=setup)
    parser_setup.add_argument('path', type=str, help='path to put file')
    parser_train = subparsers.add_parser('train', help='run training job on EC2')
    parser_train.set_defaults(func=train)
    parser_train.add_argument('path', type=str, help='yaml conf file')
    args = parser.parse_args()
    args.func(args.path)
