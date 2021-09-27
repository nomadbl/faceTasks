import subprocess
import sys
import argparse


def init(args):
    addr = args.addr
    s3 = args.s3
    print("Copy s3 privileges")

    cmd = f"scp -i ~/.ssh/lior.pem -r ~/.aws ec2-user@{addr}:~/.aws"
    # print(cmd)
    subprocess.run(cmd, shell=True)

    sshcmd = f"ssh -i ~/.ssh/lior.pem ec2-user@{addr}"

    print("preparing session")
    if s3:
        # get checkpoints etc from s3
        cmd = f'{sshcmd} "aws s3 cp s3://nomadblfaces/session session.zip & unzip session.zip"'
        subprocess.run(cmd, shell=True)
    # get latest version of python files from local machine
    cmd = f"scp -i ~/.ssh/lior.pem session.zip ec2-user@{addr}:~/session.zip"
    subprocess.run(cmd, shell=True)
    cmd = f'{sshcmd} "yes A | unzip session.zip"'
    subprocess.run(cmd, shell=True)

    print("Copied data from s3")
    print("unpacking session")
    cmd = f'{sshcmd} "unzip session.zip"'
    subprocess.run(cmd, shell=True)

    cmd = f'{sshcmd} "source activate pytorch_latest_p37 && yes | pip uninstall torch && yes | pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && pip3 install tensorboard"'
    print("upgrade pytroch")
    subprocess.run(cmd, shell=True)

    print("set up aliases: tb, train, disp, backup")
    cmd = f"scp -i ~/.ssh/lior.pem aliases ec2-user@{addr}:~/aliases"
    subprocess.run(cmd, shell=True)
    cmd = f'{sshcmd} "cat aliases >> .bashrc"'
    subprocess.run(cmd, shell=True)

    print("open tunnel")
    cmd = f"ssh -L localhost:6006:localhost:6006 -i ~/.ssh/lior.pem ec2-user@{addr}"
    subprocess.run(cmd, shell=True)


def retrieve(args):
    addr = args.addr
    print("packing session")
    sshcmd = f"ssh -i ~/.ssh/lior.pem ec2-user@{addr}"
    cmd = f'{sshcmd} "rm ~/session.zip && zip -r ~/session.zip ~/session/"'
    subprocess.run(cmd, shell=True)

    print("downloading session")
    cmd = f"scp -i ~/.ssh/lior.pem ec2-user@{addr}:~/session.zip session.zip"
    subprocess.run(cmd, shell=True)


def connect(args):
    addr = args.addr
    print("open tunnel")
    cmd = f"ssh -L localhost:6006:localhost:6006 -i ~/.ssh/lior.pem ec2-user@{addr}"
    subprocess.run(cmd, shell=True)


def upload(args):
    addr = args.addr
    file = args.file
    print(f"uploading {file}")
    file_target = "~/session/"+file.split(sep='/')[-1]
    cmd = f"scp -i ~/.ssh/lior.pem {file} ec2-user@{addr}:{file_target}"
    print(cmd)
    subprocess.run(cmd, shell=True)


def s3(args):
    addr = args.addr

    sshcmd = f"ssh -i ~/.ssh/lior.pem ec2-user@{addr}"

    print("backup session")
    cmd = f'{sshcmd} "rm ~/session.zip && zip -r ~/session.zip ~/session/ && aws s3 cp session.zip s3://nomadblfaces/session"'
    subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("init_session")

    subparsers = parser.add_subparsers(help="manage aws training session")
    init_parser = subparsers.add_parser("init", help="initialize aws session")
    connect_parser = subparsers.add_parser(
        "connect", help="connect to server")
    retrieve_parser = subparsers.add_parser(
        "retrieve", help="download session file from server")
    upload_parser = subparsers.add_parser(
        "upload", help="upload file to server")
    sss_parser = subparsers.add_parser(
        "s3", help="backup session on s3")

    init_parser.add_argument(dest="addr", type=str,
                             help="ec2 instance ipv4 address")
    init_parser.add_argument("--s3", dest="s3", action="store_true",
                             help="get session from s3")
    init_parser.set_defaults(func=init)
    connect_parser.add_argument(
        dest="addr", type=str, help="ec2 instance ipv4 address")
    connect_parser.set_defaults(func=connect)
    retrieve_parser.add_argument(
        dest="addr", type=str, help="ec2 instance ipv4 address")
    retrieve_parser.set_defaults(func=retrieve)
    upload_parser.add_argument(
        dest="addr", type=str, help="ec2 instance ipv4 address")
    upload_parser.add_argument(
        dest="file", type=str, help="path of file to upload")
    upload_parser.set_defaults(func=upload)
    sss_parser.add_argument(dest="addr", type=str,
                            help="ec2 instance ipv4 address")
    sss_parser.set_defaults(func=s3)
    args = parser.parse_args()
    args.func(args)
