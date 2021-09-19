import subprocess
import sys
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser("initiate aws training session")
    parser.add_argument(dest="addr", type=str)
    args = parser.parse_args()
    addr = args.addr
    print("Copy s3 privileges")
    cmd = f"scp -i ~/.ssh/lior.pem -r ~/.aws ec2-user@{addr}:~/.aws"
    # print(cmd)
    subprocess.run(cmd, shell=True)

    cmd = f"scp -i ~/.ssh/lior.pem session.zip ec2-user@{addr}:~/session.zip"
    subprocess.run(cmd, shell=True)

    sshcmd = f"ssh -i ~/.ssh/lior.pem ec2-user@{addr}"

    print("Copy data from s3...")
    cmd = f'{sshcmd} "aws s3 cp s3://nomadblfaces/data data.zip; unzip data.zip"'
    # print(cmd)
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
