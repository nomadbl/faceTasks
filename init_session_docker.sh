#!/bin/bash
echo 'usage: ec2init {public ipv4 adddress}'
echo "Copy s3 privileges"
addr=$1
cmd="scp -i ~/.ssh/lior.pem -r ~/.aws ec2-user@${addr}:~/.aws"
# echo $cmd
eval $cmd
cmd="scp -i ~/.ssh/lior.pem session.zip ec2-user@${addr}:~/session.zip"
# echo $cmd
eval $cmd
sshcmd="ssh -i ~/.ssh/lior.pem ec2-user@${addr}"
echo "Copy data from s3..."
cmd="${sshcmd} '"'aws s3 cp s3://nomadblfaces/data data.zip; unzip data.zip'"'"
# echo $cmd
eval $cmd
echo "Copied data from s3"
echo "unpacking session"
cmd="${sshcmd} '"'unzip session.zip'"'"
# echo $cmd
eval $cmd

cmd1="${sshcmd} '"'docker pull tensorflow/tensorflow:latest-gpu-jupyter'"'"
cmd2="${sshcmd} '"'docker run -it --rm -v $PWD:/tmp -w /tmp -p 8888:8888 -p 6006:6006 tensorflow/tensorflow:latest-gpu'"'"
eval cmd1
eval cmd2

echo "run: jupyter notebook & -> disown -h %1"
echo "docker run -it --rm -v $PWD:/tmp -w /tmp -p 8888:8888 tensorflow/tensorflow:latest-gpu-jupyter"
echo "open tunnel"
echo "$cmd"
cmd="ssh -L localhost:8888:localhost:8888 -i ~/.ssh/lior.pem ec2-user@${addr}"
eval $cmd