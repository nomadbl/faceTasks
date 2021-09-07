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

#cmd1="${sshcmd} '"'source activate tensorflow2_latest_p37'"'"
#cmd2="${sshcmd} '"'/home/ec2-user/anaconda3/envs/tensorflow2_latest_p37/bin/python -m pip install --upgrade pip'"'"
#cmd3="${sshcmd} '"'pip install --upgrade tensorflow --pre'"'"
#echo "upgrading tensorflow"
#eval $cmd1
#eval $cmd2

cmd1="${sshcmd} '"'source activate pytorch_latest_p37 && yes | pip uninstall torch && yes | pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html && pip3 install tensorboard'"'"
echo "upgrade pytroch"
eval $cmd1

#echo "run: jupyter notebook & -> disown -h %1"
echo "open tunnel"
#echo "$cmd"
cmd="ssh -L localhost:6006:localhost:6006 -i ~/.ssh/lior.pem ec2-user@${addr}"
echo $cmd
eval $cmd