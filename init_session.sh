 #!/bin/bash
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
echo "starting jupyter"
cmd="${sshcmd} '"'jupyter notebook &'"'"
# echo $cmd
eval $cmd
cmd="${sshcmd} '"'disown -h %1'"'"
# echo $cmd
eval $cmd
echo "open tunnel"
cmd="ssh -L localhost:8888:localhost:8888 -i ~/.ssh/lior.pem ec2-user@${addr}"
# echo "$cmd"
eval $cmd