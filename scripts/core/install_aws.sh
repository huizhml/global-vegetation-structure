PROJECT=/project/project_465002698
AWS_INSTALL=$PROJECT/software/aws-cli
AWS_BIN=$PROJECT/software/bin

mkdir -p "$AWS_BIN"

TMPDIR_AWS=$(mktemp -d /tmp/awscli.XXXXXX)
cd "$TMPDIR_AWS"

curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" \
  -o awscliv2.zip

unzip -q awscliv2.zip

./aws/install \
  --install-dir "$AWS_INSTALL" \
  --bin-dir "$AWS_BIN"

rm -rf "$TMPDIR_AWS"

export PATH="$AWS_BIN:$PATH"

aws --version