# Image pulling failed from GHCR due to no proper secret information
cat secrets/token.txt | docker login https://ghcr.io -u mrsyee --password-stdin

docker_password=$(cat secrets/token.txt)

kubectl create secret docker-registry regcred \
    --docker-server=https://ghcr.io \
    --docker-username="mrsyee" \
    --docker-email="khsyee@gmail.com"  \
    --docker-password="$docker_password"
