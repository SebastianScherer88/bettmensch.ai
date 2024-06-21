export HEAD_POD=$(kubectl get pods --selector=app=argo-server -n argo -o custom-columns=POD:metadata.name --no-headers)
echo $HEAD_POD

kubectl -n argo port-forward $HEAD_POD --address 0.0.0.0 2746:2746
# Check $YOUR_IP:2746 in your browser