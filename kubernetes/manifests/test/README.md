# Overview

To test the karpenter installation against manual scaling requests of pods/deployments,
run

```bash

# Second, scale the example deployment
kubectl scale deployment scale-test-1 --replicas 200

# You can watch Karpenter's controller logs with
kubectl logs -f -n karpenter -l app.kubernetes.io/name=karpenter -c controller
```