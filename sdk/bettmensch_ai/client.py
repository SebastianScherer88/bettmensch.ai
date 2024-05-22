import argo_workflows


def argo_client():
    configuration = argo_workflows.Configuration(host="https://127.0.0.1:2746")
    configuration.verify_ssl = False
    api_client = argo_workflows.ApiClient(configuration)

    return api_client


client = argo_client()
