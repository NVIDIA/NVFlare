from gunicorn.workers.sync import SyncWorker


class ClientAuthWorker(SyncWorker):
    def handle_request(self, listener, req, client, addr):
        cert = client.getpeercert()
        subject = client.getpeercert().get("subject")
        commonName = next(value for ((key, value),) in subject if key == "commonName")
        headers = dict(req.headers)
        headers["X-USER"] = commonName
        req.headers = list(headers.items())

        super(ClientAuthWorker, self).handle_request(listener, req, client, addr)
