from urllib.parse import urlparse


class XYZDriver(DriverSpec):

    resources = {}

    @classmethod
    def get_name(cls):
        return "XYZ"

    @classmethod
    def new_connector(cls, reqs: dict) -> XYZDriver:
        pass

    @classmethod
    def new_listener(cls, reqs: dict):
        url = reqs["url"]
        type = reqs["conn_type"]
        url_attrs = urlparse(url)
        if url_attrs.scheme != 'xyz':
            return None

        parts = url_attrs.netloc.split(':')
        port = None
        if len(parts) == 2:
            port = int(parts[1])
        else:
            available_ports = cls.resources.get("ports", [])
            selected = None
            for p in available_ports:
                # see whether this port is available
                port = p

        if not port:
            return None

        return XYZDriver(port)

    @classmethod
    def set_resources(cls, resources: dict):
        cls.resources = resources


