import json
import os

from nvflare.lighter.entity import Participant
from nvflare.lighter.spec import Builder, Project, ProvisionContext


class CustomBuilder(Builder):
    def __init__(self):
        Builder.__init__(self)

    def build(self, project: Project, ctx: ProvisionContext):
        proxies = project.get_all_participants("proxy")
        if not proxies:
            ctx.info("no proxies to build!")
        for p in proxies:
            assert isinstance(p, Participant)
            dest_dir = ctx.get_kit_dir(p)
            data = {"name": p.name, "url": p.get_prop("url", "flare.com/custom")}
            with open(os.path.join(dest_dir, "fed_proxy.json"), "w") as f:
                json.dump(data, f)
