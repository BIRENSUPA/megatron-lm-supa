import os

os.environ["NCCL_TOPO_FILE"] = os.getenv("BIREN_HOME", "/usr/local/birensupa/all/latest") + "/bccl/xml/topo_2c.xml"

import megatron_supa
