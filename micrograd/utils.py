import logging
from graphviz import Digraph

format='%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(name)s:%(message)s'
logging.basicConfig(
    level=logging.INFO,
    format=format,
    handlers=[
        logging.FileHandler('logdir/logs.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        nodes.add(v)
        for child in v._prev:
            edges.add((v, child))
            build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    nodes, edges = trace(root)
    dot = Digraph(format='svg', graph_attr={'rankdir':'LR'})
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{ %s | data %.4f | grad %.4f}"%(n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid+n._op, label=n._op)
            dot.edge(uid+n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n2)), str(id(n1))+n1._op)
    return dot