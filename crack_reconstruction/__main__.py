"""Predict from an explicitly prepared, aligned geometry/probability NPZ."""
import argparse,json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,help='NPZ: xyz, normals, donor_probabilities, native_probabilities')
    p.add_argument('--ranker',required=True,help='Externally fitted ranker JSON (eleven features)')
    p.add_argument('--output',required=True,help='New graph JSON path; existing files are never overwritten')
    p.add_argument('--domain',choices=['crackstructures','crackensembles'],default='crackstructures')
    args=p.parse_args()
    import numpy as np
    from .pipeline import reconstruct
    if args.domain=='crackensembles':
        from .normalized import reconstruct_normalized as reconstruct
    from .ranker import LogisticRanker
    ranker=LogisticRanker.from_dict(json.loads(Path(args.ranker).read_text(encoding='utf-8')))
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    with np.load(args.input,allow_pickle=False) as a:
        graph=reconstruct(a['xyz'],a['normals'],a['donor_probabilities'],a['native_probabilities'],ranker)
    nodes=list(graph);idx={n:i for i,n in enumerate(nodes)}
    data={'points':[np.asarray(graph.nodes[n]['pos'],float).tolist() for n in nodes],
          'edges':[[idx[a],idx[b]] for a,b in graph.edges],
          'coordinate_units':'same units as prepared input; see dataset protocol'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:json.dump(data,f,allow_nan=False)
    print(json.dumps({'nodes':len(nodes),'edges':graph.number_of_edges(),'output':str(output)}))

if __name__=='__main__':main()
