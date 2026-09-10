"""CLI for independently registered datasets and real model implementations."""
import argparse,json
from pathlib import Path
from framework.registry import manifest,load_dataset
from framework.experiments import train_experiment,evaluate_artifact
from framework.contracts import EventDataset

def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('list',help='List model and dataset implementations and capabilities.')
    for command in ('prepare','train','evaluate'):
        child=sub.add_parser(command);child.add_argument('--config',required=True,type=Path)
        child.add_argument('--output',required=True,type=Path)
        if command=='evaluate':child.add_argument('--artifact',required=True,type=Path);child.add_argument('--partition',choices=['all','train','validation','test'],default='all')
    args=parser.parse_args()
    try:
        if args.command=='list':
            for kind in ('models','datasets'):
                for entry in manifest(kind)[kind]:print(kind,entry['id'],entry.get('execution',entry.get('schema')),entry.get('status',entry.get('origin')))
            return
        config=json.loads(args.config.read_text());base=args.config.resolve().parent
        if args.command=='train':
            _,result=train_experiment(config,args.output,base);print(json.dumps(result['metrics'],indent=2));return
        if args.output.exists():raise ValueError('Output already exists; choose a new file.')
        if args.command=='prepare':
            data=load_dataset(config,base)
            if not isinstance(data,EventDataset):raise ValueError('prepare exports payment-event datasets; numeric CSV feeds the experiment runner directly.')
            output=data.document
        else:output=evaluate_artifact(args.artifact,config,base,args.partition)
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n');print(args.output)
    except ImportError as error:
        parser.exit(2,str(error)+'; install requirements-temporal.txt for graph models or requirements-models.txt for tabular models.\n')
    except (ValueError,KeyError,RuntimeError,FileNotFoundError) as error:parser.exit(2,str(error)+'\n')

if __name__=='__main__':main()
