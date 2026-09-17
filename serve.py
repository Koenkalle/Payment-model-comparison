"""Serve dataset preparation, model training and comparison on localhost."""
import argparse
import json
from pathlib import Path

from framework.comparison_service import (ComparisonService, DEFAULT_ARTIFACT,
                                          DEFAULT_SUPERVISED_ARTIFACT, make_server)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1', help='Loopback bind address (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--artifact', type=Path, default=DEFAULT_ARTIFACT,
                        help='Native artifact directory containing model.npz and manifest.json.')
    parser.add_argument('--calibration-config', type=Path,
                        help='Historical calibration config (default: ARTIFACT/calibration.config.json).')
    parser.add_argument('--supervised-artifact', type=Path, default=DEFAULT_SUPERVISED_ARTIFACT,
                        help='Fraud-label-trained artifact for the existing supervised comparison mode.')
    parser.add_argument('--dataset-config', type=Path, action='append', default=[],
                        help='Add a local fraud_dataset/prepared_fraud config to the web dataset picker (repeatable).')
    from framework.pipeline_service import DEFAULT_PIPELINE_DIR, PipelineService
    parser.add_argument('--pipeline-dir', type=Path, default=DEFAULT_PIPELINE_DIR,
                        help='Persistent dataset, job and trained artifact directory (default: artifacts/web-pipeline).')
    args = parser.parse_args()
    pipeline = None
    try:
        service = ComparisonService(args.artifact, args.calibration_config,
                                    supervised_artifact=args.supervised_artifact)
        from framework.dataset_service import DatasetService
        pipeline = PipelineService(args.pipeline_dir, config_paths=args.dataset_config)
        replay_configs = [path for path in args.dataset_config
                          if json.loads(path.read_text(encoding='utf-8')).get('loader')
                          in ('fraud_dataset', 'prepared_fraud')]
        server = make_server(service, args.host, args.port,
                             dataset_service=DatasetService(replay_configs), pipeline_service=pipeline)
    except (OSError, ValueError) as error:
        if pipeline is not None:
            pipeline.close()
        parser.exit(2, str(error) + '\n')
    address = '[' + args.host + ']' if ':' in args.host else args.host
    print(f'Comparison: http://{address}:{server.server_port}/index.html', flush=True)
    print(f'Data lab: http://{address}:{server.server_port}/data-lab.html', flush=True)
    print(f'Model trainer: http://{address}:{server.server_port}/trainer.html', flush=True)
    print(f'Saved datasets and experiments: {pipeline.root}. Ctrl+C stops the server.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        pipeline.close()


if __name__ == '__main__':
    main()
