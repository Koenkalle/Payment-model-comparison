"""Serve the comparison UI with native inference using its normal settings."""
import argparse
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
    args = parser.parse_args()
    try:
        service = ComparisonService(args.artifact, args.calibration_config,
                                    supervised_artifact=args.supervised_artifact)
        from framework.dataset_service import DatasetService
        server = make_server(service, args.host, args.port, dataset_service=DatasetService(args.dataset_config))
    except (OSError, ValueError) as error:
        parser.exit(2, str(error) + '\n')
    address = '[' + args.host + ']' if ':' in args.host else args.host
    print(f'Comparison: http://{address}:{server.server_port}/index.html', flush=True)
    print('Select a scenario or import a dataset, then choose a model in the page. Ctrl+C stops the server.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
